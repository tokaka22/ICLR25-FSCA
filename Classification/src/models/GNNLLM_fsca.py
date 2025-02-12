from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from models.GNNLLM_utils_fsca import GPT2withGNN, build_edges, construct_sequence, expand_edges, norm_weight, cosine_similarity_edges, expand_mask

from models.RevIN import RevIN


class GNNLLM_fsca(nn.Module):
    
    def __init__(self, config, data):
        super(GNNLLM_fsca, self).__init__()

        from types import SimpleNamespace
        configs = SimpleNamespace(**config)

        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        # self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])

        # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2withGNN.from_pretrained('gpt2', config=gpt2_config, args=configs)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.device = device
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        
        ####################################
        ######### prompt #########
        ####################################
        self.config = config
        ### tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True
        )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        ### task_prompt
        # self.categories = config['label_values']
        task_prompt = f"Predict category({len(config['label_values'])} in total) using previous data:"
        task_prompt_tok = self.tokenizer(task_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.task_prompt_embeddings = self.gpt2.get_input_embeddings()(task_prompt_tok.to(self.device))  # (batch, prompt_token, dim)
        
        ### prompt format
        fix_prompt = f"Predict category({len(config['label_values'])} in total) using previous data:"
        fix_prompt_tok = self.tokenizer(fix_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.fix_prompt_embeddings = self.gpt2.get_input_embeddings()(fix_prompt_tok.to(self.device))  # (1, L, dim)

        self.edge_expand_num = configs.batch_size * 1
        self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)
        self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)

        self.fix_prompt_len = self.fix_prompt_embeddings.shape[1]
        self.total_len = self.patch_num + self.fix_prompt_len

        self.ln_proj = nn.LayerNorm(configs.d_ff * self.total_len)
        self.out_layer = nn.Linear(configs.d_ff * self.total_len, self.num_classes)

        self.d_ff = configs.d_ff

        ####################################
        ######### graph part #########
        ####################################

        ### fine scale graph
        self.sequence = construct_sequence(self.patch_num, fix_prompt_tok.shape[1], configs.split_len, task_prompt_tok.shape[1])
        edges, self.edge1_mask_ori = build_edges(self.sequence)
        len_edge1 = len(self.edge1_mask_ori)
        self.edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=device)

        assert len(self.sequence) == self.total_len 

        self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_len)
        self.B_edge_all = self.B_edge_all.to(device=device)

        ### coarse scale graph
        self.l_sequence = np.arange(1, max(self.sequence)+1)
        l_edges, self.l_edge1_mask_ori = build_edges(self.l_sequence)
        # len_l_edge1 = len(self.l_edge1_mask_ori)
        self.l_edges = torch.tensor(l_edges, dtype=torch.long).t().contiguous()

        self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
        self.l_B_edge_all = self.l_B_edge_all.to(device=device)

        self.len_s = self.patch_num
        self.large_layer1 = nn.Linear(self.len_s * configs.d_model, configs.d_model)
        self.len_p = fix_prompt_tok.shape[1]
        self.large_layer0 = nn.Linear(self.len_p * configs.d_model, configs.d_model)

        # Get indices for 1
        self.indices_1 = np.where((self.sequence == 1))[0]

        # Get indices for 2
        self.indices_2 = np.where((self.sequence == 2))[0]

        self.l2s = nn.Linear(configs.d_l_comp, configs.d_model)

        if configs.w_l2s_flag:
            self.w_l2s = nn.Parameter(torch.tensor(0.5, dtype=torch.float32).to(device=device))
        else:
            self.w_l2s = torch.tensor(configs.w_l2s_v, dtype=torch.float32).to(device=device)
        
        # length
        L_f = len(self.sequence)
        L_l = len(self.l_sequence)

        # allocation
        allocation_matrix = np.zeros((L_f, L_l), dtype=int)
        for i in range(L_f):
            for j in range(L_l):
                if self.sequence[i] == self.l_sequence[j]:
                    allocation_matrix[i, j] = 1

        column_sums = allocation_matrix.sum(axis=0)
        normalized_allocation_matrix = allocation_matrix / column_sums

        self.allocation_tensor = torch.tensor(normalized_allocation_matrix, dtype=torch.float32)
        self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(device)

        self.revin_flag = configs.revin_flag
        if self.revin_flag == 1:
            self.revin_layer = RevIN(1).to(device)

        self.B = configs.batch_size
        # self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.maxof_l_sequence = int(max(self.l_sequence))
        pass

        
    def forward(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape

        if B != self.task_prompt_embeddings_expand.shape[0]:

            self.edge_expand_num = B * 1
            
            self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device) 

            self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device)

            self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=self.device)

            self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_len)
            self.B_edge_all = self.B_edge_all.to(device=self.device)
            
            self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
            self.l_B_edge_all = self.l_B_edge_all.to(device=self.device)

            self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(self.device)

        self.edge_expand_num = B * 1
        # self.edge_expand_num = B * M
        self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)
        self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        enc_out = self.enc_embedding(input_x, None)

        ######### splicing example prompts #########
        outputs = []
        outputs.append(torch.cat([enc_out, self.task_prompt_embeddings_expand], dim=1))
        outputs = torch.cat(outputs, dim=1) # print(outputs) BM, L, D

        ######### egdes
        # keep order with np.concatenate((edges_1_to_2, edges_4_to_5, edges_2_to_3))
        self.B_edge_weights = cosine_similarity_edges(outputs, self.B_edge_all)
        self.B_edge_weights_norm = norm_weight(self.B_edge_weights, self.edge1_mask)

        ######### coarse scale
        outputs_1 = outputs[:, self.indices_1]
        outputs_1 = outputs_1.reshape(self.edge_expand_num, -1, self.len_s * self.d_model).detach().clone()
        outputs_2 = outputs[:, self.indices_2]
        outputs_2 = outputs_2.reshape(self.edge_expand_num, -1, self.len_p * self.d_model).detach().clone()

        outputs_1 = self.large_layer1(outputs_1) # BM D
        outputs_2 = self.large_layer0(outputs_2)

        stacked = torch.stack((outputs_1, outputs_2), dim=2)  # B, 2, 
        outputs_l = stacked.reshape(self.edge_expand_num, self.maxof_l_sequence, 768)  # to B, L_large, 768

        ######### into gpt2
        outputs = self.gpt2(inputs_embeds=outputs, edge_index_all=self.B_edge_all, \
            edge_weight_all=self.B_edge_weights_norm, large_flag=0, \
            l_inputs_embeds = outputs_l, \
            edge_index_all_l = self.l_B_edge_all, \
            odd_indices = None, \
            even_indices = None, \
            edge_expand_num = self.edge_expand_num, \
            large_layer1 = self.large_layer1, \
            large_layer0 = self.large_layer0, \
            maxof_l_sequence = self.maxof_l_sequence, \
            afc = self.afc, \
            l2s = self.l2s, \
            len_s = self.len_s, \
            len_p = self.len_p,
            w_l2s = self.w_l2s
            ).last_hidden_state

        ######### output layer
        outputs = outputs[:, :, :self.d_ff]
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs
    
    

    

