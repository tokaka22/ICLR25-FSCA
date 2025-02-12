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
from models.GNNLLM_utils_vca import GPT2withGNN, build_edges, construct_sequence, expand_edges, norm_weight, cosine_similarity_edges, expand_mask

from models.RevIN import RevIN


class GNNLLM_vca(nn.Module):
    
    def __init__(self, config, data):
        super(GNNLLM_vca, self).__init__()

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
        
        self.config = config
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

        fix_prompt = f"Predict category({len(config['label_values'])} in total) using previous data:"
        fix_prompt_tok = self.tokenizer(fix_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.fix_prompt_embeddings = self.gpt2.get_input_embeddings()(fix_prompt_tok.to(self.device))  # (1, L, dim)

        self.edge_expand_num = configs.batch_size * 1
        # self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)
        self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1)

        self.fix_prompt_len = self.fix_prompt_embeddings.shape[1]
        self.total_len = self.patch_num * 3 + self.fix_prompt_len * 3 + 2

        self.is_valid_len = configs.is_valid_len
        if configs.is_valid_len:    
            self.valid_len = self.patch_num
            self.ln_proj = nn.LayerNorm(configs.d_ff * self.valid_len)
            self.out_layer = nn.Linear(configs.d_ff * self.valid_len, self.num_classes)
        else:
            self.ln_proj = nn.LayerNorm(configs.d_ff * self.total_len)
            self.out_layer = nn.Linear(configs.d_ff * self.total_len, self.num_classes)
        

        self.d_ff = configs.d_ff

        self.sequence = construct_sequence(self.patch_num, fix_prompt_tok.shape[1], configs.split_len)
        edges, self.edge1_mask_ori = build_edges(self.sequence)
        len_edge1 = len(self.edge1_mask_ori)
        self.edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=device)

        assert len(self.sequence) == self.total_len 

        self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_len)
        self.B_edge_all = self.B_edge_all.to(device=device)

        self.l_sequence = np.arange(1, max(self.sequence)+1)
        l_edges, self.l_edge1_mask_ori = build_edges(self.l_sequence)
        # len_l_edge1 = len(self.l_edge1_mask_ori)
        self.l_edges = torch.tensor(l_edges, dtype=torch.long).t().contiguous()

        self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
        self.l_B_edge_all = self.l_B_edge_all.to(device=device)

        self.len_s = self.patch_num
        self.large_layer_ts = nn.Linear(self.len_s * configs.d_model, configs.d_model)
        self.len_p = fix_prompt_tok.shape[1]
        self.large_layer_p = nn.Linear(self.len_p * configs.d_model, configs.d_model)

        # Get indices
        self.indices_e0 = np.where((self.sequence == 1))[0]
        self.indices_e1 = np.where((self.sequence == 4))[0]
        self.indices_x = np.where((self.sequence == 7))[0]

        self.indices_l0 = np.where((self.sequence == 3))[0]
        self.indices_l1 = np.where((self.sequence == 6))[0]

        self.l2s = nn.Linear(configs.d_l_comp, configs.d_model)

        if configs.w_l2s_flag:
            self.w_l2s = nn.Parameter(torch.tensor(0.5, dtype=torch.float32).to(device=device))
        else:
            self.w_l2s = torch.tensor(configs.w_l2s_v, dtype=torch.float32).to(device=device)
        
        L_f = len(self.sequence)
        L_l = len(self.l_sequence)

        allocation_matrix = np.zeros((L_f, L_l), dtype=int)
        for i in range(L_f):
            for j in range(L_l):
                if self.sequence[i] == self.l_sequence[j]:
                    allocation_matrix[i, j] = 1

        column_sums = allocation_matrix.sum(axis=0)
        assert column_sums.sum() == self.total_len 
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

        self.categories = config['label_values']
        inputs = self.tokenizer.batch_encode_plus([str(cat) for cat in self.categories], return_tensors='pt', padding=True)['input_ids'].to(self.device)

        assert inputs.shape[0] == len(self.categories)
         
        with torch.no_grad():
            categories_out = self.gpt2.get_input_embeddings()(inputs)[:, 0, :]
        self.categories_embeddings = nn.Embedding.from_pretrained(categories_out, freeze=True)
        pass

        
    def forward(self, examples, examples_labels, x_enc, x_mark_enc):
        B, L, M = x_enc.shape

        if B != self.fix_prompt_embeddings_expand.shape[0]:

            self.edge_expand_num = B * 1
            
            # # self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device) 

            self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device)

            self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=self.device)

            self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_len)
            self.B_edge_all = self.B_edge_all.to(device=self.device)
            
            self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
            self.l_B_edge_all = self.l_B_edge_all.to(device=self.device)

            self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(self.device)

        examples_label_emds = []
        for examples_label in examples_labels:
            examples_label_emd = self.categories_embeddings(examples_label.to(torch.int64).cuda())
            examples_label_emd = examples_label_emd.unsqueeze(0).repeat(B, 1, 1)
            examples_label_emds.append(examples_label_emd)

        examples = [example.unsqueeze(0).repeat(B, 1, 1) for example in examples]
        x_all = examples + [x_enc]
        len_ts = len(x_all)
        x_all = torch.cat(x_all, dim=0) # B L_all D

        input_x = rearrange(x_all, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        enc_out_all = self.enc_embedding(input_x, None)

        assert 3 == len_ts
        enc_out_all = enc_out_all.reshape(3, -1, enc_out_all.shape[-2], enc_out_all.shape[-1])
        enc_out_e0 = enc_out_all[0]
        enc_out_e1 = enc_out_all[1]
        enc_out = enc_out_all[2]

        # outputs = []
        # outputs.append()
        outputs = torch.cat([enc_out_e0, self.fix_prompt_embeddings_expand, examples_label_emds[0], \
                                  enc_out_e1, self.fix_prompt_embeddings_expand, examples_label_emds[1], \
                                  enc_out, self.fix_prompt_embeddings_expand,], dim=1) # print(outputs) BM, L, D

        self.B_edge_weights = cosine_similarity_edges(outputs, self.B_edge_all)
        self.B_edge_weights_norm = norm_weight(self.B_edge_weights, self.edge1_mask)

        # example 0
        outputs_e0 = outputs[:, self.indices_e0]
        outputs_e0 = outputs_e0.reshape(self.edge_expand_num, -1, self.len_s * self.d_model).detach().clone()

        # example 1
        outputs_e1 = outputs[:, self.indices_e1]
        outputs_e1 = outputs_e1.reshape(self.edge_expand_num, -1, self.len_s * self.d_model).detach().clone()

        # x
        outputs_x = outputs[:, self.indices_x]
        outputs_x = outputs_x.reshape(self.edge_expand_num, -1, self.len_s * self.d_model).detach().clone()

        # p 
        outputs_p = self.fix_prompt_embeddings_expand
        outputs_p = outputs_p.reshape(self.edge_expand_num, -1, self.len_p * self.d_model).detach().clone()

        outputs_ts = torch.cat([outputs_e0, outputs_e1, outputs_x], dim=0)

        outputs_ts = self.large_layer_ts(outputs_ts) # BM D
        outputs_p = self.large_layer_p(outputs_p)

        outputs_ts_splits = outputs_ts.reshape(3, -1, outputs_ts.shape[-2], outputs_ts.shape[-1])
        outputs_e0 = outputs_ts_splits[0]
        outputs_e1 = outputs_ts_splits[1]
        outputs_x = outputs_ts_splits[2]

        # l0
        outputs_l0 = outputs[:, self.indices_l0]
        # l1
        outputs_l1 = outputs[:, self.indices_l1]

        outputs_l = torch.cat([outputs_e0, outputs_p, outputs_l0, \
                               outputs_e1, outputs_p, outputs_l1, \
                               outputs_x, outputs_p,], dim=1) # print(outputs) BM, L, D

        outputs = self.gpt2(inputs_embeds=outputs, edge_index_all=self.B_edge_all, \
            edge_weight_all=self.B_edge_weights_norm, large_flag=0, \
            l_inputs_embeds = outputs_l, \
            edge_index_all_l = self.l_B_edge_all, \
            odd_indices = None, \
            even_indices = None, \
            edge_expand_num = self.edge_expand_num, \
            large_layer_ts = self.large_layer_ts, \
            large_layer_p = self.large_layer_p, \
            maxof_l_sequence = self.maxof_l_sequence, \
            afc = self.afc, \
            l2s = self.l2s, \
            len_s = self.len_s, \
            len_p = self.len_p,
            w_l2s = self.w_l2s
            ).last_hidden_state

        if self.is_valid_len:    
            outputs = outputs[:, :self.valid_len, :self.d_ff]
        else:
            outputs = outputs[:, :, :self.d_ff]
        
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs
    
    

    

