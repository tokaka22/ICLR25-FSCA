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
from layers.Embed import DataEmbedding, DataEmbedding_wo_time

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from models.LLM.utils.LLMwithGNN import GPT2withGNN, build_edges, construct_sequence, expand_edges, norm_weight, cosine_similarity_edges, expand_mask
from models.LLM.utils.RevIN import RevIN


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    
    def __init__(self, configs):
        ########## most are original part ##########
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        ############## input/output layers
        # self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)

        # Embedding uses TCN
        self.enc_embedding = DataEmbedding_wo_time(self.patch_size, 
                                            configs.d_model, configs.embed, configs.freq, configs.in_dropout)
        
        # self.out_layer = nn.Linear(configs.d_model * (self.patch_num + 232), configs.pred_len)

        # ref TimeLLM
        self.d_ff = configs.d_ff
        self.head_nf = self.d_ff * self.patch_num
        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                                 head_dropout=configs.out_dropout)


        config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2withGNN.from_pretrained('gpt2', config=config, args=configs)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.device = device
            self.gpt2.to(device=device)
            
        ########## GNN settings ##########
        self.split_len = configs.split_len
        self.edge_expand_num = configs.batch_size * configs.enc_in
        self.p_interval = self.patch_num // self.split_len

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
        self.edge_expand_num = configs.batch_size * configs.enc_in

        task_prompt = f"Predict future sequences using previous data:"
        task_prompt_tok = self.tokenizer(task_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.task_prompt_embeddings = self.gpt2.get_input_embeddings()(task_prompt_tok.to(self.device))  # (batch, prompt_token, dim)
        self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(device) 

        ### fixed prompt format
        fix_prompt = f"Predict future sequences using previous data:"
        fix_prompt_tok = self.tokenizer(fix_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.fix_prompt_embeddings = self.gpt2.get_input_embeddings()(fix_prompt_tok.to(self.device))  # (1, L, dim)
        self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(device)

        ### small scale graph
        self.sequence = construct_sequence(self.patch_num, fix_prompt_tok.shape[1], configs.split_len, task_prompt_tok.shape[1])
        edges1, edges2, self.edge1_mask_ori = build_edges(self.sequence)
        len_edge1 = len(self.edge1_mask_ori)
        self.edges = torch.tensor(edges1, dtype=torch.long).t().contiguous()
        
        self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=device)

        self.total_prompt_len = self.patch_num + fix_prompt_tok.shape[1] * (self.split_len -1 ) + task_prompt_tok.shape[1] 
        assert len(self.sequence) == self.total_prompt_len 

        self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_prompt_len)
        self.B_edge_all = self.B_edge_all.to(device=device)

        ### large scale graph
        self.l_sequence = np.arange(1, max(self.sequence)+1)
        l_edges1, _, self.l_edge1_mask_ori = build_edges(self.l_sequence)
        # len_l_edge1 = len(self.l_edge1_mask_ori)
        self.l_edges = torch.tensor(l_edges1, dtype=torch.long).t().contiguous()
        # self.l_edge1_mask = expand_mask(torch.tensor(self.l_edge1_mask_ori), self.edge_expand_num).to(device=device)

        self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
        self.l_B_edge_all = self.l_B_edge_all.to(device=device)

        self.len_s = self.patch_num // configs.split_len
        #TODO gpt2 default demension - 768
        self.llm_d = 768
        if self.patch_num % configs.split_len != 0:
            self.len_s += 1
        self.large_layer1 = nn.Linear(self.len_s * self.llm_d, self.llm_d) # odd
        self.len_p = fix_prompt_tok.shape[1]
        self.large_layer0 = nn.Linear(self.len_p * self.llm_d, self.llm_d) # even

        self.odd_indices = np.where(self.sequence % 2 != 0)[0]
        self.even_indices = np.where(self.sequence % 2 == 0)[0]

        self.l2s = nn.Linear(configs.d_l_comp, self.llm_d)

        if configs.w_l2s_flag:
            self.w_l2s = nn.Parameter(torch.tensor(0.5, dtype=torch.float32).to(device=device))
        else:
            self.w_l2s = torch.tensor(configs.w_l2s_v, dtype=torch.float32).to(device=device)
        
        for layer in (self.large_layer1, self.large_layer0, self.l2s):
            layer.to(device=device)
            layer.train()

        L_f = len(self.sequence)
        L_l = len(self.l_sequence)

        allocation_matrix = np.zeros((L_f, L_l), dtype=int)
        for i in range(L_f):
            for j in range(L_l):
                if self.sequence[i] == self.l_sequence[j]:
                    allocation_matrix[i, j] = 1

        self.allocation_tensor = torch.tensor(allocation_matrix, dtype=torch.float32)
        self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(device)
        
        self.B = configs.batch_size
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.maxof_l_sequence = int(max(self.l_sequence))
        
        #TODO Originally integrated with multiple tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.predict_linear = nn.Linear(self.patch_size, configs.enc_in)
            self.ln = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out)
       

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, test=0):
        #TODO Originally integrated with multiple tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, test)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, test=0):
        B, L, M = x_enc.shape
        
        if test == 1 or B * self.enc_in != self.edge_expand_num:
            self.edge_expand_num = B * self.enc_in
            
            self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device) 

            self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device)

            self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=self.device)

            self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_prompt_len)
            self.B_edge_all = self.B_edge_all.to(device=self.device)
            
            self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
            self.l_B_edge_all = self.l_B_edge_all.to(device=self.device)

            self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(self.device)

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1).contiguous() # B,M,L

        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p') 

        enc_out = self.enc_embedding(x_enc) # (b m) n d
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        ####### small scale #######
        outputs = []
        for i in range(self.split_len):

            begin = i * self.p_interval
            end = self.patch_num if i == (self.split_len - 1) else (i + 1) * self.p_interval

            if i == (self.split_len - 1):
                outputs.append(torch.cat([enc_out[:, begin : end]], dim=1))
            else:
                outputs.append(torch.cat([enc_out[:, begin : end], self.fix_prompt_embeddings_expand], dim=1))

        outputs.append(self.task_prompt_embeddings_expand)

        outputs = torch.cat(outputs, dim=1) # print(outputs) BM, L, D

        # keep the order with torch.cat((edges1, edges2), dim=1)
        self.B_edge_weights = cosine_similarity_edges(outputs, self.B_edge_all)
        self.B_edge_weights_norm = norm_weight(self.B_edge_weights, self.edge1_mask)

        ####### large sacle #######
        # large scale
        outputs_odd = outputs[:, self.odd_indices]
        if outputs_odd.shape[1] % 2 != 0:
            padding = torch.zeros((outputs_odd.shape[0], 1, outputs_odd.shape[2]), dtype=outputs_odd.dtype, device=outputs_odd.device)
            outputs_odd = torch.cat([outputs_odd, padding], dim=1)

        outputs1 = outputs_odd.reshape(self.edge_expand_num, -1, self.len_s * self.llm_d).detach().clone()
        outputs0 = outputs[:, self.even_indices].reshape(self.edge_expand_num, -1, self.len_p * self.llm_d).detach().clone()

        outputs1 = self.large_layer1(outputs1) # BM D
        outputs0 = self.large_layer0(outputs0)

        stacked = torch.stack((outputs1, outputs0), dim=2)  # eg. B, 2, 2, 768
        outputs_l = stacked.reshape(self.edge_expand_num, self.maxof_l_sequence, 768)  # B, L_large, 768

        ####### LLMwithGNN #######
        dec_out = self.gpt2(inputs_embeds=outputs, edge_index_all=self.B_edge_all, \
                edge_weight_all=self.B_edge_weights_norm, large_flag=0, \
                l_inputs_embeds = outputs_l, \
                edge_index_all_l = self.l_B_edge_all, \
                odd_indices = self.odd_indices, \
                even_indices = self.even_indices, \
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
        
        # ref TimeLLM
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, M, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_num:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = dec_out * stdev
        dec_out = dec_out + means
        
        return dec_out

   