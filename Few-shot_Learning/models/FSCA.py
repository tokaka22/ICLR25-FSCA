import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from models.utils.LLMwithGNN import GPT2withGNN, build_edges, construct_sequence, expand_edges, norm_weight, cosine_similarity_edges, expand_mask
from models.utils.RevIN import RevIN


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
    

class FSCA(nn.Module):
    
    def __init__(self, configs, device):
        super(FSCA, self).__init__()
        self.device = device
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        # assert self.patch_num % configs.split_len == 0
        
        if configs.is_gpt:
            if configs.pretrain:
                config = GPT2Config.from_pretrained('gpt2')
                self.gpt2 = GPT2withGNN.from_pretrained('gpt2', config=config, args=configs)

                if configs.freeze and configs.pretrain:
                    for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                        if 'ln' in name or 'wpe' in name:
                            param.requires_grad = True
                        elif 'gnn' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

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

            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
            
        # Embedding
        self.enc_embedding = DataEmbedding_wo_time(self.patch_size, 
                                            configs.d_model, configs.embed, configs.freq, configs.in_dropout)
        
        # ref TimeLLM
        self.d_ff = configs.d_ff
        self.head_nf = self.d_ff * self.patch_num
        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                                 head_dropout=configs.out_dropout)
        
        

        for layer in (self.gpt2, self.enc_embedding, self.output_projection):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

        self.top_k = 5
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        ### record
        self.split_len = configs.split_len
        self.p_interval = self.patch_num // self.split_len

        ### task_prompt
        self.edge_expand_num = configs.batch_size * configs.enc_in

        task_prompt = f"Predict future sequences using previous data:"
        task_prompt_tok = self.tokenizer(task_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.task_prompt_embeddings = self.gpt2.get_input_embeddings()(task_prompt_tok.to(self.device))  # (batch, prompt_token, dim)
        self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(device) 

        test = "Predict"
        test_tokens = self.tokenizer("Future", return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        decoded_text = self.tokenizer.decode(test_tokens[0])

        ### prompt format
        fix_prompt = f"Predict future sequences using previous data:"
        fix_prompt_tok = self.tokenizer(fix_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        self.fix_prompt_embeddings = self.gpt2.get_input_embeddings()(fix_prompt_tok.to(self.device))  # (1, L, dim)
        self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(device)

        ### graph
        self.sequence = construct_sequence(self.patch_num, fix_prompt_tok.shape[1], configs.split_len, task_prompt_tok.shape[1])
        edges1, edges2, self.edge1_mask_ori = build_edges(self.sequence)
        len_edge1 = len(self.edge1_mask_ori)
        self.edges = torch.tensor(edges1, dtype=torch.long).t().contiguous()
        
        self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=device)

        #TODO eg. 4 * 5
        self.total_prompt_len = self.patch_num + fix_prompt_tok.shape[1] * (self.split_len -1 ) + task_prompt_tok.shape[1] 
        assert len(self.sequence) == self.total_prompt_len 

        self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_prompt_len)
        self.B_edge_all = self.B_edge_all.to(device=device)

        ### large scale graph
        self.l_sequence = np.arange(1, max(self.sequence)+1)
        l_edges1, _, self.l_edge1_mask_ori = build_edges(self.l_sequence)
        self.l_edges = torch.tensor(l_edges1, dtype=torch.long).t().contiguous()

        self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
        self.l_B_edge_all = self.l_B_edge_all.to(device=device)

        self.len_s = self.patch_num // configs.split_len
        if self.patch_num % configs.split_len != 0:
            self.len_s += 1
        self.large_layer1 = nn.Linear(self.len_s * configs.d_model, configs.d_model)
        self.len_p = fix_prompt_tok.shape[1]
        self.large_layer0 = nn.Linear(self.len_p * configs.d_model, configs.d_model)

        self.odd_indices = np.where(self.sequence % 2 != 0)[0]
        self.even_indices = np.where(self.sequence % 2 == 0)[0]

        self.l2s = nn.Linear(configs.d_l_comp, configs.d_model)

        if configs.w_l2s_flag:
            self.w_l2s = nn.Parameter(torch.tensor(0.5, dtype=torch.float32).to(device=device))
        else:
            self.w_l2s = torch.tensor(configs.w_l2s_v, dtype=torch.float32).to(device=device)
        
        for layer in (self.large_layer1, self.large_layer0, self.l2s):
            layer.to(device=device)
            layer.train()

        # lengths
        L_f = len(self.sequence)
        L_l = len(self.l_sequence)

        # allocation matrix
        allocation_matrix = np.zeros((L_f, L_l), dtype=int)
        for i in range(L_f):
            for j in range(L_l):
                if self.sequence[i] == self.l_sequence[j]:
                    allocation_matrix[i, j] = 1

        # norm
        column_sums = allocation_matrix.sum(axis=0)
        normalized_allocation_matrix = allocation_matrix / column_sums
        
        # to Tensor
        self.allocation_tensor = torch.tensor(normalized_allocation_matrix, dtype=torch.float32)
        self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(device)

        self.revin_flag = configs.revin_flag
        if self.revin_flag == 1:
            self.revin_layer = RevIN(1).to(device)

        self.B = configs.batch_size
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.maxof_l_sequence = int(max(self.l_sequence))
        pass

    def segment_points_variable(self, len, split_num):
        return [i * (len // split_num) for i in range(1, split_num)]

    def get_correspond_posi(self, ori, stride, patch_size):
        return ori * stride + patch_size
    
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1) # B,M,L
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    

    def forward(self, x, itr):
        B, L, M = x.shape # torch.Size([32, 720, 1])
        
        if B != self.B: # for test wo drop_last
            self.edge_expand_num = B * self.enc_in
            
            self.task_prompt_embeddings_expand = self.task_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device) 

            self.fix_prompt_embeddings_expand = self.fix_prompt_embeddings.float().expand(self.edge_expand_num, -1, -1).to(self.device)

            self.edge1_mask = expand_mask(torch.tensor(self.edge1_mask_ori), self.edge_expand_num).to(device=self.device)

            self.B_edge_all = expand_edges(self.edges, self.edge_expand_num, self.total_prompt_len)
            self.B_edge_all = self.B_edge_all.to(device=self.device)
            
            self.l_B_edge_all = expand_edges(self.l_edges, self.edge_expand_num, len(self.l_sequence))
            self.l_B_edge_all = self.l_B_edge_all.to(device=self.device)

            self.afc = self.allocation_tensor.unsqueeze(0).expand(self.edge_expand_num, -1, -1).to(self.device)

        # assert M == 1

        if self.revin_flag == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev

        x = x.permute(0, 2, 1).contiguous() # B,M,L

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # B,M,N,P
        x = rearrange(x, 'b m n p -> (b m) n p') # (B,M),N,P

        enc_out = self.enc_embedding(x) # (b m) n d

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

        # keep same order with torch.cat((edges1, edges2), dim=1)
        self.B_edge_weights = cosine_similarity_edges(outputs, self.B_edge_all)
        self.B_edge_weights_norm = norm_weight(self.B_edge_weights, self.edge1_mask)

        # large scale
        outputs_odd = outputs[:, self.odd_indices]
        if outputs_odd.shape[1] % 2 != 0:
            padding = torch.zeros((outputs_odd.shape[0], 1, outputs_odd.shape[2]), dtype=outputs_odd.dtype, device=outputs_odd.device)
            outputs_odd = torch.cat([outputs_odd, padding], dim=1)

        outputs1 = outputs_odd.reshape(self.edge_expand_num, -1, self.len_s * self.d_model).detach().clone()
        outputs0 = outputs[:, self.even_indices].reshape(self.edge_expand_num, -1, self.len_p * self.d_model).detach().clone()

        outputs1 = self.large_layer1(outputs1) # BM D
        outputs0 = self.large_layer0(outputs0)

        stacked = torch.stack((outputs1, outputs0), dim=2)  # eg B, 2, 2, 768
        outputs_l = stacked.reshape(self.edge_expand_num, self.maxof_l_sequence, 768)  # eg B, L_large, 768

        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs, edge_index_all=self.B_edge_all, \
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
        outputs = outputs[:, :, :self.d_ff]
        outputs = torch.reshape(
            outputs, (-1, M, outputs.shape[-2], outputs.shape[-1]))
        outputs = outputs.permute(0, 1, 3, 2).contiguous()
        outputs = self.output_projection(outputs[:, :, :, -self.patch_num:])
        outputs = outputs.permute(0, 2, 1).contiguous()

        if self.revin_flag == 1:
            outputs = self.revin_layer(outputs, 'denorm')
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs
