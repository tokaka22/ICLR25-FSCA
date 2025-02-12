import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import *
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from torch_geometric.nn import SAGEConv, GCNConv, GATConv


class GPT2withGNN(GPT2Model):
    def __init__(self, config, args):
        super().__init__(config)
        self.n_layer = config.n_layer

        # You can use difference GNNs here
        self.gnn_layer = GCNConv(config.hidden_size, config.hidden_size, bias=True)
        self.gnn_layer_l = GCNConv(config.hidden_size, config.hidden_size, bias=True)

        self.gnn_layer_index = args.gnn_layer_index
        self.l_gnn_layer_index = args.l_gnn_layer_index

        self.d_model = args.d_model

        self.d_l_comp = args.d_l_comp

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        l_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        edge_index_all = None,
        edge_weight_all = None,
        edge_index_all_l = None,
        edge_weight_all_l = None,
        large_flag = 0,
        odd_indices = None,
        even_indices = None,
        edge_expand_num = None,
        large_layer1 = None,
        large_layer0 = None,
        maxof_l_sequence = None,
        afc = None,
        l2s = None,
        len_s = None,
        len_p = None,
        w_l2s = None    
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        l_input_shape = l_inputs_embeds.size()[:-1]
        l_position_ids = torch.arange(past_length, l_input_shape[-1] + past_length, dtype=torch.long, device=device)
        l_position_ids = l_position_ids.unsqueeze(0)
        l_position_embeds = self.wpe(l_position_ids)
        l_hidden_states = l_inputs_embeds + l_position_embeds
                    
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        l_hidden_states = self.drop(l_hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if i in self.gnn_layer_index:
                orig_size = hidden_states.size()
                hidden_states = hidden_states.view(-1, self.embed_dim) # release batch dimension
                hidden_states = self.gnn_layer(hidden_states, edge_index=edge_index_all, edge_weight=edge_weight_all)
                hidden_states = hidden_states.view(*orig_size)  

                if i in self.l_gnn_layer_index:
                    # GNN
                    l_orig_size = l_hidden_states.size()
                    l_hidden_states = l_hidden_states.view(-1, self.embed_dim) # release batch dimension
                    l_hidden_states = self.gnn_layer_l(l_hidden_states, edge_index=edge_index_all_l)
                    l_hidden_states = l_hidden_states.view(*l_orig_size)  

                    ### large scale to fine scale
                    # [B, L_large, 768] to [B, L_fine, 768]
                    # afc shape [B, L_fine, L_large]
                    l_hidden_states_compress = l_hidden_states[..., :self.d_l_comp]
                    hidden_states = hidden_states + w_l2s * l2s(torch.bmm(afc, l_hidden_states_compress))


            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                outputs_l = block(
                    l_hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]   
            l_hidden_states = outputs_l[0]
            
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        if len(self.h) in self.gnn_layer_index:
            orig_size = hidden_states.size()
            hidden_states = hidden_states.view(-1, self.embed_dim) # release batch dimension
            hidden_states = self.gnn_layer(hidden_states, edge_index=edge_index_all, edge_weight=edge_weight_all)
            hidden_states = hidden_states.view(*orig_size)  

            if len(self.h) in self.l_gnn_layer_index:
                # GNN
                l_orig_size = l_hidden_states.size()
                l_hidden_states = l_hidden_states.view(-1, self.embed_dim) # release batch dimension
                l_hidden_states = self.gnn_layer_l(l_hidden_states, edge_index=edge_index_all_l)
                l_hidden_states = l_hidden_states.view(*l_orig_size)  

                ### large scale to fine scale
                # [B, L_large, 768] to [B, L_fine, 768]
                # afc shape [B, L_fine, L_large]
                l_hidden_states_compress = l_hidden_states[..., :self.d_l_comp]
                hidden_states = hidden_states + w_l2s * l2s(torch.bmm(afc, l_hidden_states_compress))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


def build_edges(array):
    even_values = [x for x in set(array) if x % 2 == 0]
    even_last_indices = {x: np.max(np.where(array == x)[0]) for x in even_values}
    even_first_indices = {x: np.min(np.where(array == x)[0]) for x in even_values}

    odd_values = [x for x in set(array) if x % 2 != 0]
    odd_indices = {x: np.where(array == x)[0] for x in odd_values}

    edges1 = [] 
    edges2 = []

    edges1_idx_list = []

    edge1_mask = 0

    for even in sorted(even_values):
        for odd in odd_values:
            if odd < even:
                for odd_index in odd_indices[odd]:
                    edges1.append((odd_index, even_first_indices[even]))
                    edges1_idx_list.append(edge1_mask)
        edge1_mask+=1
                    
        if even < max(even_values):
            next_odd = even + 1
            if next_odd in odd_indices:
                for next_odd_index in odd_indices[next_odd]:
                    edges1.append((even_last_indices[even], next_odd_index))
                    edges1_idx_list.append(edge1_mask)
        edge1_mask+=1
                    
    edges1 = np.array(edges1, dtype=np.int64)
    edges2 = np.array(edges2, dtype=np.int64)
    return edges1, edges2, edges1_idx_list


def construct_sequence(L_p, L_fix_prompt, s, L_t):
    p_data = np.ones(L_p)
    
    f_data = np.ones(L_fix_prompt)
    
    t_data = np.ones(L_t)
    # assert L_p % s == 0
    interval = L_p // s
    
    result = np.array([])
    
    idx = 1
    for i in range(s):
        begin = i * interval
        end = L_p if i == (s - 1) else (i + 1) * interval

        result = np.concatenate((result, p_data[begin: end] * idx))
        idx += 1

        if i == (s - 1):
            pass
        else:
            result = np.concatenate((result, f_data * idx))
            idx += 1
    
    result = np.concatenate((result, t_data * idx))
    
    return result


def expand_mask(mask, B):
    # mask = torch.tensor([1, 1, 2, 3, 3], dtype=torch.int)
    batch_size = B

    max_val = mask.max()
    offsets = torch.arange(0, batch_size) * (max_val + 1)

    batch_masks = mask + offsets.unsqueeze(1)
    result = batch_masks.view(-1)

    return result


def expand_edges(edge_all, B, N):
    E = edge_all.shape[1]
    expanded_edges = edge_all.unsqueeze(0).repeat(B, 1, 1)  # (B, 2, |E|)
    offsets = torch.arange(B) * N
    offsets = offsets[:, None, None]  
    expanded_edges += offsets 

    expanded_edges = expanded_edges.permute(1, 0, 2).contiguous().reshape(2, -1) # (B, 2, |E|) -> (2, B * |E|)

    return expanded_edges


def cosine_similarity_edges(input, B_edge_all):
    B, V, D = input.shape
    _, BE = B_edge_all.shape  # Assuming B_edge_all has shape [2, BE]
    E = BE / B

    # Reshape edge indices for easy access
    # Start indices and end indices for all edges across the batch
    start_indices = B_edge_all[0, :].reshape(-1)  # Shape [BE]
    end_indices = B_edge_all[1, :].reshape(-1)    # Shape [BE]

    # Flatten the input to make it easier to index
    # Shape [B*V, D]
    flat_input = input.reshape(-1, D)

    # Gather start and end node features according to edge indices
    start_features = flat_input[start_indices]  # Shape [BE, D]
    end_features = flat_input[end_indices]      # Shape [BE, D]

    # Compute cosine similarity
    # Normalize the feature vectors to unit length
    start_features = torch.nn.functional.normalize(start_features, p=2, dim=1)
    end_features = torch.nn.functional.normalize(end_features, p=2, dim=1)

    # Dot product of normalized vectors (cosine similarity)
    cosine_sim = torch.sum(start_features * end_features, dim=1) + 1  # Shape [BE]

    return cosine_sim


def norm_weight(edge_weights, mask):
    # edge_weights = torch.tensor([12, 17, 71, 42, 28, 30, 55, 45], dtype=torch.float32)
    # mask = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3], dtype=torch.int)

    sums = torch.zeros(mask.max().item() + 1, device=edge_weights.device)
    sums.index_add_(0, mask, edge_weights)

    normalized_weights = edge_weights / sums[mask]
    # print("Normalized Weights:", normalized_weights)

    return normalized_weights