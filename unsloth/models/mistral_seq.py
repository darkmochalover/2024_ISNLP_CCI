# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .llama import *
from .llama_seq import *
import os
from ._utils import __version__
from .llama import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
    MistralForSequenceClassification
)
# For Pytorch 2.1.1
try:
    from transformers.models.mistral.modeling_mistral import (
        MistralSdpaAttention,
        MistralFlashAttention2,
    )
except:
    MistralSdpaAttention   = MistralAttention
    MistralFlashAttention2 = MistralAttention
pass


def MistralAttention_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    bool = False,
    use_cache:            bool = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention
    pass

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    assert(n_kv_heads * n_groups == n_heads)

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads,    head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # Extend RoPE dynamically to fit in VRAM
    self.rotary_emb.extend_rope_embedding(V, seq_len = kv_seq_len)

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len = kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)
    pass

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim = 2)
        V = torch.cat([past_key_value[1], V], dim = 2)
    pass
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and attention_mask is None):
        # Xformers memory efficient attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        K_M = V_M = bsz * kv_seq_len
        Q_M = bsz * q_len

        has_swa = isinstance(causal_mask, xformers.attn_bias.BlockDiagonalCausalMask)

        # Group query attention
        K = K  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        V = V  .view(bsz, kv_seq_len, n_kv_heads,        1, head_dim)
        K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
        if hidden_states.requires_grad:
            K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
            V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_heads, head_dim)
                K = K.view(1, K_M, n_heads, head_dim)
                V = V.view(1, V_M, n_heads, head_dim)
            pass
        else:
            # Xformers does support the forward pass though
            Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)

            if has_swa:
                Q = Q.view(1, Q_M, n_kv_heads, n_groups, head_dim)
                K = K.view(1, K_M, n_kv_heads, n_groups, head_dim)
                V = V.view(1, V_M, n_kv_heads, n_groups, head_dim)
            pass
        pass

        A = xformers_attention(Q, K, V, attn_bias = causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        sw = getattr(self.config, "sliding_window", None)
        sw = kv_seq_len if (sw is None or sw == "null") else sw
        window = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)
        A = flash_attn_func(Q, K, V, causal = True, window_size = window)
    else:
        # Grouped query attention
        # if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        # pass
        # Must be contiguous or else results are False!
        # https://github.com/pytorch/pytorch/issues/112577
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        A = scaled_dot_product_attention(Q, K, V, attn_mask = attention_mask, is_causal = False)
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()
    pass
    
    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value
pass


def MistralForSequenceClassification_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, SequenceClassifierOutputWithPast]:

    # if causal_mask is None and past_key_values is None:
    #     bsz, q_len = input_ids.shape
    #     sliding_window = getattr(self.config, "sliding_window", None)
    #     if sliding_window is None or sliding_window == "null" or sliding_window <= 0:
    #         causal_mask = xformers.attn_bias.LowerTriangularMask()
    #     elif q_len <= sliding_window:
    #         causal_mask = xformers.attn_bias.LowerTriangularMask()
    #     else:
    #         # Fix from https://github.com/Rypo
    #         causal_mask = xformers.attn_bias.BlockDiagonalCausalMask\
    #             .from_seqlens([q_len]*bsz)\
    #             .make_local_attention(window_size = sliding_window)
    # pass

    # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # output_hidden_states = (
    #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    # )
    
    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # self.model._has_no_labels = labels is None
    
    if past_key_values is not None:
        outputs = LlamaModel_fast_forward_inference(
            self,
            input_ids,
            past_key_values,
            position_ids = position_ids,
            attention_mask = attention_mask,
        )
    else:
        causal_mask = xformers.attn_bias.LowerTriangularMask()
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        self.model._has_no_labels = labels is None
            
        outputs = self.model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
        # outputs = BaseModelOutputWithPast
    pass

    hidden_states = outputs[0]
    bsz, q_len, hd = hidden_states.shape # torch.Size([1, 769, 6144])
    score = self.score.weight
    
    if bsz == 1 and q_len == 1:
        logits = torch.mv(score, hidden_states.ravel().to(score.dtype))
        logits = logits.unsqueeze(0).unsqueeze(0)
    else:
        logits = self.score(hidden_states.to(score.dtype))
    pass

    logits = logits.to(self.config.torch_dtype)

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    
    if self.model.config.pad_token_id is None and batch_size != 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    if self.model.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1


    pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss = None
    if labels is not None:
        #only work on the single_label_classification as for now
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
    pass
    

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # return CausalLMOutputWithPast(
    #     loss=loss,
    #     logits=logits,
    #     past_key_values=outputs.past_key_values,
    #     hidden_states=outputs.hidden_states,
    #     attentions=outputs.attentions,
    # )
    
    # if not return_dict:
    #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    # return BaseModelOutputWithPast(
    #     last_hidden_state=hidden_states,
    #     past_key_values=next_cache,
    #     hidden_states=all_hidden_states,
    #     attentions=all_self_attns,
    # )

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
pass


# Transformers had to update for Mistral Nemo 12b since Attention is (5120, 4096) now.
def patch_mistral_nemo_attention(function):
    function = function.replace(
        "(self.head_dim * self.num_heads) != self.hidden_size",
        "False",
    )
    function = function.replace(
        "self.head_dim = self.hidden_size // self.num_heads",
        "self.head_dim = config.head_dim",
    )
    function = function.replace(
        "self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)",
        "self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)",
    )
    return function
pass


class FastMistralModelSequenceClassification(FastLlamaModelSequenceClassification):

    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name         = "mistral",
            rope_module        = LlamaRotaryEmbedding,
            scaled_rope_module = LlamaLinearScalingRotaryEmbedding,
            attention_module   = MistralAttention,
        )
        # Just for Mistral Nemo models!
        if function is not None:
            function = patch_mistral_nemo_attention(function)
            # if True:#init_name is not None:
            exec(function, globals())
            MistralAttention.__init__  = eval(init_name)
        pass
        MistralAttention      .forward = MistralAttention_fast_forward
        MistralSdpaAttention  .forward = MistralAttention_fast_forward
        MistralFlashAttention2.forward = MistralAttention_fast_forward
        MistralDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        MistralModel          .forward = LlamaModel_fast_forward
        MistralForSequenceClassification    .forward = MistralForSequenceClassification_fast_forward
        # MistralForSequenceClassification    .forward = Sequence_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM  .forward = PeftModelForSequenceClassification_fast_forward
        # fix_prepare_inputs_for_generation(MistralForSequenceClassification)
        
        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.mistral.modeling_mistral
        transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = LlamaRotaryEmbedding
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name        = "unsloth/mistral-7b-bnb-4bit",
        max_seq_length    = None,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None, # Mistral does not support RoPE scaling
        fix_tokenizer     = True,
        model_patcher     = None,
        tokenizer_name    = None,
        trust_remote_code = False,
        num_labels        = None,
        **kwargs,
    ):
        return FastLlamaModelSequenceClassification.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = dtype,
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = FastMistralModelSequenceClassification,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            num_labels        = num_labels,
            **kwargs,
        )
    pass
pass
