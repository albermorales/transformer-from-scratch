import copy
import torch
from torch.nn import LayerNorm
import numpy as np
from src.dataset.English2GermanWMT14 import DEFAULT_TOKENIZER_NAME, _load_tokenizer
from src.dataset.sentencepiece_tokenizer import SentencePieceTokenizer

class Transformer(torch.nn.Module):
    def __init__(self,  device, 
                        vocab_size = None,
                        model_dim: int = 512,
                        query_dim: int=128,
                        num_heads: int=8,
                        num_blocks: int=8,
                        dtype: torch.dtype = torch.bfloat16,
                        max_length: int=1001,
                        dropout: float=0.15,
                        tokenizer: SentencePieceTokenizer | None = None,
                        ):
        
        super().__init__()

        self.model_dim = model_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.device = device
        self.dtype = dtype

        self.tokenizer = tokenizer or _load_tokenizer(DEFAULT_TOKENIZER_NAME)
        self.vocab_size = vocab_size if vocab_size is not None else self.tokenizer.vocab_size

        self.max_length = max_length
        self.dropout = dropout

        self.encoder = Encoder(self, dtype=dtype, device=device)
        self.decoder = Decoder(self, dtype=dtype, device=device)
        self.final_poyection = torch.nn.Linear(self.model_dim, self.vocab_size, dtype=dtype, device=device)

        shared_embedding = self.decoder.embedding.weight

        self.encoder.embedding.weight = shared_embedding
        self.final_poyection.weight = shared_embedding

        torch.nn.init.normal_(
            shared_embedding,
            mean=0.0,
            std=self.model_dim ** -0.5,
        )

    def forward(self, enc_input, dec_input, enc_padding_mask=None, dec_padding_mask=None):

        enc_output = self.encoder(enc_input, enc_padding_mask)
        dec_oputput = self.decoder(enc_output, dec_input, enc_padding_mask, dec_padding_mask)

        return self.final_poyection(dec_oputput)


    @torch.no_grad()
    def generate(self, enc_input, enc_padding_mask, beam=4, return_logits=False, shortness_penalization=0.6):
        """Generate up to max_length new tokens, excluding BOS and including EOS."""
        if not isinstance(beam, int) or isinstance(beam, bool) or not 1 <= beam <= self.vocab_size:
            raise ValueError("beam must be an integer between 1 and vocab_size")
        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")
        if shortness_penalization < 0:
            raise ValueError("shortness_penalization must be non-negative")

        generation_device = next(self.parameters()).device
        self.device = generation_device
        for component in (self.encoder, self.decoder):
            component.device = generation_device
            component.pos_embedding.positions = component.pos_embedding.positions.to(
                generation_device,
            )
        enc_input = enc_input.to(device=generation_device, dtype=torch.long)
        enc_padding_mask = enc_padding_mask.to(device=generation_device, dtype=torch.bool)

        # normalize batch and unbatched inputs
        if enc_input.ndim == 1:
            enc_input = enc_input.unsqueeze(dim=0)
            enc_padding_mask = enc_padding_mask.unsqueeze(dim=0)
        if enc_input.ndim != 2 or enc_padding_mask.shape != enc_input.shape:
            raise ValueError("Inputs and mask must have matching [batch, length] shapes")
        if enc_input.shape[0] == 0 or enc_input.shape[1] == 0:
            raise ValueError("Inputs must contain at least one non-empty sequence")
        if enc_input.shape[1] > self.max_length:
            raise ValueError("Source length exceeds the model's positional capacity")

        batch_size = enc_input.shape[0]
        sentence = torch.full(
            (batch_size, 1), self.tokenizer.bos_token_id,
            device=self.device, dtype=torch.long,
        )

        # pre-compute encoding
        enc_output = self.encoder(enc_input, enc_padding_mask)

        # first iteration
        logits = self.decoder(enc_output, sentence, enc_padding_mask)
        logits = self.final_poyection(logits[:, -1, :])
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        scores, pred_token = torch.topk(log_probs, k=beam, dim=-1)
        pred_token = pred_token.reshape(-1, 1)
        scores = scores.reshape(-1, 1)

        # redefine variables to match batch*beam dimension
        sentence = sentence.repeat_interleave(beam, dim=0)
        enc_output = enc_output.repeat_interleave(beam, dim=0)
        enc_padding_mask = enc_padding_mask.repeat_interleave(beam, dim=0)

        # add first predicted token
        sentence = torch.cat([sentence, pred_token], dim=1)
        sentence_lengths = torch.ones(
            (batch_size * beam, 1), dtype=torch.long, device=self.device,
        )
        continue_generating = ~(sentence == self.tokenizer.eos_token_id).any(dim=1)

        while continue_generating.any() and sentence.shape[1] - 1 < self.max_length:
            logits = self.decoder(
                enc_output[continue_generating],
                sentence[continue_generating],
                enc_padding_mask[continue_generating],
            )
            logits = self.final_poyection(logits[:, -1, :])
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            pred_token = torch.full(
                (batch_size * beam, beam), self.tokenizer.pad_token_id,
                device=self.device, dtype=torch.long,
            )
            token_score = torch.full(
                (batch_size * beam, beam), float("-inf"), device=self.device,
            )
            # Finished candidates offer just one PAD with zero additional score.
            token_score[~continue_generating, 0] = 0.0
            token_score[continue_generating], pred_token[continue_generating] = torch.topk(
                log_probs, k=beam, dim=-1,
            )

            # add counting scores and select the best continuations per input
            current_score = (token_score + scores).reshape(batch_size, beam * beam)
            pred_token = pred_token.reshape(batch_size, beam * beam)
            token_score = token_score.reshape(batch_size, beam * beam)
            _, token_idx = torch.topk(current_score, k=beam, dim=-1)

            pred_token = pred_token.gather(dim=1, index=token_idx).reshape(-1, 1)
            token_score = token_score.gather(dim=1, index=token_idx)

            # select each continuation's parent history, score and real length
            parent_idx = token_idx // beam
            continue_generating = continue_generating.reshape(batch_size, beam).gather(
                dim=1, index=parent_idx,
            ).reshape(-1)
            parent_idx = parent_idx.unsqueeze(-1)

            sentence = sentence.reshape(batch_size, beam, -1)
            sentence = sentence.gather(
                dim=1, index=parent_idx.expand(-1, -1, sentence.shape[-1]),
            ).reshape(batch_size * beam, -1)
            scores = scores.reshape(batch_size, beam, 1).gather(
                dim=1, index=parent_idx,
            ).reshape(-1, 1)
            sentence_lengths = sentence_lengths.reshape(batch_size, beam, 1).gather(
                dim=1, index=parent_idx,
            ).reshape(-1, 1)

            # add new tokens; finished candidates keep their score and length
            sentence = torch.cat([sentence, pred_token], dim=1)
            scores += token_score.reshape(-1, 1)
            sentence_lengths += continue_generating.unsqueeze(-1).to(torch.long)
            continue_generating = continue_generating & (
                pred_token[:, 0] != self.tokenizer.eos_token_id
            )

        # normalize using real lengths: BOS and padding after EOS do not count
        scores /= ((5.0 + sentence_lengths.float()) / 6.0) ** shortness_penalization

        # select one winning sentence per input
        scores = scores.reshape(batch_size, beam)
        score_idx = scores.argmax(dim=1)
        batch_idx = torch.arange(batch_size, device=self.device)
        sentence = sentence.reshape(batch_size, beam, -1)[batch_idx, score_idx]
        generated_token_ids = sentence[:, 1:]
        padding_mask = generated_token_ids != self.tokenizer.pad_token_id

        if return_logits:
            # Encoder copies are identical within each input's beam.
            selected_idx = batch_idx * beam + score_idx
            decoder_input = sentence[:, :-1]
            logits = self.final_poyection(self.decoder(
                enc_output[selected_idx], decoder_input, enc_padding_mask[selected_idx],
            ))
            after_eos = (decoder_input == self.tokenizer.eos_token_id).cumsum(dim=1) > 0
            logits = logits.masked_fill(after_eos.unsqueeze(-1), float("-inf"))
            logits[..., self.tokenizer.pad_token_id] = logits[
                ..., self.tokenizer.pad_token_id
            ].masked_fill(after_eos, 0.0)
            return {"logits": logits, "padding_mask": padding_mask}

        return {"token_ids": generated_token_ids, "padding_mask": padding_mask}


    @torch.no_grad()
    def export_onnx(self,
                    output_path: str,
                    opset: int = 17,
                    device: str = "cpu"):
        """Export greedy generation, including its dynamic EOS/length loop.

        Inputs: int64 enc_input and bool enc_padding_mask, both [batch, source].
        Outputs: int64 token_ids (without BOS, including EOS) and bool
        padding_mask, both [batch, generated]. max_length counts new tokens,
        excluding BOS and including EOS, as in generate(). The graph contains
        the tokenizer's special-token IDs,
        but text encoding/decoding still requires the tokenizer.

        Export a float32 copy for runtime compatibility; leave the training
        model's weights, dtype, device and train/eval mode unchanged.
        """
        if self.max_length < 2:
            raise ValueError("ONNX generation requires max_length >= 2")
        if opset < 17:
            raise ValueError("ONNX generation requires opset >= 17")

        export_device = torch.device(device)
        base = copy.deepcopy(self).eval().to(device=export_device, dtype=torch.float32)
        base.device = export_device
        for component in (base.encoder, base.decoder):
            component.device = export_device
            component.pos_embedding.positions = component.pos_embedding.positions.to(
                device=export_device, dtype=torch.float32,
            )

        source_length = min(8, self.max_length)
        enc_input = torch.full(
            (1, source_length), base.tokenizer.bos_token_id,
            dtype=torch.long, device=export_device,
        )
        enc_padding_mask = torch.ones_like(enc_input, dtype=torch.bool)
        dec_input = torch.full(
            (1, 2), base.tokenizer.bos_token_id,
            dtype=torch.long, device=export_device,
        )

        # Trace only tensor computations. Script the outer loop so ONNX gets
        # a Loop operator, rather than a fixed number of decoder calls.
        encoder = torch.jit.trace(base.encoder, (enc_input, enc_padding_mask))
        memory = base.encoder(enc_input, enc_padding_mask)
        decoder_step = torch.jit.trace(
            _DecoderExportStep(base).eval(), (memory, dec_input, enc_padding_mask),
        )
        wrapped = torch.jit.script(_GenerationExport(
            encoder, decoder_step,
            base.tokenizer.bos_token_id, base.tokenizer.eos_token_id,
            base.tokenizer.pad_token_id, base.max_length,
        ).eval())

        torch.onnx.export(
            wrapped,
            (enc_input, enc_padding_mask),
            output_path,
            export_params=True,
            opset_version=opset,
            dynamo=False,
            input_names=["enc_input", "enc_padding_mask"],
            output_names=["token_ids", "padding_mask"],
            dynamic_axes={
                "enc_input": {0: "batch_size", 1: "encoder_length"},
                "enc_padding_mask": {0: "batch_size", 1: "encoder_length"},
                "token_ids": {0: "batch_size", 1: "generated_length"},
                "padding_mask": {0: "batch_size", 1: "generated_length"},
            },
        )


class Encoder(torch.nn.Module):
    def __init__(self, model: Transformer, dtype: torch.dtype = torch.bfloat16, device: str='cuda'):
        super().__init__()

        self.model_dim = model.model_dim
        self.query_dim = model.query_dim
        self.num_heads = model.num_heads
        self.num_blocks = model.num_blocks
        self.dropout = model.dropout

        self.dtype = dtype
        self.device = device

        self.embedding = torch.nn.Embedding(num_embeddings=model.tokenizer.vocab_size, embedding_dim=self.model_dim, dtype=dtype, device=device)
        self.pos_embedding = PositionalEmbedding(dmodel=self.model_dim, max_length=model.max_length, device=self.device, dtype=self.dtype)


        class EncoderBlock(torch.nn.Module):
            def __init__(self, encoder: Encoder, dtype: torch.dtype = torch.bfloat16, device: str='cuda'):
                super().__init__()

                self.multiHeadAttention = VectorizedMultiHeadAttention(encoder.model_dim, encoder.query_dim, encoder.num_heads, dtype, device)
                self.attentionLayerNorm = LayerNorm(encoder.model_dim, dtype=dtype, device=device)

                self.ffn = FeedForwardNetwork(encoder.model_dim, encoder.model_dim*4, encoder.model_dim, dtype, device)
                self.ffnLayerNorm = LayerNorm(encoder.model_dim, dtype=dtype, device=device)

                self.dropout = torch.nn.Dropout(encoder.dropout)


            def forward(self, x, affinity_mask=None):
                x = self.attentionLayerNorm(x + self.dropout(self.multiHeadAttention(x, affinity_mask=affinity_mask)))
                x = self.ffnLayerNorm(x + self.dropout(self.ffn(x)))

                return x


        self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(self, self.dtype, self.device) for _ in range(self.num_blocks)])

    def forward(self, x, affinity_mask=None):
        
        x = self.embedding(x.to(dtype=torch.long, device=self.device)) * self.model_dim ** 0.5
        x += self.pos_embedding(x).to(x.dtype).to(x.device)

        for block in self.encoder_blocks:
            x = block(x, affinity_mask=affinity_mask)

        return x
        

class Decoder(torch.nn.Module):
    def __init__(self, model: Transformer, dtype: torch.dtype = torch.bfloat16, device: str='cuda'):
        super().__init__()

        self.model_dim = model.model_dim
        self.query_dim = model.query_dim
        self.num_heads = model.num_heads
        self.num_blocks = model.num_blocks
        self.dropout = model.dropout

        self.dtype = dtype
        self.device = device

        self.embedding = torch.nn.Embedding(num_embeddings=model.tokenizer.vocab_size, embedding_dim=self.model_dim, dtype=dtype, device=self.device)

        self.pos_embedding = PositionalEmbedding(dmodel=self.model_dim, max_length=model.max_length, device=self.device, dtype=self.dtype)

        class DecoderBlock(torch.nn.Module):
            def __init__(self, decoder: Decoder, dtype: torch.dtype = torch.bfloat16, device: str='cuda'):
                super().__init__()
                self.maskedMultiHeadAttention = VectorizedMultiHeadAttention(decoder.model_dim, decoder.query_dim, decoder.num_heads, dtype, device, causal_mask=True)
                self.maskedMultiHeadAttentionLayerNorm = LayerNorm(decoder.model_dim, dtype=dtype, device=device)

                self.multiHeadAttention = VectorizedMultiHeadAttention(decoder.model_dim, decoder.query_dim, decoder.num_heads, dtype, device)
                self.multiHeadAttentionLayerNorm = LayerNorm(decoder.model_dim, dtype=dtype, device=device)

                self.ffn = FeedForwardNetwork(decoder.model_dim, decoder.model_dim*4, decoder.model_dim,  dtype, device)
                self.ffnLayerNorm = LayerNorm(decoder.model_dim, dtype=dtype, device=device)

                self.dropout = torch.nn.Dropout(decoder.dropout)

            def forward(self, x, enc_output=None, enc_affinity_mask=None, dec_affinity_mask=None):
                x = self.maskedMultiHeadAttentionLayerNorm(x + self.dropout(self.maskedMultiHeadAttention(x, affinity_mask=dec_affinity_mask)))
                x = self.multiHeadAttentionLayerNorm(x + self.dropout(self.multiHeadAttention(x, enc_output=[enc_output if enc_output is not None else x][0], affinity_mask=enc_affinity_mask)))
                x = self.ffnLayerNorm(x + self.dropout(self.ffn(x)))

                return x

        self.decoder_blocks = torch.nn.ModuleList([DecoderBlock(self, self.dtype, self.device) for _ in range(self.num_blocks)])

            
    def forward(self, enc_output, dec_input,  enc_affinity_mask=None, dec_affinity_mask=None):
        x = self.embedding(dec_input.to(dtype=torch.long, device=self.device)) * self.model_dim ** 0.5
        x += self.pos_embedding(x).to(x.dtype).to(x.device)

        for block in self.decoder_blocks:
            x = block(x, enc_output=enc_output, enc_affinity_mask=enc_affinity_mask, dec_affinity_mask=dec_affinity_mask)

        return x


class MultiHeadAttention(torch.nn.Module):
    """Deprecated: implementación antigua; el modelo usa VectorizedMultiHeadAttention."""

    def __init__(self, model_dim: int = 512, query_dim: int=128, num_heads: int=8, dtype: torch.dtype=torch.bfloat16, device: str='cuda', causal_mask: bool=False):

        super().__init__()

        self.attention_dim = int(model_dim // num_heads)
        self.num_heads = num_heads       

        self.singleHeadAttentionStack = torch.nn.ModuleList([ScaleDotAttention(model_dim, query_dim, num_heads, dtype, device, causal_mask) for _ in range(num_heads)])
        self.output_linear = torch.nn.Linear(model_dim, model_dim, dtype=dtype, device=device)

    def forward(self, x, enc_output=None, affinity_mask=None):

        # Pre-alocate to speed up
        attention_result = torch.empty(x.shape[0], x.shape[1], self.attention_dim, self.num_heads, dtype=x.dtype, device=x.device)    

        # manage multiple input args
        x = (enc_output, x) if enc_output is not None else (x,)

        for n, attention_head in enumerate(self.singleHeadAttentionStack):
            attention_result[:,:,:,n] = attention_head(x, affinity_mask=affinity_mask)

        # Concat: T x d_at x h -> T x (d_at · h)
        attention_result = torch.reshape(attention_result, [attention_result.shape[0], attention_result.shape[1], attention_result.shape[2]*attention_result.shape[3]])

        return self.output_linear(attention_result)


class ScaleDotAttention(torch.nn.Module):
    """Deprecated: cabeza de atención de la implementación antigua no vectorizada."""

    def __init__(self, model_dim: int = 512, query_dim: int=128, num_heads: int=8, dtype: torch.dtype=torch.bfloat16, device: str='cuda', causal_mask: bool=False):
        super().__init__()

        assert model_dim % num_heads == 0, f"Dimension conflict. Model dimension must be a num_heads multiple, but model_dim % num_heads = {model_dim % num_heads}"
        self.attention_dim = int(model_dim // num_heads)
        self.query_dim = query_dim
        self.causal_mask = causal_mask

        self.W_k = torch.nn.Parameter(torch.empty(model_dim, self.query_dim, dtype=dtype, device=device))  # Uniform random
        torch.nn.init.xavier_uniform_(self.W_k)
        self.W_q = torch.nn.Parameter(torch.empty(model_dim, self.query_dim, dtype=dtype, device=device))
        torch.nn.init.xavier_uniform_(self.W_q)
        self.W_v = torch.nn.Parameter(torch.empty(model_dim, self.attention_dim, dtype=dtype, device=device))
        torch.nn.init.xavier_uniform_(self.W_v)


    def forward(self, x, affinity_mask=None):
           
        K = torch.matmul(x[0], self.W_k)
        Q = torch.matmul(x[1] if len(x)>1 else x[0], self.W_q)  # Query comes from decoder if 2 inputs are provided
        V = torch.matmul(x[0], self.W_v)

        affinity = torch.matmul(Q,torch.transpose(K, dim0=1, dim1=2))

        if affinity_mask is not None:
            affinity = affinity.masked_fill(~affinity_mask[:, None, :].to(x[0].device),-torch.inf)

        if self.causal_mask:
            mask = torch.arange(x[0].shape[1], device=affinity.device)[None, :] > torch.arange(x[0].shape[1], device=affinity.device)[:, None]
            affinity[:,mask] = -torch.inf

        attention = torch.matmul(torch.nn.functional.softmax(affinity*(self.query_dim ** -0.5), dim=-1), V)

        return attention


class VectorizedMultiHeadAttention(torch.nn.Module):
    def __init__(self, model_dim: int = 512, query_dim: int=128, num_heads: int=8, dtype: torch.dtype=torch.bfloat16, device: str='cuda', causal_mask: bool=False):
        super().__init__()

        self.num_heads = num_heads
        self.query_dim = query_dim
        self.attention_dim = model_dim//num_heads

        self.causal_mask =  causal_mask

        self.W_q = torch.nn.Linear(in_features=model_dim, out_features=(self.query_dim*self.num_heads), dtype=dtype, device=device)
        self.W_k = torch.nn.Linear(in_features=model_dim, out_features=(self.query_dim*self.num_heads), dtype=dtype, device=device)
        self.W_v = torch.nn.Linear(in_features=model_dim, out_features=(model_dim), dtype=dtype, device=device)

        self.output_linear = torch.nn.Linear(model_dim, model_dim, dtype=dtype, device=device)


    def forward(self, x, enc_output=None, affinity_mask=None):

        Q = self.W_q(x)                                                        # [B, T_q, num_heads * query_dim]

        kv_input = x if enc_output is None else enc_output

        K = self.W_k(kv_input)                                                 # [B, T_k, num_heads * query_dim]
        V = self.W_v(kv_input)                                                 # [B, T_k, model_dim]

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.query_dim)   # [B, T_q, num_heads, query_dim]
        Q = Q.permute(0, 2, 1, 3)                                              # [B, num_heads, T_q, query_dim]

        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.query_dim)   # [B, T_k, num_heads, query_dim]
        K = K.permute(0, 2, 3, 1)                                              # [B, num_heads, query_dim, T_k]

        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.attention_dim)  # [B, T_k, num_heads, attention_dim]
        V = V.permute(0, 2, 1, 3)                                              # [B, num_heads, T_k, attention_dim]

        affinity = Q @ K                                                       # [B, num_heads, T_q, T_k]
        if affinity_mask is not None:
            affinity = affinity.masked_fill(~affinity_mask[:, None, None, :].to(x[0].device),-torch.inf)

        if self.causal_mask:
            mask = torch.arange(x.shape[1], device=affinity.device)[None, :] > torch.arange(x.shape[1], device=affinity.device)[:, None]
            affinity = affinity.masked_fill(mask[None, None, :, :], -torch.inf)

        attention = torch.matmul(torch.nn.functional.softmax(affinity*(self.query_dim ** -0.5), dim=-1), V)     # [B, num_heads, T_q, attention_dim]
        attention = attention.permute(0, 2, 1, 3)                                # [B, T_q, num_heads, attention_dim]
        attention = attention.reshape(attention.shape[0], attention.shape[1], -1)  # [B, T_q, model_dim]

        return self.output_linear(attention)

           
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=512, dtype: torch.dtype=torch.bfloat16, device: str='cuda'):
        super().__init__()

        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.layer_2 = torch.nn.Linear(hidden_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x):
            x = torch.nn.functional.relu(self.layer_1(x))
            return self.layer_2(x)


class PositionalEmbedding():
    def __init__(self, dmodel: int=512, max_length: int=120, dtype: torch.dtype=torch.bfloat16, device: str='cuda'):
        self.dmodel = dmodel
        self.dtype = dtype
        self.device = device
        self.max_length = max_length
        self.positions = self.compute_positions()

    def compute_positions(self):

        pos_emb = torch.zeros([self.max_length, self.dmodel], dtype=torch.float32, device='cpu')

        W = 1/torch.pow(10000,2*torch.arange(0, np.ceil(self.dmodel/2), dtype=torch.float32, device='cpu')/self.dmodel)

        P = torch.arange(0, self.max_length, dtype=torch.float32, device='cpu')

        angles = torch.matmul(P.reshape([-1,1]), W.reshape([1,-1]))

        pos_emb[:, 0::2] = torch.sin(angles)
        pos_emb[:, 1::2] = torch.cos(angles)

        return pos_emb.to(self.dtype).to(self.device)


    def __call__(self, x):
        return self.positions[:x.shape[1]].unsqueeze(0).expand(x.shape[0], -1, -1)


    

class _DecoderExportStep(torch.nn.Module):
    """One decoder step; the encoder output is reused by the generation loop."""

    def __init__(self, base):
        super().__init__()
        self.decoder = base.decoder
        self.projection = base.final_poyection

    def forward(self, memory, token_ids, enc_padding_mask):
        hidden = self.decoder(memory, token_ids, enc_padding_mask)
        return self.projection(hidden[:, -1, :])


class _GenerationExport(torch.nn.Module):
    """Script the stopping condition instead of tracing an example's iterations."""

    __constants__ = ["bos_id", "eos_id", "pad_id", "max_length"]

    def __init__(self, encoder, decoder_step, bos_id, eos_id, pad_id, max_length):
        super().__init__()
        self.encoder = encoder
        self.decoder_step = decoder_step
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

    def forward(self, enc_input: torch.Tensor, enc_padding_mask: torch.Tensor):
        memory = self.encoder(enc_input, enc_padding_mask)
        sentence = torch.full(
            (enc_input.size(0), 1), self.bos_id,
            dtype=torch.long, device=enc_input.device,
        )
        finished = torch.zeros(
            (enc_input.size(0),), dtype=torch.bool, device=enc_input.device,
        )
        # Keep the full batch and append PAD to finished rows. This avoids
        # data-dependent batch sizes inside the exported decoder.
        while sentence.size(1) - 1 < self.max_length and not bool(torch.all(finished)):
            logits = self.decoder_step(memory, sentence, enc_padding_mask)
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.pad_id), next_token,
            )
            sentence = torch.cat((sentence, next_token.unsqueeze(1)), dim=1)
            finished = finished | (next_token == self.eos_id)

        token_ids = sentence[:, 1:]
        return token_ids, token_ids != self.pad_id

