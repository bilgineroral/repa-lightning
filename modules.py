from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import PretrainConfig, arch_predict_offset, arch_single_stream
from rope import RoPE, apply_rotary_pos_emb, get_patches_center_coordinates, prepend_prefix_tokens

class PatchEmbed(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, 
            kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """ img: [B,C_in,H,W] """
        return self.patch_embed(img).flatten(2).transpose(1, 2) # [B,N,D]

class DropPath(nn.Module):
    def __init__(self, drop_p: Optional[float] = None) -> None:
        super().__init__()
        self.drop_p = drop_p if drop_p is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_p == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        if init_value is not None and init_value > 0:
            self.lambda1 = nn.Parameter(init_value * torch.ones(dim))
        else:
            self.lambda1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.lambda1 is None else (x * self.lambda1)

class SelfAttention(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dim = config.hidden_size

        self.attn_drop_p = config.attn_drop

        self.q_proj = nn.Linear(config.hidden_size, self.dim, bias=config.qkv_bias)
        self.kv_proj = nn.Linear(config.hidden_size, 2*self.dim, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop = nn.Dropout(config.drop)

        if config.qk_norm:
            self.q_norm = nn.LayerNorm(
                self.head_dim, eps=config.layer_norm_eps,
                elementwise_affine=config.qk_norm_affine, bias=config.qk_norm_bias
            )
            self.k_norm = nn.LayerNorm(
                self.head_dim, eps=config.layer_norm_eps,
                elementwise_affine=config.qk_norm_affine, bias=config.qk_norm_bias
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, # (cos, sin)
        attn_mask: Optional[torch.Tensor] = None,
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
        is_causal: bool = False,
        q_expand: Optional[Tuple[int, int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """ x_q: [B, Nq, D], x_kv: [B, Nk, D] or None, attn_mask: [Nq, Nk] bool, False = masked
        is_causal: square causal attention without a mask (attn_mask must be None)
        q_expand: (B, Nq) when x_q is one row [1, 1, D] shared by every query (perf_query_proj_once) """
        if x_kv is None:
            x_kv = x_q

        B, Nq, D = x_q.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x_q)
        q = q.view(B, Nq, H, d).transpose(1, 2)  # [B,H,Nq,d]
        q = self.q_norm(q)
        if q_expand is not None:
            B, Nq = q_expand
            q = q.expand(B, H, Nq, d)

        if kv is None:
            Nk = x_kv.shape[1]
            k, v = self.kv_proj(x_kv).split(D, dim=-1)
            k = k.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
            v = v.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
            k = self.k_norm(k)

            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            k, v = kv
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, _ = apply_rotary_pos_emb(q, None, cos, sin)

        mask = None if attn_mask is None else attn_mask[None, None, :, :] # [1,1,Nq,Nk]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        out = self.proj(out)
        out = self.drop(out)
        
        if return_kv:
            return out, k, v
        return out

    def project_kv(
        self, x_kv: torch.Tensor, position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ K/V exactly as forward computes them, for a layer whose own queries are not needed (perf_skip_last_content) """
        B, Nk, D = x_kv.shape
        H, d = self.num_heads, self.head_dim
        k, v = self.kv_proj(x_kv).split(D, dim=-1)
        k = k.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
        v = v.view(B, Nk, H, d).transpose(1, 2)  # [B,H,Nk,d]
        k = self.k_norm(k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            k, _ = apply_rotary_pos_emb(k, None, cos, sin)
        return k, v

class MLP(nn.Module):
    def __init__(self, config: PretrainConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        width = intermediate_size or config.intermediate_size # arch_query_mlp_width: a narrower query MLP
        self.up_proj = nn.Linear(config.hidden_size, width)
        self.down_proj = nn.Linear(width, config.hidden_size)
        self.dropout = nn.Dropout(config.drop)
        
        self.use_gated_mlp = config.use_gated_mlp
        if self.use_gated_mlp:
            self.gate_proj = nn.Linear(config.hidden_size, width)

        if config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "swish":
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up_proj(x)
        x = self.act_fn(self.gate_proj(x)) * x_up if self.use_gated_mlp else self.act_fn(x_up)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

def sincos_2d_table(grid: int, dim: int) -> torch.Tensor:
    """ fixed 2D sin-cos table [grid * grid, dim]: half the channels encode y, half x (sin and cos each), at RoPE's patch
    centers (get_patches_center_coordinates) mapped from [-1, 1] to grid units, with MAE-style frequencies (option B's
    parameter-free target-position variant) """
    pos = (get_patches_center_coordinates(grid, grid, torch.float32, "cpu") + 1) / 2 * grid # [N, 2] (y, x)
    d = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(d, dtype=torch.float32) / d)
    parts = []
    for axis in (0, 1):
        angles = pos[:, axis:axis + 1] * omega[None]
        parts += [torch.sin(angles), torch.cos(angles)]
    return torch.cat(parts, dim=1)

class REPALayer(nn.Module):
    def __init__(self, config: PretrainConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.attention = SelfAttention(config)
        self.mlp = MLP(config)

        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.ls1_h = LayerScale(config.hidden_size, config.layerscale_value)
        self.ls2_h = LayerScale(config.hidden_size, config.layerscale_value)
        if config.use_query_stream:
            self.ls1_g = LayerScale(config.hidden_size, config.layerscale_value)
            self.ls2_g = LayerScale(config.hidden_size, config.layerscale_value)

        self.ln1_h = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2_h = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_query_stream:
            self.ln1_g = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ln2_g = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # arch_query_mlp_width: the query stream gets its own MLP of this width instead of sharing self.mlp
        query_mlp_width = getattr(config, "arch_query_mlp_width", 0)
        self.query_mlp = MLP(config, query_mlp_width) if (config.use_query_stream and query_mlp_width) else None

    def forward(
        self,
        h: torch.Tensor,  # content stream [B,1+N,D] (CLS + patches)
        g: Optional[torch.Tensor],  # query stream [B,N,D], None without the query stream
        position_embeddings_h: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_g: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask_h: Optional[torch.Tensor] = None,
        attn_mask_g: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        content_needed: bool = True,
        w: Optional[torch.Tensor] = None,
    ):
        """ h: [B,1+N,D], g: [B,N,D] or None
        is_causal: both streams have N rows and use causal attention without masks (perf_causal_no_mask)
        content_needed=False: the content stream only provides K/V; returns h=None (perf_skip_last_content)
        w: [1,1,D] when every row of g is w (layer 1): LN + Q projection run on one row (perf_query_proj_once) """
        h_in, g_in = h, g
        h_norm = self.ln1_h(h_in)
        g_src = w if w is not None else g_in
        g_norm = self.ln1_g(g_src) if g is not None else None # before the content attention: keeps the op order

        # self-attention
        if content_needed:
            h_attn, k, v = self.attention(
                x_q=h_norm, x_kv=h_norm,
                position_embeddings=position_embeddings_h,
                attn_mask=attn_mask_h,
                return_kv=True,
                is_causal=is_causal and attn_mask_h is None,
            )
            h = h_in + self.drop_path1(self.ls1_h(h_attn))
        else:
            k, v = self.attention.project_kv(h_norm, position_embeddings_h)

        if g is None: # content stream only
            h = h + self.drop_path2(self.ls2_h(self.mlp(self.ln2_h(h))))
            return h, None

        g_attn = self.attention(
            x_q=g_norm, x_kv=h_norm,
            position_embeddings=position_embeddings_g,
            attn_mask=attn_mask_g,
            kv=(k, v),
            is_causal=is_causal and attn_mask_g is None, # arch_predict_frac keeps an explicit (offset) query mask
            q_expand=tuple(g_in.shape[:2]) if w is not None else None,
        )
        g = g_in + self.drop_path1(self.ls1_g(g_attn))

        # mlp
        if content_needed:
            h_ff = self.mlp(self.ln2_h(h))
            h = h + self.drop_path2(self.ls2_h(h_ff))
        else:
            h = None

        mlp_g = self.mlp if self.query_mlp is None else self.query_mlp
        g_ff = mlp_g(self.ln2_g(g))
        g = g + self.drop_path2(self.ls2_g(g_ff))

        return h, g
    
class REPAEncoder(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        dpr = np.linspace(0, config.drop_path, config.num_hidden_layers).tolist()
        self.layers = nn.ModuleList([REPALayer(config, drop_path_rate=dpr[i]) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        h: torch.Tensor, g: Optional[torch.Tensor],
        position_embeddings_h: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_g: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask_h: Optional[torch.Tensor] = None,
        attn_mask_g: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        skip_last_content: bool = False,
        w: Optional[torch.Tensor] = None,
        query_start: int = 0,
        h_add: Optional[torch.Tensor] = None,
        h_add_from: int = 0,
    ):
        """ query_start: first layer that runs the query stream (arch_query_top_k); g is the query input at that layer
        h_add: added to h before each layer from h_add_from on (arch_target_pos_layers) """
        last = len(self.layers) - 1
        g_start, g = g, (g if query_start == 0 else None)
        for i, layer in enumerate(self.layers):
            if h_add is not None and i >= h_add_from:
                h = h + h_add
            if i == query_start:
                g = g_start
            h, g = layer(
                h, g, position_embeddings_h, position_embeddings_g, attn_mask_h, attn_mask_g,
                is_causal=is_causal,
                content_needed=not (skip_last_content and i == last),
                w=w if i == query_start else None,
            )
        return h, g

class REPA(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(config)
        self.dropout = nn.Dropout(config.drop)

        num_patches = (config.image_size // config.patch_size) ** 2
        if config.use_rope:
            self.rope_embed = RoPE(config) # spatial position
            self.pos_embed = None
        else:
            self.rope_embed = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=config.initializer_range)

        self.encoder = REPAEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # final norm, as NEPA

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=config.initializer_range)
        if config.use_query_stream:
            self.w = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) # query stream input
            nn.init.trunc_normal_(self.w, mean=0.0, std=config.initializer_range)
        else:
            self.w = None

        # speed flags (two-stream model only); getattr: configs pickled into older checkpoints don't have them
        self.perf_skip_last_content = getattr(config, "perf_skip_last_content", False)
        self.perf_causal_no_mask = getattr(config, "perf_causal_no_mask", False)
        self.perf_query_proj_once = getattr(config, "perf_query_proj_once", False)

        # architecture flags (change the model); getattr: configs pickled into older checkpoints don't have them
        self.arch_single_stream = arch_single_stream(config)
        self.arch_target_pos = getattr(config, "arch_single_stream_target_pos", False)
        self.arch_target_pos_layers = getattr(config, "arch_target_pos_layers", 0)
        self.arch_query_top_k = getattr(config, "arch_query_top_k", 0)
        self.arch_predict_offset = arch_predict_offset(config)
        self.tgt_null = None
        if not self.arch_target_pos:
            self.tgt_pos_embed = None
        elif getattr(config, "arch_target_pos_type", "learned") == "sincos": # parameter-free variant
            grid = config.image_size // config.patch_size
            self.register_buffer("tgt_pos_embed", sincos_2d_table(grid, config.hidden_size)[None], persistent=False)
        else: # option B default: learned table [1, N, D], initialized like pos_embed
            self.tgt_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
            nn.init.trunc_normal_(self.tgt_pos_embed, mean=0.0, std=config.initializer_range)
        if self.arch_target_pos and getattr(config, "arch_target_pos_null_p", 0.0) > 0:
            self.tgt_null = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) # option B: null target position
            nn.init.trunc_normal_(self.tgt_null, mean=0.0, std=config.initializer_range)

        self.apply(self._init_weights)

    def _make_attn_masks(self, N: int, device: torch.device):
        # key masks over [CLS, x_1..x_N]: the content stream is causal, the query for target t sees CLS and x_<t
        attn_mask_h = torch.tril(torch.ones(1 + N, 1 + N, dtype=torch.bool, device=device), diagonal=0)
        attn_mask_g = torch.tril(torch.ones(N, 1 + N, dtype=torch.bool, device=device), diagonal=0)
        return attn_mask_h, attn_mask_g

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, LayerScale]):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, LayerScale):
            if module.lambda1 is not None:
                module.lambda1.data.fill_(self.config.layerscale_value)

    def forward(self, img: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                null_rows: Optional[torch.Tensor] = None):
        """ img: [B,C_in,H,W], position_ids: [B,N] or None
        null_rows: [B,N] bool, rows given the null target position (arch_target_pos_null_p, training only) """
        patches = self.patch_embed(img)
        B, N, D = patches.shape

        if position_ids is None:
            tgt = patches
        else:
            tgt = patches.gather(1, position_ids[:, :, None].expand(B, N, D))
        cls = self.cls_token.expand(B, -1, -1)

        # Position embeddings: RoPE applied in attention (CLS unrotated), APE added to patch tokens
        if self.rope_embed is not None:
            position_embeds_g = self.rope_embed(img, position_ids=position_ids)
            position_embeds_h = prepend_prefix_tokens(position_embeds_g)
            h0 = torch.cat([cls, tgt], dim=1)
            g0 = self.w.expand(B, N, D) if self.w is not None else None
        else:
            # APE: gather in permuted order and add to both streams (not target)
            if position_ids is None:
                pos = self.pos_embed.expand(B, -1, -1)
            else:
                pos = self.pos_embed.expand(B, -1, -1).gather(1, position_ids[:, :, None].expand(B, N, D))
            h0 = torch.cat([cls, tgt + pos], dim=1)
            g0 = self.w.expand(B, N, D) + pos if self.w is not None else None
            position_embeds_h = position_embeds_g = None

        h0 = self.dropout(h0)  # content stream sees the CLS token and patch embeddings

        if self.arch_single_stream:
            return self._single_stream_forward(h0, tgt, position_embeds_h, position_ids, null_rows)

        off = self.arch_predict_offset
        if off and g0 is not None: # arch_predict_frac: no query rows for the first `off` targets of each order
            g0 = g0[:, off:]
            if position_embeds_g is not None:
                position_embeds_g = tuple(pe[:, off:] for pe in position_embeds_g)

        two_stream = g0 is not None
        causal = self.perf_causal_no_mask and two_stream
        skip_last_content = self.perf_skip_last_content and two_stream
        # with RoPE every row of g0 is w; with APE each row also has its position added
        w = self.w if (self.perf_query_proj_once and two_stream and self.rope_embed is not None) else None

        if causal:
            # no query row attends to the last permuted patch (query t sees CLS and x_<=t), so dropping it gives both
            # streams N rows and the same square causal mask: SDPA's is_causal, no explicit masks
            h0 = h0[:, :-1]
            if position_embeds_h is not None:
                position_embeds_h = tuple(pe[:, :-1] for pe in position_embeds_h)
            attn_mask_h = attn_mask_g = None
        else:
            attn_mask_h, attn_mask_g = self._make_attn_masks(N, tgt.device)
        if off and two_stream: # query row i is target off + i: it sees keys 0..off + i (CLS, x_pi0..x_pi(off+i-1))
            attn_mask_g = torch.tril(torch.ones(N - off, h0.shape[1], dtype=torch.bool, device=tgt.device), diagonal=off)
        L = len(self.encoder.layers)
        query_start = L - self.arch_query_top_k if (self.arch_query_top_k and two_stream) else 0

        h, g = self.encoder(
            h0, g0,
            position_embeddings_h=position_embeds_h,
            position_embeddings_g=position_embeds_g,
            attn_mask_h=attn_mask_h,
            attn_mask_g=attn_mask_g,
            is_causal=causal,
            skip_last_content=skip_last_content,
            w=w,
            query_start=query_start,
        )

        if g is None:
            # NEPA: the content stream at CLS, x_1..x_{N-1} predicts x_1..x_N
            return tgt, self.norm(h[:, :-1])
        # parameters that no longer reach the loss stay in the graph with zero gradient, as h.sum() * 0.0 does, so DDP
        # needs no find_unused_parameters
        unused = []
        for layer in self.encoder.layers[:query_start]: # arch_query_top_k: query-stream parts of the lower layers
            unused += [layer.ln1_g.weight, layer.ln1_g.bias, layer.ln2_g.weight, layer.ln2_g.bias,
                       layer.ls1_g.lambda1, layer.ls2_g.lambda1]
            if layer.query_mlp is not None:
                unused += list(layer.query_mlp.parameters())
        tgt = tgt[:, off:] if off else tgt # arch_predict_frac: pred row i is target off + i
        if skip_last_content:
            # these last-layer parameters only fed the skipped content output
            last = self.encoder.layers[-1]
            unused += [last.ln2_h.weight, last.ln2_h.bias, last.ls1_h.lambda1, last.ls2_h.lambda1]
            if last.query_mlp is not None: # arch_query_mlp_width: the shared MLP only served the skipped content output
                unused += list(last.mlp.parameters())
            unused = [p for p in unused if p is not None]
            return tgt, self.norm(g) + sum(p.sum() for p in unused) * 0.0
        unused = [p for p in unused if p is not None]
        if unused:
            return tgt, self.norm(g) + h.sum() * 0.0 + sum(p.sum() for p in unused) * 0.0
        return tgt, self.norm(g) + h.sum() * 0.0

    def _single_stream_forward(self, h0, tgt, position_embeds_h, position_ids, null_rows):
        """ arch_single_stream_ref / arch_single_stream_target_pos (option B, ~/repa-notes/option_b_design.md):
        the content stream alone, square causal over the permuted order. Row j (CLS for j = 0, else x_pi(j-1)) predicts
        x_pi(j); the last permuted patch is dropped from the input since no row predicts from it. Option B adds the
        target position e_tgt[pi(j)] to row j, at the input or before each of the last k layers. """
        B, N, D = tgt.shape
        h0 = h0[:, :-1]
        if position_embeds_h is not None:
            position_embeds_h = tuple(pe[:, :-1] for pe in position_embeds_h)
        h_add, h_add_from = None, 0
        if self.arch_target_pos:
            table = self.tgt_pos_embed.expand(B, -1, -1)
            e_tgt = table if position_ids is None else table.gather(1, position_ids[:, :, None].expand(B, N, D))
            if null_rows is not None and self.tgt_null is not None:
                e_tgt = torch.where(null_rows[:, :, None], self.tgt_null.expand_as(e_tgt), e_tgt)
            if self.arch_target_pos_layers:
                h_add, h_add_from = e_tgt, len(self.encoder.layers) - self.arch_target_pos_layers
            else:
                h0 = h0 + e_tgt
        h, _ = self.encoder(h0, None, position_embeddings_h=position_embeds_h, is_causal=True,
                            h_add=h_add, h_add_from=h_add_from)
        return tgt, self.norm(h)

class SingleStreamREPALayer(nn.Module):
    """ Content stream of a REPALayer, run with bidirectional attention """
    def __init__(self, layer: REPALayer, drop_path: float = 0.0):
        super().__init__()
        self.attention = layer.attention
        self.mlp = layer.mlp
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls1 = layer.ls1_h
        self.ls2 = layer.ls2_h

        self.ln1 = layer.ln1_h
        self.ln2 = layer.ln2_h

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        x = x + self.drop_path(self.ls1(self.attention(self.ln1(x), position_embeddings=position_embeddings)))
        x = x + self.drop_path(self.ls2(self.mlp(self.ln2(x))))
        return x

class SingleStreamREPAEncoder(nn.Module):
    def __init__(self, encoder: REPAEncoder, drop_path: float = 0.0):
        super().__init__()
        dpr = np.linspace(0, drop_path, len(encoder.layers)).tolist()
        self.layers = nn.ModuleList([
            SingleStreamREPALayer(layer, drop_path=dpr[i]) for i, layer in enumerate(encoder.layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        for layer in self.layers:
            x = layer(x, position_embeddings)
        return x

def add_qk_norm_affine(layers: nn.ModuleList, config: PretrainConfig) -> None:
    """ Learnable QK-norm scales (initialized to 1) on top of pretrained non-affine QK-norms, as NEPA's finetuning """
    if not config.qk_norm or config.qk_norm_affine:
        return
    for layer in layers:
        attn = layer.attention
        attn.q_norm = nn.LayerNorm(attn.head_dim, eps=config.layer_norm_eps, bias=config.qk_norm_bias)
        attn.k_norm = nn.LayerNorm(attn.head_dim, eps=config.layer_norm_eps, bias=config.qk_norm_bias)

if __name__ == "__main__":
    c = PretrainConfig(
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048
    )
    model = REPA(c)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params/1e6:.2f}M")
