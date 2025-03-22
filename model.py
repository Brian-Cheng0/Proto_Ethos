import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ----- Timeline Prototype Matcher (Additional Layer) -----
class TimelinePrototypeMatcher(nn.Module):
    def __init__(self, num_prototypes, prototype_shape, temp, patch_select, radius, direction='right'):
        """
        Args:
            num_prototypes (int): Total number of prototypes.
            prototype_shape (tuple): Tuple of (num_prototypes, D, n_p).
            temp (float): Temperature scaling factor.
            patch_select (Tensor): Patch weighting tensor of shape (1, num_prototypes, n_p).
            radius (int): Radius for restricting matching to contiguous events.
            direction (str): 'right' or 'left'; restrict subsequent selections to this direction.
        """
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_shape = prototype_shape
        self.temp = temp
        self.register_buffer('patch_select', patch_select)
        self.radius = radius
        self.direction = direction

        self.prototype_vectors = nn.Parameter(
            torch.rand(num_prototypes, prototype_shape[1], prototype_shape[2]),
            requires_grad=True
        )

    def subpatch_dist(self, x):
        """
        Computes similarity scores between input embeddings and each subpatch of the prototypes.

        Args:
            x (Tensor): Input tensor of shape (B, T, D).

        Returns:
            event_features (Tensor): The original input x.
            dist_all (Tensor): Similarity scores of shape (B, num_prototypes, T, n_p).
        """
        event_features = x
        embedding_norm = F.normalize(event_features, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototype_vectors, p=2, dim=1)
        B, T, D = event_features.shape
        n_p = self.prototype_shape[-1]
        dist_all_list = []
        for i in range(n_p):
            proto_i = prototypes_norm[:, :, i]  # (num_prototypes, D)
            similarity = torch.matmul(embedding_norm, proto_i.t())  # (B, T, num_prototypes)
            similarity = similarity.transpose(1, 2).unsqueeze(-1)   # (B, num_prototypes, T, 1)
            dist_all_list.append(similarity)
        dist_all = torch.cat(dist_all_list, dim=-1)
        return event_features, dist_all

    def neighboring_mask(self, center_indices, T):
        """
        Computes a mask for tokens within a given radius of a center index.

        Args:
            center_indices (Tensor): Tensor of shape (B, num_prototypes, 1).
            T (int): Total number of tokens.

        Returns:
            mask (Tensor): Tensor of shape (B, num_prototypes, T) with 1 for tokens within radius, else 0.
        """
        center_indices = center_indices.squeeze(-1)
        B, num_proto = center_indices.shape
        all_indices = torch.arange(T, device=center_indices.device).unsqueeze(0).unsqueeze(0).expand(B, num_proto, T)
        center_expanded = center_indices.unsqueeze(-1).expand(B, num_proto, T)
        mask = (torch.abs(all_indices - center_expanded) <= self.radius).float()
        return mask

    def directional_mask(self, prev_indices, T):
        """
        Creates a mask that allows only tokens to the right (or left) of the given indices.

        Args:
            prev_indices (Tensor): Tensor of shape (B, num_prototypes, 1).
            T (int): Total number of tokens.

        Returns:
            mask (Tensor): Tensor of shape (B, num_prototypes, T).
        """
        prev_indices = prev_indices.squeeze(-1)
        B, num_proto = prev_indices.shape
        all_indices = torch.arange(T, device=prev_indices.device).unsqueeze(0).unsqueeze(0).expand(B, num_proto, T)
        if self.direction == 'right':
            mask = (all_indices > prev_indices.unsqueeze(-1)).float()
        else:
            mask = (all_indices < prev_indices.unsqueeze(-1)).float()
        return mask

    def greedy_distance(self, x, timeline_mask=None, get_f=False):
        """
        Computes the matching between input events and each prototype's subpatches.

        Args:
            x (Tensor): Input tensor of shape (B, T, D).
            timeline_mask (Tensor, optional): Tensor of shape (B, T) where 1 indicates valid tokens.
            get_f (bool): If True, returns the original event features along with computed min_distances and indices.

        Returns:
            If get_f is False:
                (max_activation_slots, min_distances, indices_reordered)
            If get_f is True:
                (event_features, min_distances, indices_reordered)
        """
        event_features, dist_all = self.subpatch_dist(x)  # (B, num_prototypes, T, n_p)
        if timeline_mask is not None:
            tm = timeline_mask.unsqueeze(1).unsqueeze(-1)  # shape (B, 1, T, 1)
            dist_all = dist_all * tm + (1 - tm) * (-1e5)

        # 'slots' is the subpatch weighting; factor is used to rescale
        slots = torch.sigmoid(self.patch_select * self.temp)  # (1, num_prototypes, n_p)
        factor = (slots.sum(-1)).unsqueeze(-1) + 1e-10         # (1, num_prototypes, 1)

        B, num_proto, T, n_p = dist_all.shape
        mask_act = torch.ones((B, num_proto, T), device=dist_all.device)
        mask_subpatch = torch.ones((B, num_proto, n_p), device=dist_all.device)
        mask_all = torch.ones((B, num_proto, T, n_p), device=dist_all.device)
        adjacent_mask = torch.ones((B, num_proto, T), device=dist_all.device)

        indices = torch.empty((B, num_proto, 0), device=dist_all.device)
        values = torch.empty((B, num_proto, 0), device=dist_all.device)
        subpatch_ids = torch.empty((B, num_proto, 0), dtype=torch.long, device=dist_all.device)

        # We iterate over n_p subpatches
        for _ in range(n_p):
            # Mask out invalid positions
            dist_all_masked = dist_all + (1 - mask_all * adjacent_mask.unsqueeze(-1)) * (-1e5)
            # For each prototype, find the best position in T for each subpatch
            max_subs, max_subs_id = dist_all_masked.max(dim=2)        # (B, num_proto, n_p)
            max_sub_act, max_sub_act_id = max_subs.max(dim=-1)        # (B, num_proto)
            max_event_id = max_subs_id.gather(dim=2, index=max_sub_act_id.unsqueeze(-1))  # (B, num_proto, 1)

            # Build neighbor and directional masks
            neighbor_mask = self.neighboring_mask(max_event_id, T)
            if _ > 0:
                dir_mask = self.directional_mask(max_event_id, T)
                combined_mask = neighbor_mask * dir_mask
            else:
                combined_mask = neighbor_mask
            adjacent_mask = combined_mask

            # -- Vectorized approach to set mask_act=0 at positions [b, p, idx_val] --
            # Instead of calling .item() for each, we gather all indices and use scatter_.
            max_event_id_2d = max_event_id.squeeze(-1)  # shape (B, num_proto)
            # shape of mask_act is (B, num_proto, T), we want to zero out dimension=2 at index max_event_id_2d
            mask_act.scatter_(2, max_event_id_2d.unsqueeze(-1), 0)

            # Similarly, vectorize for mask_subpatch, shape (B, num_proto, n_p)
            # max_sub_act_id shape (B, num_proto)
            mask_subpatch.scatter_(2, max_sub_act_id.unsqueeze(-1), 0)

            # Now update mask_all to exclude the newly used positions
            mask_all = mask_all * mask_act.unsqueeze(-1)
            mask_all = mask_all.permute(0, 1, 3, 2)      # (B, num_proto, n_p, T)
            mask_all = mask_all * mask_subpatch.unsqueeze(-1)
            mask_all = mask_all.permute(0, 1, 3, 2)      # (B, num_proto, T, n_p)

            # Collect subpatch and event indices
            subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim=-1)
            indices = torch.cat([indices, max_event_id], dim=-1)
            values = torch.cat([values, max_sub_act.unsqueeze(-1)], dim=-1)

        # Reorder subpatches
        subpatch_ids = subpatch_ids.to(torch.int64)
        _, sub_indexes = subpatch_ids.sort(dim=-1)
        values_reordered = torch.gather(values, dim=-1, index=sub_indexes)
        indices_reordered = torch.gather(indices, dim=-1, index=sub_indexes)

        values_slot = values_reordered.clone() * (slots * n_p / factor)
        max_activation_slots = values_slot.sum(dim=-1)  # (B, num_proto)
        min_distances = n_p - max_activation_slots

        if get_f:
            # event_features is the original x
            return event_features, min_distances, indices_reordered
        return max_activation_slots, min_distances, indices_reordered


# ----- Model Definitions -----
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, attention_weights: Optional[list] = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or attention_weights is not None:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )
        self.attention_weights = attention_weights

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash and self.attention_weights is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            self.attention_weights.append(att.detach().cpu())
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, attention_weights: Optional[list] = None):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, attention_weights=attention_weights)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocabulary size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class Ethos(nn.Module):
    def __init__(self, config, return_attention=False):
        """
        Model definition. This module provides only the model structure.
        Original Ethos parameters are frozen so that only the timeline_matcher remains trainable.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.return_attention = return_attention
        self.attention_weights = [] if return_attention else None

        # Initialize the additional timeline matcher (trainable)
        self.timeline_matcher = TimelinePrototypeMatcher(
            num_prototypes=6,
            prototype_shape=(6, config.n_embd, 4),
            temp=1.0,
            patch_select=torch.ones(1, 6, 4),
            radius=2,
            direction='right'
        )

        # Original Ethos transformer components
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config, self.attention_weights) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # Freeze all original Ethos parameters so that only timeline_matcher remains trainable.
        # Comment to train whole model
        for name, param in self.named_parameters():
            if "timeline_matcher" not in name:
                param.requires_grad = False

    def get_num_params(self, non_embedding=True):
        """
        Returns the total number of parameters in the model.
        If non_embedding is True, subtracts the positional embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, context_length=0, return_representations=False):
        """
        Forward pass of the model.

        Args:
            idx (Tensor): Input token indices of shape (B, T).
            targets (Tensor, optional): Target token indices.
            context_length (int): Length of context to ignore for loss.
            return_representations (bool): If True, returns the final hidden states.

        Returns:
            If targets is provided: (logits, loss)
            Otherwise: (logits, None)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        if self.return_attention:
            self.attention_weights.clear()

        # Obtain token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through the transformer blocks first
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Now apply the timeline matcher to the transformer's output
        # It returns the original embeddings (which in this case are the transformer outputs)
        # and the min_distances computed from matching.
        event_features, min_distances, indices_reordered = self.timeline_matcher.greedy_distance(
            x, timeline_mask=torch.ones(x.shape[0], x.shape[1], device=x.device), get_f=True
        )
        # Compute a bias (scalar per sample) from min_distances and add it to the transformer outputs.
        bias = min_distances.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # shape: (B, 1, 1)
        x = event_features + bias

        # Compute final logits and loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="none",
            )
            if context_length:
                loss.view(logits.size()[:2])[:, :context_length] = 0
            loss = loss.mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if return_representations:
            return logits, loss, x.detach().cpu()
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Returns an AdamW optimizer configured to update only the trainable parameters.

        Args:
            weight_decay (float): Weight decay factor.
            learning_rate (float): Learning rate.
            betas (tuple): Beta parameters for AdamW.
            device_type (str): Device type, e.g., "cuda" or "cpu".

        Returns:
            optimizer (torch.optim.Optimizer): Configured AdamW optimizer.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate the model's MFLOPS utilization (MFU).

        Args:
            fwdbwd_per_iter (int): Number of forward/backward passes per iteration.
            dt (float): Time per iteration in seconds.

        Returns:
            mfu (float): Estimated MFU as a ratio.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt  # per second
        flops_promised = 312e12  # e.g., A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given an initial input.

        Args:
            idx (Tensor): Input token indices.
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): If provided, use top-k sampling.

        Returns:
            idx (Tensor): Concatenated tokens after generation.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def get_next_token(self, tokens, return_probs=False, top_k=None):
        """
        Get the next token given the current tokens.

        Args:
            tokens (Tensor): Current tokens.
            return_probs (bool): Whether to return token probabilities.
            return_representations (bool): Whether to return the hidden states.
            top_k (int, optional): If provided, use top-k sampling.

        Returns:
            Depending on flags, returns the next token along with probabilities and/or representations.
        """
        if tokens.size(1) > self.config.block_size:
            tokens = tokens[:, -self.config.block_size :]
        logits, _ = self(tokens)
        logits = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if return_probs:
            return next_token, probs
        return next_token
        # if tokens.size(1) > self.config.block_size:
        #     tokens = tokens[:, -self.config.block_size :]
        # logits, _, representations = self(tokens, return_representations=return_representations)
        # logits = logits[:, -1, :]
        # if top_k is not None:
        #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        #     logits[logits < v[:, [-1]]] = -float("Inf")
        # probs = F.softmax(logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1)
        # if return_representations:
        #     if return_probs:
        #         return next_token, probs, representations
        #     return next_token, representations
        # if return_probs:
        #     return next_token, probs
        # return next_token