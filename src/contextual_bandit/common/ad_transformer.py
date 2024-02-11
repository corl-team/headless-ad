import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from xformers.ops import SwiGLU

from src.tiny_llama.config import Config
from src.tiny_llama.transformer import Block, LLaMAMLP, build_rope_cache

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class Model(nn.Module):
    def __init__(
        self, config: Config, token_dim: int, num_actions: int, state_dim: int
    ) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                proj=nn.Linear(token_dim, config.n_embd),
                reward_emb=nn.Linear(1, token_dim),
                action_emb=nn.Embedding(num_actions, token_dim),
                state_emb=nn.Linear(state_dim, token_dim),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(
                    config.n_embd, eps=config.norm_eps, dropout=config.dropout
                ),
                head=nn.Linear(config.n_embd, num_actions, bias=False),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.n_embd = config.n_embd
        self.token_dim = token_dim

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        # print module name
        if isinstance(module, nn.Embedding):
            # RWKV: set it to 1e-4
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1))
            )
            # torch.nn.init.normal_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            # fan-in variance scaling intializer
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1))
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (
                name == "w3.weight" and isinstance(module, SwiGLU)
            ):  # if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(p.shape[-1]) / n_layer)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        max_seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        b, t = rewards.shape[:2]
        T = rewards.shape[1] * 3
        device = rewards.device

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            block_size >= T
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(device=device)

        cos, sin = self.rope_cache

        cos = cos[:T]
        sin = sin[:T]

        # Forward
        state_emb = self.transformer.state_emb(states)
        rew_emb = self.transformer.reward_emb(rewards.unsqueeze(-1))
        act_emb = self.transformer.action_emb(actions)

        sequence = (
            torch.stack([state_emb, act_emb, rew_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * t, self.token_dim)
        )

        x = self.transformer.proj(sequence)

        for block in self.transformer.h:
            x, *_ = block(x, (cos, sin), max_seq_length)

        x = x[:, 0::3]

        logits = self.transformer.head(self.transformer.ln_f(x))

        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, device: torch.device) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=device,
            condense_ratio=self.config.condense_ratio,
        )
