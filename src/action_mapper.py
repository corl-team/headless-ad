from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.utils.misc import index_mask, orthogonal_


class ActionMapper(nn.Module):
    """
    This class contains functions for a work with actions' random embeddings.
    :param action_embed_dim: dimensionality of action embeddings
    :param num_actions: number of actions in an environment or a maximum amount of actions in all environments
    :param sim_measure: if 'loss_type=contrastive' will be a similarity measure type.
                        used for mapping from embedding space into action indices.
                        Is not used if 'loss_type=mse'
    :param loss_type: the type of the loss used
    """

    def __init__(
        self,
        action_embed_dim: int,
        num_actions: int,
        sim_measure: str,
        rand_emb_type: str,
        device: str,
    ):
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.num_actions = num_actions
        self.sim_measure = sim_measure
        self.device = device

        self.action_map = torch.empty((num_actions, action_embed_dim), device=device)
        self.action_map.requires_grad = False
        self.rand_emb_type = rand_emb_type

    @torch.no_grad()
    def regenerate(self, seed: Optional[int] = None) -> None:
        """
        Generate random embeddings for each action index.
        """
        if self.rand_emb_type == "orthogonal":
            if seed is not None:
                gen = torch.Generator(self.device).manual_seed(seed)
                orthogonal_(gen, self.action_map, gain=1)
            else:
                torch.nn.init.orthogonal_(self.action_map, gain=1)
        elif self.rand_emb_type == "normal":
            if seed is not None:
                gen = torch.Generator(self.device).manual_seed(seed)
                self.action_map = torch.randn(
                    *self.action_map.shape,
                    generator=gen,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                self.action_map = torch.randn(
                    *self.action_map.shape, device=self.device, dtype=torch.float32
                )
        else:
            raise NotImplementedError

    @torch.no_grad()
    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Map action indices into random embeddings.
        :param actions: a history of actions taken in an environment.
        """
        # action_map: [num_actions, d]
        # actions: [...]
        embeds = self.action_map[actions]
        embeds.requires_grad = False

        return embeds

    def _euclid_sim(self, preds, acts):
        """
        Calculates the similarities between predictions and action embeddings
        as the euclidean distance.
        """
        preds_expanded = preds.unsqueeze(-2)  # Shape becomes (N1, L, 1, d)
        acts_expanded = acts.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, N2, d)

        # Computing the MSE
        diff = preds_expanded - acts_expanded  # Difference
        sq_diff = diff**2  # Squared difference
        mse = sq_diff.mean(dim=-1)  # Mean along the last dimension (d)

        return mse

    def _get_sims(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        This function calculates the similarities between prediction vector and action embeddings.

        :param embeds: shape [b, t, d] - predicted embeddings for each action
        """
        if self.sim_measure == "cosine":
            embeds = embeds / torch.norm(embeds, p=2, dim=-1).unsqueeze(-1)
            sims = embeds @ self.action_map.T
        elif self.sim_measure == "dot":
            sims = embeds @ self.action_map.T
        elif self.sim_measure == "euclid":
            sims = -self._euclid_sim(preds=embeds, acts=self.action_map)
        else:
            raise NotImplementedError

        return sims

    def _get_probs(
        self, sims: torch.Tensor, num_actions_per_env: torch.Tensor
    ) -> torch.Tensor:
        """
        This function converts the similarities into a probability distribution.
        Next, it zeros out probabilities for unavailable arms.
        :param sims: shape [b, t, num_actions]. A tensor with similarities between
                     each prediction and each action embedding.
        :param num_actions_per_env: shape [b, t]. The amount of arms in each bandit.
        """

        # [b, num_actions]
        # Contains 1 when i < num_actions_per_env and 0 otherwise
        mask = index_mask(
            num_actions=num_actions_per_env, num_total_actions=self.action_map.shape[0]
        )

        probs = torch.nn.functional.softmax(sims, dim=-1)
        # these will not sum to 1, but Categorical will renormalize them
        probs = probs * mask.unsqueeze(1)

        return probs

    @torch.no_grad()
    def get_action(
        self,
        embeds: torch.Tensor,
        num_actions_per_env: torch.Tensor,
        with_entropy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps prediction vectors into action indices.
        Returns an action sampled with probability proportional to the similarities,
        a closest action in the embeddings space and the entropy of probability distribution.

        :param embeds: shape [b, t, d] - predicted embeddings for each action
        :param num_actions_per_env: shape [b, t]. The amount of arms in each bandit.
        :param with_entropy: if True, returns an entropy, otherwise doesn't
        """

        sims = self._get_sims(embeds=embeds)
        probs = self._get_probs(sims=sims, num_actions_per_env=num_actions_per_env)

        dist = torch.distributions.Categorical(probs=probs)
        actions_sample, actions_argmax = dist.sample(), dist.probs.argmax(-1)
        assert torch.all(actions_sample < num_actions_per_env.unsqueeze(-1))
        assert torch.all(actions_argmax < num_actions_per_env.unsqueeze(-1))
        if with_entropy:
            return actions_sample, actions_argmax, dist.entropy()
        else:
            return actions_sample, actions_argmax

    @torch.no_grad()
    def _get_action_map_as_context(
        self, num_actions_per_env: Optional[torch.Tensor] = None
    ):
        """
        Returns the list of embeddings for a later use as the action set prompt.
        Zeros out embeddings for unavailable actions.

        :param num_actions_per_env: the amount of arms in each bandit. This is how much
                                    actions will not be zeroed out in each environment.
        """
        batch_size = num_actions_per_env.shape[0]
        actions = torch.tile(self.action_map.unsqueeze(0), (batch_size, 1, 1))
        mask = index_mask(
            num_actions=num_actions_per_env, num_total_actions=self.num_actions
        )
        actions *= mask.unsqueeze(-1)

        return actions
