"""Decoupled marked temporal point process with neural ODE influences.

This module implements the linear Dec-ODE baseline from Song et al. (ICLR
2024), "Decoupled Marked Temporal Point Process Using Neural ODEs". The model
keeps each historical event as an independently propagated influence trajectory
``h(t; e_i)`` and reconstructs marked intensities as
``lambda(t, k) = lambda_g(t) * f(k | t)``.

Common hyperparameter ranges reported or used by the paper:
    - ``hidden_size``: 32 or 64 for each event influence state.
    - ``model_specs.ode_hidden_size``: 128 or 256 for the ODE network width.
    - ``model_specs.num_ode_layers``: 3, 4, or 5.
    - ``model_specs.ode_steps``: 16 during training and 64 during testing.
    - ``model_specs.ode_solver``: ``euler`` for speed or ``rk4`` for accuracy.
    - ``model_specs.combine_mode``: ``linear`` for the main paper baseline;
      ``nonlinear`` for the appendix variant with inhibitory effects.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class _InfluenceODE(nn.Module):
    """Mark-conditioned neural ODE vector field for event influences."""

    def __init__(
        self,
        hidden_size: int,
        mark_emb_size: int,
        ode_hidden_size: int,
        num_ode_layers: int,
        dropout_rate: float,
    ) -> None:
        """Initialize the ODE vector field.

        Args:
            hidden_size: Dimension of ``h(t; e_i)``.
            mark_emb_size: Dimension of the event mark context embedding.
            ode_hidden_size: Width of hidden MLP layers.
            num_ode_layers: Number of hidden MLP layers before the output.
            dropout_rate: Dropout probability between hidden layers.
        """
        super().__init__()
        if num_ode_layers < 1:
            raise ValueError("num_ode_layers must be at least 1")

        # Song et al. define gamma(h(t; e_i), t, k_i; theta).  The scalar time
        # coordinate is therefore an explicit input rather than an implicit
        # property of the fixed-step solver.
        input_size = hidden_size + mark_emb_size + 1
        layers = []
        for layer_index in range(num_ode_layers):
            layers.append(
                nn.Linear(
                    input_size if layer_index == 0 else ode_hidden_size,
                    ode_hidden_size,
                ),
            )
            layers.append(nn.Tanh())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(ode_hidden_size, hidden_size))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        mark_embedding: torch.Tensor,
        current_time: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``dh / dt``.

        Args:
            state: Influence state with final dimension ``hidden_size``.
            mark_embedding: Broadcast-compatible mark context embedding.
            current_time: Broadcast-compatible scalar time feature.

        Returns:
            Derivative tensor with the same shape as ``state``.
        """
        return self.network(torch.cat([state, mark_embedding, current_time], dim=-1))


class Decoupled(TorchBaseModel):
    """Linear Dec-ODE marked temporal point process."""

    def __init__(
        self,
        model_config: Any,
    ) -> None:
        """Initialize Decoupled from an EasyTPP model config.

        Args:
            model_config: EasyTPP model configuration. Extra Dec-ODE knobs are
                read from ``model_config.model_specs``.
        """
        super().__init__(model_config)
        specs = model_config.model_specs
        self.ode_steps = int(specs.get("ode_steps", 16))
        self.combine_mode = specs.get("combine_mode", "linear").lower()
        self.intensity_floor = float(specs.get("intensity_floor", 1e-8))

        solver_name = specs.get("ode_solver", "euler").lower()
        if solver_name not in {"euler", "rk4"}:
            raise ValueError("Decoupled ode_solver must be 'euler' or 'rk4'")
        self.solver_name = solver_name
        if self.combine_mode not in {"linear", "nonlinear"}:
            raise ValueError("Decoupled combine_mode must be 'linear' or 'nonlinear'")

        ode_hidden_size = int(specs.get("ode_hidden_size", 256))
        num_ode_layers = int(specs.get("num_ode_layers", 3))
        ode_dropout = float(specs.get("ode_dropout", model_config.dropout_rate))

        # W_e(k) in Eq. (6): initial state for the influence created by mark k.
        self.event_state_emb = nn.Embedding(
            self.num_event_types_pad,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )
        # Mark context supplies k_i; the scalar time input is passed at every
        # ODE step so gamma matches the paper's gamma(h, t, k).
        self.mark_context_emb = nn.Embedding(
            self.num_event_types_pad,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )
        self.ode_func = _InfluenceODE(
            hidden_size=self.hidden_size,
            mark_emb_size=self.hidden_size,
            ode_hidden_size=ode_hidden_size,
            num_ode_layers=num_ode_layers,
            dropout_rate=ode_dropout,
        )
        self.ground_head = nn.Linear(self.hidden_size, 1)
        self.mark_head = nn.Linear(self.hidden_size, self.num_event_types)

        # A learned background keeps lambda positive before any non-padding
        # history exists, which matters for datasets with a left-window pad mark.
        self.background_ground = nn.Parameter(torch.zeros(1))
        self.background_mark_logits = nn.Parameter(torch.zeros(self.num_event_types))

    def _valid_event_mask(
        self,
        type_seqs: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the mask for events that may influence future intensities.

        Args:
            type_seqs: Event type ids with shape ``[batch_size, seq_len]``.
            seq_mask: Optional EasyTPP non-padding mask with the same shape.

        Returns:
            Boolean tensor that is true only for observed non-padding marks.
        """
        valid_mask = type_seqs.lt(self.num_event_types)
        if seq_mask is not None:
            valid_mask = valid_mask & seq_mask.bool()
        return valid_mask

    def _integrate_influences(
        self,
        initial_state: torch.Tensor,
        mark_embedding: torch.Tensor,
        delta_time: torch.Tensor,
        start_time: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate independent event influences to query offsets.

        Args:
            initial_state: Initial event states, shape ``[..., hidden_size]``.
            mark_embedding: Mark context embeddings, same leading dimensions.
            delta_time: Non-negative offsets from source time to query time.
            start_time: Absolute source-event times broadcastable to
                ``delta_time``.  Passing absolute time follows the paper's
                ``gamma(h(t; e_i), t, k_i)`` dynamics and keeps same-offset
                events distinguishable when they occur at different times.

        Returns:
            Propagated states with shape ``[..., hidden_size]``.
        """
        solver_steps = max(self.ode_steps, 1)
        step_dt = (delta_time.clamp_min(0.0) / solver_steps).unsqueeze(-1)
        current_time = start_time.clamp_min(0.0).unsqueeze(-1) + torch.zeros_like(
            step_dt,
        )
        broadcast_base = torch.zeros(
            *step_dt.shape[:-1],
            self.hidden_size,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        state = initial_state + broadcast_base
        mark_embedding = mark_embedding + broadcast_base

        for _ in range(solver_steps):
            state = self._ode_step(
                state=state,
                mark_embedding=mark_embedding,
                current_time=current_time,
                step_dt=step_dt,
            )
            current_time = current_time + step_dt
        return state

    def _ode_step(
        self,
        state: torch.Tensor,
        mark_embedding: torch.Tensor,
        current_time: torch.Tensor,
        step_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Take one time-aware Euler or RK4 step for Dec-ODE influences."""
        if self.solver_name == "euler":
            derivative = self.ode_func(state, mark_embedding, current_time)
            return state + step_dt * derivative

        half_dt = step_dt / 2.0
        k1 = self.ode_func(state, mark_embedding, current_time)
        k2 = self.ode_func(
            state + half_dt * k1,
            mark_embedding,
            current_time + half_dt,
        )
        k3 = self.ode_func(
            state + half_dt * k2,
            mark_embedding,
            current_time + half_dt,
        )
        k4 = self.ode_func(
            state + step_dt * k3,
            mark_embedding,
            current_time + step_dt,
        )
        return state + step_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _combine_influences(
        self,
        influence_states: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode event influences and combine them into marked intensities.

        Args:
            influence_states: Propagated states, shape
                ``[..., num_sources, hidden_size]``.
            source_mask: Boolean source mask, shape ``[..., num_sources]``.

        Returns:
            Marked intensities, shape ``[..., num_event_types]``.
        """
        mask = source_mask.to(influence_states.dtype).unsqueeze(-1)
        raw_ground = self.ground_head(influence_states).squeeze(-1)
        mark_logits = self.mark_head(influence_states) * mask

        if self.combine_mode == "linear":
            # Main-paper Dec-ODE: every event contributes a non-negative scalar.
            ground = F.softplus(raw_ground) * source_mask.to(raw_ground.dtype)
            ground = ground.sum(dim=-1) + F.softplus(self.background_ground)
        else:
            # Appendix A.1: summing before softplus allows inhibitory effects.
            raw_sum = (raw_ground * source_mask.to(raw_ground.dtype)).sum(dim=-1)
            ground = F.softplus(raw_sum + self.background_ground)

        mark_logits = mark_logits.sum(dim=-2) + self.background_mark_logits
        mark_probs = torch.softmax(mark_logits, dim=-1)
        return ground.clamp_min(self.intensity_floor).unsqueeze(-1) * mark_probs

    def _intensity_at_event_positions(
        self,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute left-limit intensities at observed event positions.

        Args:
            time_seqs: Absolute event times, shape ``[batch_size, seq_len]``.
            type_seqs: Event type ids, shape ``[batch_size, seq_len]``.
            seq_mask: EasyTPP non-padding mask, shape ``[batch_size, seq_len]``.

        Returns:
            Marked intensities for targets ``t_1 ... t_N``, shape
            ``[batch_size, seq_len - 1, num_event_types]``.
        """
        source_states = self.event_state_emb(type_seqs)
        source_marks = self.mark_context_emb(type_seqs)
        valid_sources = self._valid_event_mask(type_seqs, seq_mask)
        intensities = []

        for target_index in range(1, time_seqs.size(1)):
            source_slice = slice(0, target_index)
            delta_time = time_seqs[:, target_index, None] - time_seqs[:, source_slice]
            query_states = self._integrate_influences(
                initial_state=source_states[:, source_slice, :],
                mark_embedding=source_marks[:, source_slice, :],
                delta_time=delta_time,
                start_time=time_seqs[:, source_slice],
            )
            intensities.append(
                self._combine_influences(
                    influence_states=query_states,
                    source_mask=valid_sources[:, source_slice],
                ),
            )
        return torch.stack(intensities, dim=1)

    def loglike_loss(
        self,
        batch: Tuple[torch.Tensor, ...],
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Compute EasyTPP negative log-likelihood loss.

        Args:
            batch: EasyTPP batch tuple containing times, deltas, marks,
                non-padding mask, and attention mask.
            **kwargs: Unused. Kept for interface compatibility.

        Returns:
            Tuple of total loss and number of events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        lambda_at_event = self._intensity_at_event_positions(
            time_seqs=time_seqs,
            type_seqs=type_seqs,
            seq_mask=batch_non_pad_mask,
        )
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])
        lambda_t_sample = self.compute_intensities_at_sample_times(
            time_seqs=time_seqs[:, :-1],
            time_delta_seqs=time_delta_seqs[:, :-1],
            type_seqs=type_seqs[:, :-1],
            sample_dtimes=sample_dtimes,
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            time_delta_seq=time_delta_seqs[:, 1:],
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:],
        )
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_intensities_at_sample_times(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        sample_dtimes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute marked intensities after each observed prefix event.

        Args:
            time_seqs: Absolute prefix event times, shape ``[batch_size, seq_len]``.
            time_delta_seqs: Inter-event deltas, kept for the EasyTPP interface.
            type_seqs: Prefix event type ids, shape ``[batch_size, seq_len]``.
            sample_dtimes: Offsets after each prefix event, shape
                ``[batch_size, seq_len, num_samples]``.
            **kwargs: Supports EasyTPP's ``compute_last_step_only`` flag.

        Returns:
            Marked intensities with shape
            ``[batch_size, seq_len, num_samples, num_event_types]``.
        """
        del time_delta_seqs
        compute_last_step_only = kwargs.get("compute_last_step_only", False)
        if compute_last_step_only:
            target_indices = [time_seqs.size(1) - 1]
            sample_dtimes = sample_dtimes[:, -1:, :]
        else:
            target_indices = list(range(time_seqs.size(1)))

        source_states = self.event_state_emb(type_seqs)
        source_marks = self.mark_context_emb(type_seqs)
        valid_sources = self._valid_event_mask(type_seqs)
        sampled_intensities = []

        for output_index, target_index in enumerate(target_indices):
            source_slice = slice(0, target_index + 1)
            query_times = (
                time_seqs[:, target_index, None] + sample_dtimes[:, output_index, :]
            )
            delta_time = query_times[:, None, :] - time_seqs[:, source_slice, None]
            history_mask = valid_sources[:, source_slice, None].expand_as(delta_time)
            query_states = self._integrate_influences(
                initial_state=source_states[:, source_slice, None, :],
                mark_embedding=source_marks[:, source_slice, None, :],
                delta_time=delta_time,
                start_time=time_seqs[:, source_slice, None],
            )
            sampled_intensities.append(
                self._combine_influences(
                    influence_states=query_states.transpose(1, 2),
                    source_mask=history_mask.transpose(1, 2),
                ),
            )

        return torch.stack(sampled_intensities, dim=1)
