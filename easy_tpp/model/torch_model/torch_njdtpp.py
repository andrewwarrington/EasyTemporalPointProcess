"""Neural Jump-Diffusion Temporal Point Process for EasyTPP.

This module implements the PyTorch side of Zhang et al. (ICML 2024),
"Neural Jump-Diffusion Temporal Point Processes".  The paper models the
marked log-intensity vector directly as a neural jump-diffusion SDE,

    d eta_t = f(eta_t) dt + g(eta_t) dW_t + h(eta_t) dN_t,

where eta_t[m] = log lambda_m(t).  EasyTPP already owns batching, masking, and
negative log-likelihood aggregation, so this file focuses on producing marked
intensities at event and sampled times.

Common hyperparameter ranges reported or used by the paper/released code:
    - ``hidden_size`` / ``model_specs.hidden_size``: 16, 32, 64.
    - ``model_specs.num_hidden_layers``: 1, 2, 3; the paper uses 2 in most
      experiments.
    - ``model_specs.num_sde_steps``: 10 for training-time Euler-Maruyama
      subdivision between events in the released scripts.
    - ``model_specs.activation``: ``tanh`` in the paper; ``relu`` and
      ``sigmoid`` are also Lipschitz activations discussed by the authors.
    - optimizer defaults from the paper: Adam with lr in {1e-3, 1e-2, 1e-1}
      and weight decay 1e-5.  The public implementation uses lr 1e-3 for
      the neural SDE and lr 1e-1 for ``eta0``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class _NJDMLP(nn.Module):
    """Small MLP used for the drift, diffusion, and jump coefficient nets."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation: str,
        init_std: float,
    ) -> None:
        """Initialize a coefficient network.

        Args:
            input_size: Size of the log-intensity state vector.
            output_size: Number of coefficient values to produce.
            hidden_size: Width of each hidden layer.
            num_hidden_layers: Number of hidden layers before the output.
            activation: Name of the nonlinearity used between linear layers.
            init_std: Standard deviation for Gaussian weight initialization.
        """
        super().__init__()
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")

        if activation.lower() == "tanh":
            activation_layer: nn.Module = nn.Tanh()
        elif activation.lower() == "relu":
            activation_layer = nn.ReLU()
        elif activation.lower() == "sigmoid":
            activation_layer = nn.Sigmoid()
        else:
            raise ValueError(
                "NJDTPP activation must be one of {'tanh', 'relu', 'sigmoid'}",
            )

        layer_sizes = [input_size]
        layer_sizes.extend([hidden_size] * num_hidden_layers)
        layer_sizes.append(output_size)

        layers = []
        for layer_index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[layer_index], layer_sizes[layer_index + 1])
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            nn.init.uniform_(linear.bias, a=-init_std, b=init_std)
            layers.append(linear)
            if layer_index < len(layer_sizes) - 2:
                layers.append(activation_layer)
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the coefficient network.

        Args:
            state: Log-intensity state with final dimension ``num_event_types``.

        Returns:
            Coefficients with the same leading dimensions as ``state``.
        """
        return self.network(state)


class NJDTPP(TorchBaseModel):
    """Neural Jump-Diffusion Temporal Point Process.

    The model keeps an M-dimensional log-intensity vector, where M is the
    number of non-padding event types.  Between observed events the vector is
    advanced with a fixed-step Euler-Maruyama solver.  At an observed event of
    type k, the k-th column of the learned jump matrix h(eta) is added to eta,
    matching Eq. (18)-(19) in Zhang et al.
    """

    def __init__(
        self,
        model_config: Any,
    ) -> None:
        """Initialize NJDTPP from an EasyTPP model config.

        Args:
            model_config: EasyTPP model configuration.  Extra NJDTPP knobs are
                read from ``model_config.model_specs``.
        """
        super().__init__(model_config)
        specs = model_config.model_specs
        coefficient_hidden_size = specs.get("hidden_size", self.hidden_size)
        num_hidden_layers = specs.get("num_hidden_layers", model_config.num_layers)
        activation = specs.get("activation", "tanh")
        init_std = specs.get("init_std", 0.01)

        # Euler-Maruyama substeps.  The public NJDTPP code calls this
        # ``num_divide`` and uses 10 for real-world experiments.
        self.num_sde_steps = int(specs.get("num_sde_steps", 10))
        self.num_prediction_samples = int(specs.get("num_prediction_samples", 1000))
        self.diffusion_scale = float(specs.get("diffusion_scale", 1.0))
        self.log_intensity_clip = float(specs.get("log_intensity_clip", 20.0))
        self.eta0_learning_rate = float(specs.get("eta0_learning_rate", 1.0e-1))
        # eta0 is learned directly, as in the released NJDTPP implementation.
        eta0_mean = float(specs.get("eta0_init_mean", 0.0))
        eta0_std = float(specs.get("eta0_init_std", 0.1))
        self.eta0 = nn.Parameter(
            torch.empty(self.num_event_types).normal_(mean=eta0_mean, std=eta0_std),
        )

        self.drift_net = _NJDMLP(
            input_size=self.num_event_types,
            output_size=self.num_event_types,
            hidden_size=coefficient_hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            init_std=init_std,
        )
        self.diffusion_net = _NJDMLP(
            input_size=self.num_event_types,
            output_size=self.num_event_types,
            hidden_size=coefficient_hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            init_std=init_std,
        )
        self.jump_net = _NJDMLP(
            input_size=self.num_event_types,
            output_size=self.num_event_types * self.num_event_types,
            hidden_size=coefficient_hidden_size,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            init_std=init_std,
        )

    def _clamp_eta(
        self,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp log-intensities before exponentiating.

        Args:
            eta: Log-intensity tensor.

        Returns:
            Numerically bounded log-intensity tensor.
        """
        return eta.clamp(
            min=-self.log_intensity_clip,
            max=self.log_intensity_clip,
        )

    def _noise_scale(self) -> float:
        """Return the Brownian noise scale."""
        return self.diffusion_scale

    def _apply_jump(
        self,
        eta_left: torch.Tensor,
        event_types: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the mark-conditioned jump update at observed events.

        Args:
            eta_left: Left-limit log-intensity state, shape ``[..., M]``.
            event_types: Event mark ids aligned with ``eta_left`` leading dims.
            event_mask: Boolean mask indicating non-padding observed events.

        Returns:
            Right-limit log-intensity after the event jump.
        """
        original_shape = eta_left.shape
        flat_eta = eta_left.reshape(-1, self.num_event_types)
        flat_types = event_types.reshape(-1).clamp(
            min=0,
            max=self.num_event_types - 1,
        )
        flat_mask = event_mask.reshape(-1).bool()

        jump_matrix = self.jump_net(flat_eta).view(
            -1,
            self.num_event_types,
            self.num_event_types,
        )
        gather_index = flat_types[:, None, None].expand(
            -1,
            self.num_event_types,
            1,
        )
        selected_jump = jump_matrix.gather(dim=2, index=gather_index).squeeze(-1)
        selected_jump = selected_jump * flat_mask[:, None].to(selected_jump.dtype)

        eta_right = flat_eta + selected_jump
        return self._clamp_eta(eta_right.view(original_shape))

    def _euler_maruyama_step(
        self,
        eta_state: torch.Tensor,
        step_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Take one Euler-Maruyama step of the neural jump-diffusion SDE."""
        drift = self.drift_net(eta_state) * step_dt
        noise_scale = self._noise_scale()
        if noise_scale > 0.0:
            brownian_increment = torch.randn_like(eta_state) * step_dt.sqrt()
            diffusion = self.diffusion_net(eta_state) * brownian_increment * noise_scale
        else:
            diffusion = torch.zeros_like(eta_state)
        return self._clamp_eta(eta_state + drift + diffusion)

    def _euler_maruyama(
        self,
        eta_initial: torch.Tensor,
        delta_time: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Advance eta over a time interval with Euler-Maruyama.

        Args:
            eta_initial: Initial log-intensity state, shape ``[..., M]``.
            delta_time: Interval length, shape matching ``eta_initial[..., 0]``.
            num_steps: Optional override for the number of solver substeps.

        Returns:
            Left-limit state at the end of the interval.
        """
        solver_steps = max(int(num_steps or self.num_sde_steps), 1)
        eta_state = eta_initial
        step_dt = (delta_time.clamp_min(0.0) / solver_steps).unsqueeze(-1)

        for _ in range(solver_steps):
            eta_state = self._euler_maruyama_step(eta_state, step_dt)
        return eta_state

    def _make_fixed_grid_dtimes(
        self,
        delta_time: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Return the fixed Euler grid used to record one SDE path."""
        solver_steps = max(int(num_steps), 1)
        ratios = torch.linspace(
            start=0.0,
            end=1.0,
            steps=solver_steps + 1,
            device=delta_time.device,
            dtype=delta_time.dtype,
        )
        return delta_time.clamp_min(0.0).unsqueeze(-1) * ratios

    def _solve_fixed_grid_path(
        self,
        eta_initial: torch.Tensor,
        delta_time: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Advance one Euler-Maruyama path and record every grid state.

        The official NJDTPP ``EulerSolver`` samples one Brownian trajectory per
        interval, records the initial state, then records each fixed Euler step.
        This helper mirrors that contract and avoids adding stochastic steps at
        arbitrary query/sample times.
        """
        solver_steps = max(int(num_steps or self.num_sde_steps), 1)
        eta_state = eta_initial
        step_dt = (delta_time.clamp_min(0.0) / solver_steps).unsqueeze(-1)
        path_states = [eta_state]

        for _ in range(solver_steps):
            eta_state = self._euler_maruyama_step(eta_state, step_dt)
            path_states.append(eta_state)

        return torch.stack(path_states, dim=-2)

    def _make_loglike_sample_dtimes(
        self,
        time_delta_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Create likelihood samples aligned with the NJDTPP SDE solver grid.

        The released ``Zh-Shuai/NJDTPP`` code evaluates the likelihood integral
        on the ``EulerSolver`` trajectory, which records ``num_divide + 1``
        states per interval. For trapezoid integration we mirror that decision
        with ``num_sde_steps + 1`` points. If EasyTPP's configured loss samples
        are explicitly enabled, we still evolve them along one shared interval
        path.
        """
        if self.use_mc_samples:
            num_steps = max(self.loss_integral_num_sample_per_step - 1, 1)
        else:
            num_steps = max(self.num_sde_steps, 1)
        return self._make_fixed_grid_dtimes(time_delta_seq, num_steps)

    def _solve_path_at_sorted_times(
        self,
        eta_initial: torch.Tensor,
        sorted_dtimes: torch.Tensor,
        final_dtime: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Advance one fixed-grid interval path and read states at sorted offsets.

        Zhang et al. (ICML 2024, Eq. 18-21) define the marked log-intensity as
        a neural jump-diffusion SDE and use Euler-Maruyama trajectories for the
        likelihood integral.  The official ``Zh-Shuai/NJDTPP`` implementation's
        ``EulerSolver`` similarly records states along one path per interval.
        This helper keeps queried intensities on a shared fixed Euler path
        instead of inserting extra Brownian increments at every query offset.

        Args:
            eta_initial: Right-limit states at interval starts, shape ``[..., M]``.
            sorted_dtimes: Monotone offsets inside each interval, shape
                ``[..., num_samples]``.
            final_dtime: Optional interval endpoint.  When provided, the returned
                final state is advanced along the same path after the samples.

        Returns:
            Pair of sampled states ``[..., num_samples, M]`` and optional final
            state ``[..., M]``.
        """
        solver_steps = max(int(num_steps or max(sorted_dtimes.size(-1) - 1, 1)), 1)
        num_samples = sorted_dtimes.size(-1)
        original_shape = eta_initial.shape[:-1]

        flat_eta = eta_initial.reshape(-1, self.num_event_types)
        flat_samples = sorted_dtimes.reshape(-1, num_samples).clamp_min(0.0)
        if final_dtime is None:
            flat_final_time = flat_samples[:, -1]
        else:
            flat_final_time = final_dtime.reshape(-1).clamp_min(0.0)

        flat_grid_states = self._solve_fixed_grid_path(
            eta_initial=flat_eta,
            delta_time=flat_final_time,
            num_steps=solver_steps,
        )
        safe_final_time = flat_final_time[:, None].clamp_min(torch.finfo(flat_eta.dtype).eps)
        clamped_samples = torch.minimum(flat_samples, flat_final_time[:, None])
        sample_ratios = torch.where(
            flat_final_time[:, None] > 0.0,
            clamped_samples / safe_final_time,
            torch.zeros_like(clamped_samples),
        )
        nearest_grid_index = (sample_ratios * solver_steps).round().long().clamp(
            min=0,
            max=solver_steps,
        )
        gather_index = nearest_grid_index.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_event_types,
        )
        flat_sampled_states = flat_grid_states.gather(dim=1, index=gather_index)

        sampled_states = flat_sampled_states.view(
            *original_shape,
            num_samples,
            self.num_event_types,
        )
        final_state = None
        if final_dtime is not None:
            final_state = flat_grid_states[:, -1, :].view(
                *original_shape,
                self.num_event_types,
            )

        return sampled_states, final_state

    def _solve_at_sample_times(
        self,
        eta_initial: torch.Tensor,
        sample_dtimes: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Advance each interval to many sampled offsets on a shared path."""
        sorted_dtimes, sort_indices = sample_dtimes.clamp_min(0.0).sort(dim=-1)
        sorted_states, _ = self._solve_path_at_sorted_times(
            eta_initial=eta_initial,
            sorted_dtimes=sorted_dtimes,
            num_steps=num_steps,
        )
        inverse_indices = sort_indices.argsort(dim=-1)
        gather_index = inverse_indices.unsqueeze(-1).expand(
            *inverse_indices.shape,
            self.num_event_types,
        )
        return sorted_states.gather(dim=-2, index=gather_index)

    def _compute_loglike_states(
        self,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_mask: torch.Tensor,
        sample_dtimes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute event and integral states from one path per interval.

        This mirrors the paper likelihood: the event term uses the left-limit
        log-intensity at the next event, while the non-event term integrates
        ``sum_m exp(eta_m(t))`` over the interval.  Sharing the path avoids
        mixing independent Brownian draws inside a single interval integral.
        """
        batch_size, seq_len = time_delta_seqs.shape

        eta_left = self.eta0.unsqueeze(0).expand(batch_size, -1)
        first_event_mask = (
            seq_mask[:, 0].bool()
            & (type_seqs[:, 0] >= 0)
            & (type_seqs[:, 0] < self.num_event_types)
        )
        eta_right = self._apply_jump(
            eta_left=eta_left,
            event_types=type_seqs[:, 0],
            event_mask=first_event_mask,
        )

        solver_steps = max(sample_dtimes.size(-1) - 1, 1)
        event_left_states = []
        sample_states = []
        for event_index in range(1, seq_len):
            interval_states = self._solve_fixed_grid_path(
                eta_initial=eta_right,
                delta_time=time_delta_seqs[:, event_index],
                num_steps=solver_steps,
            )
            eta_left = interval_states[:, -1, :]
            sample_states.append(interval_states)
            event_left_states.append(eta_left)

            event_mask = (
                seq_mask[:, event_index].bool()
                & (type_seqs[:, event_index] >= 0)
                & (type_seqs[:, event_index] < self.num_event_types)
            )
            eta_right = self._apply_jump(
                eta_left=eta_left,
                event_types=type_seqs[:, event_index],
                event_mask=event_mask,
            )

        return torch.stack(event_left_states, dim=1), torch.stack(sample_states, dim=1)

    def _compute_event_loglike(
        self,
        left_states: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute marked event log-likelihood from NJDTPP log-intensities."""
        target_types = type_seqs[:, 1:]
        valid_event_mask = (
            seq_mask[:, 1:].bool()
            & (target_types >= 0)
            & (target_types < self.num_event_types)
        )
        safe_types = target_types.clamp(min=0, max=self.num_event_types - 1)
        event_loglike = self._clamp_eta(left_states).gather(
            dim=-1,
            index=safe_types.unsqueeze(-1),
        ).squeeze(-1)
        event_loglike = event_loglike * valid_event_mask.to(event_loglike.dtype)
        return event_loglike, valid_event_mask, int(valid_event_mask.sum().item())

    def _compute_integral_loglike(
        self,
        lambda_t_sample: torch.Tensor,
        sample_dtimes: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate total intensity with trapezoid widths from sample times."""
        total_lambda = lambda_t_sample.sum(dim=-1)
        sorted_dtimes, sort_indices = sample_dtimes.clamp_min(0.0).sort(dim=-1)
        sorted_lambda = total_lambda.gather(dim=-1, index=sort_indices)
        sample_widths = (sorted_dtimes[..., 1:] - sorted_dtimes[..., :-1]).clamp_min(
            0.0,
        )
        interval_integrals = (
            0.5
            * (sorted_lambda[..., 1:] + sorted_lambda[..., :-1])
            * sample_widths
        ).sum(dim=-1)
        return interval_integrals * seq_mask.to(interval_integrals.dtype)

    def _compute_event_states(
        self,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute left- and right-limit eta states at all sequence positions.

        Args:
            time_delta_seqs: Inter-event deltas, shape ``[batch_size, seq_len]``.
            type_seqs: Event type ids, shape ``[batch_size, seq_len]``.
            seq_mask: Boolean non-padding mask, shape ``[batch_size, seq_len]``.

        Returns:
            Pair ``(left_states, right_states)`` with shape
            ``[batch_size, seq_len, M]`` each.
        """
        batch_size, seq_len = time_delta_seqs.shape
        left_states = []
        right_states = []

        eta_left = self.eta0.unsqueeze(0).expand(batch_size, -1)
        for event_index in range(seq_len):
            if event_index > 0:
                eta_left = self._euler_maruyama(
                    eta_initial=right_states[-1],
                    delta_time=time_delta_seqs[:, event_index],
                )

            event_mask = (
                seq_mask[:, event_index].bool()
                & (type_seqs[:, event_index] >= 0)
                & (type_seqs[:, event_index] < self.num_event_types)
            )
            eta_right = self._apply_jump(
                eta_left=eta_left,
                event_types=type_seqs[:, event_index],
                event_mask=event_mask,
            )

            left_states.append(eta_left)
            right_states.append(eta_right)

        return torch.stack(left_states, dim=1), torch.stack(right_states, dim=1)

    def loglike_loss(
        self,
        batch: Tuple[torch.Tensor, ...],
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Compute EasyTPP negative log-likelihood loss.

        Args:
            batch: EasyTPP batch tuple containing times, deltas, marks, masks,
                and attention masks.
            **kwargs: Unused. Kept for interface compatibility.

        Returns:
            Tuple ``(loss, num_events)``.
        """
        _, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        sample_dtimes = self._make_loglike_sample_dtimes(time_delta_seqs[:, 1:])
        left_states, sample_states = self._compute_loglike_states(
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            seq_mask=batch_non_pad_mask,
            sample_dtimes=sample_dtimes,
        )
        lambda_t_sample = torch.exp(self._clamp_eta(sample_states))

        event_ll, valid_event_mask, num_events = self._compute_event_loglike(
            left_states=left_states,
            type_seqs=type_seqs,
            seq_mask=batch_non_pad_mask,
        )
        non_event_ll = self._compute_integral_loglike(
            lambda_t_sample=lambda_t_sample,
            sample_dtimes=sample_dtimes,
            seq_mask=batch_non_pad_mask[:, 1:],
        )
        non_event_ll = non_event_ll * valid_event_mask.to(non_event_ll.dtype)
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events

    def get_optimizer_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """Return optimizer groups that keep eta0 at a larger learning rate."""
        eta0_lr = self.eta0_learning_rate
        eta0_id = id(self.eta0)
        sde_params = [
            param for param in self.parameters()
            if param.requires_grad and id(param) != eta0_id
        ]
        return [
            {
                "params": sde_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "lr_scale": 1.0,
            },
            {
                "params": [self.eta0],
                "lr": eta0_lr,
                "weight_decay": weight_decay,
                "lr_scale": eta0_lr / base_lr if base_lr > 0 else 1.0,
            },
        ]

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
            time_seqs: Absolute event times, shape ``[batch_size, seq_len]``.
                The values are unused because NJDTPP evolves by inter-event
                deltas, but the argument is kept for the EasyTPP interface.
            time_delta_seqs: Inter-event deltas, shape ``[batch_size, seq_len]``.
            type_seqs: Event type ids, shape ``[batch_size, seq_len]``.
            sample_dtimes: Offsets after each prefix event, shape
                ``[batch_size, seq_len, num_samples]``.
            **kwargs: Supports EasyTPP's ``compute_last_step_only`` flag.

        Returns:
            Marked intensities with shape
            ``[batch_size, seq_len, num_samples, num_event_types]``.
        """
        del time_seqs
        compute_last_step_only = kwargs.get("compute_last_step_only", False)
        seq_mask = type_seqs.ne(self.pad_token_id)
        _, right_states = self._compute_event_states(
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            seq_mask=seq_mask,
        )

        if compute_last_step_only:
            right_states = right_states[:, -1:, :]
            sample_dtimes = sample_dtimes[:, -1:, :]

        solver_steps = max(sample_dtimes.size(-1) - 1, 1)
        sample_states = self._solve_at_sample_times(
            eta_initial=right_states,
            sample_dtimes=sample_dtimes,
            num_steps=solver_steps,
        )
        return torch.exp(self._clamp_eta(sample_states))

    def predict_one_step_at_every_event(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the next event after each observed prefix.

        The standard EasyTPP implementation uses thinning, which assumes a
        deterministic intensity function while estimating an upper bound and
        then accepting samples.  NJDTPP intensities are stochastic paths, so we
        follow the released NJDTPP evaluation style instead: integrate the
        next-event density on one Euler-Maruyama path and predict the mark from
        the left-limit state at the actual next event time.
        """
        _, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        left_states, right_states = self._compute_event_states(
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            seq_mask=batch_non_pad_mask,
        )
        prefix_right_states = right_states[:, :-1, :]
        next_left_states = left_states[:, 1:, :]

        horizon_scale = getattr(self.event_sampler, "dtime_max", 5.0)
        prediction_horizon = torch.max(
            time_delta_seqs[:, :-1] * horizon_scale,
            time_delta_seqs[:, :-1] + horizon_scale,
        )
        prediction_steps = max(self.num_prediction_samples - 1, 1)
        sample_dtimes = self._make_fixed_grid_dtimes(
            prediction_horizon,
            prediction_steps,
        )
        sample_states = self._solve_fixed_grid_path(
            eta_initial=prefix_right_states,
            delta_time=prediction_horizon,
            num_steps=prediction_steps,
        )
        total_intensities = torch.exp(self._clamp_eta(sample_states)).sum(dim=-1)
        sample_widths = sample_dtimes[..., 1:] - sample_dtimes[..., :-1]

        integral_increments = (
            0.5
            * (total_intensities[..., 1:] + total_intensities[..., :-1])
            * sample_widths
        )
        cumulative_integral = torch.cat(
            [
                total_intensities.new_zeros((*total_intensities.shape[:-1], 1)),
                integral_increments.cumsum(dim=-1),
            ],
            dim=-1,
        )
        density = total_intensities * torch.exp(-cumulative_integral)
        weighted_dtimes = sample_dtimes * density
        dtimes_pred = (
            0.5
            * (weighted_dtimes[..., 1:] + weighted_dtimes[..., :-1])
            * sample_widths
        ).sum(dim=-1)

        types_pred = torch.argmax(next_left_states, dim=-1)
        return dtimes_pred, types_pred
