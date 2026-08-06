"""
FlowModel1dSamplingMixin: transport operators (to_prior / to_target,
including whole-integration CUDA graphs), sample(), constrained
sampling/optimization, log_prob, freeze/unfreeze, interpolate.
"""
from typing import Callable, Optional
import torch
import torch.nn as nn
from kemsekov_torch.log_prop_approx import log_prob_inverse
from kemsekov_torch.fm.samplers import sample_base
from kemsekov_torch.fm.cuda_graph import _CudaGraph


class FlowModel1dSamplingMixin:
    def to_prior(
        self,
        data: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps=None,
        return_intermediates=False
    ):
        """
        Transport samples from target space back into latent prior space.

        This is the inverse transport operator of the learned flow and is used
        internally for density estimation, interpolation, constrained
        optimization and latent-space editing.

        Parameters
        ----------
        data : torch.Tensor
            Samples from the target distribution.

            Expected shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors associated with the samples.

            Expected shape:

            ``[batch_size, conditional_dim]``

        steps : int | None, optional
            Number of integration steps.

            If None, ``self.default_steps`` is used.

        return_intermediates : bool, default=False
            Whether to return intermediate integration states.

        Returns
        -------
        torch.Tensor
            Corresponding latent vectors in prior space.

        tuple[torch.Tensor, list[torch.Tensor]]
            Returned when ``return_intermediates=True``.

            Contains:

            - Final latent vectors
            - Intermediate transport states

        Notes
        -----
        The returned latent vectors approximately follow a standard Gaussian
        distribution when the model is trained successfully.
        """

        if not steps: steps = self.default_steps
        input_device = data.device
        if self._can_graph() and not return_intermediates:
            return self._graph_integrate(data, condition, steps, input_device, inverse=True)
        model_inference = lambda xt,t: self(xt,t,condition)
        data,condition=self._prepare_data(data,condition)
        out = self.fm.integrate(model_inference,data,steps,inverse=True,return_intermediates=return_intermediates)
        if return_intermediates:
            out,inter = out
            out = out.to(input_device)
            inter = [l.to(input_device) for l in inter]
            out = (out,inter)
        else:
            out = out.to(input_device)
        return out
    def to_target(
        self,
        normal_noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        steps=None,
        return_intermediates=False
    ):
        """
        Transport samples from latent prior space into target data space.

        This is the forward transport operator of the learned flow. Samples
        from the Gaussian prior are iteratively transformed into samples from
        the target distribution using the learned velocity field.

        For conditional models, generation is conditioned on the supplied
        condition vectors.

        Parameters
        ----------
        normal_noise : torch.Tensor
            Samples from latent prior space.

            Expected shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

            Expected shape:

            ``[batch_size, conditional_dim]``

            If None and the model is conditional, zero-conditioning is used.

        steps : int | None, optional
            Number of integration steps.

            If None, ``self.default_steps`` is used.

        return_intermediates : bool, default=False
            Whether to return intermediate integration states.

        Returns
        -------
        torch.Tensor
            Generated samples in target space.

        tuple[torch.Tensor, list[torch.Tensor]]
            Returned when ``return_intermediates=True``.

            Contains:

            - Final generated samples
            - Intermediate transport states

        Notes
        -----
        After ReFlow distillation this method may become a one-step or two-step
        generator depending on the value of ``self.default_steps``.
        """

        if not steps: steps = self.default_steps
        input_device = normal_noise.device
        if self._can_graph() and not return_intermediates:
            return self._graph_integrate(normal_noise, condition, steps, input_device, inverse=False)
        model_inference = lambda xt,t: self(xt,t,condition)
        normal_noise,condition=self._prepare_data(normal_noise,condition)
        out = self.fm.integrate(model_inference,normal_noise,steps,return_intermediates=return_intermediates)
        if return_intermediates:
            out,inter = out
            out = out.to(input_device)
            inter = [l.to(input_device) for l in inter]
            out = (out,inter)
        else:
            out = out.to(input_device)
        return out
    def _eager_to_target(self, normal_noise: torch.Tensor, condition=None, steps=None):
        """
        Forward transport without CUDA-graph acceleration (used inside
        captured training steps, where nested graph capture is illegal).
        """
        if not steps: steps = self.default_steps
        model_inference = lambda xt,t: self._eager_forward(xt,t,condition)
        normal_noise,condition=self._prepare_data(normal_noise,condition)
        return self.fm.integrate(model_inference,normal_noise,steps)
    def _eager_to_prior(self, data: torch.Tensor, condition=None, steps=None):
        """
        Inverse transport without CUDA-graph acceleration (used inside
        captured training steps, where nested graph capture is illegal).
        """
        if not steps: steps = self.default_steps
        model_inference = lambda xt,t: self._eager_forward(xt,t,condition)
        data,condition=self._prepare_data(data,condition)
        return self.fm.integrate(model_inference,data,steps,inverse=True)
    def _graph_integrate(self, x: torch.Tensor, condition, steps, input_device, inverse=False):
        """
        Runs the whole integration (momentum_heun / one-step / rk2 / rk3,
        i.e. the complete to_target/to_prior transport) as a single captured
        CUDA graph, keyed by (input shape, steps, direction).
        """
        x, condition = self._prepare_data(x, condition)
        key = (tuple(x.shape), steps, inverse)
        cg = self._integ_graphs.get(key)
        if cg is None:
            xb = torch.empty_like(x)
            cb = torch.empty_like(condition)
            def fn(xb, cb):
                model_inference = lambda xt, t: self._eager_forward(xt, t, cb)
                return (self.fm.integrate(model_inference, xb, steps, inverse=inverse),)
            cg = _CudaGraph.capture(fn, [xb, cb])
            if cg is None:
                self.cuda_graphs = False
                model_inference = lambda xt,t: self(xt,t,condition)
                out = self.fm.integrate(model_inference, x, steps, inverse=inverse)
                return out.to(input_device)
            self._integ_graphs[key] = cg
        cg.inputs[0].copy_(x)
        cg.inputs[1].copy_(condition)
        cg.replay()
        return cg.outputs[0].clone().to(input_device)

    def sample(
        self,
        num_samples : int,
        condition: Optional[torch.Tensor] = None,
        steps : Optional[int]=None,
        sobol : bool=False
    ):
        """
        Generate samples from the learned distribution.

        Samples are generated by drawing latent vectors from a Gaussian prior
        and transporting them into target space using ``to_target()``.

        Supports both unconditional and conditional generation.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        condition : torch.Tensor | None, optional
            Conditioning vectors.

            Expected shape:

            ``[num_samples, conditional_dim]``

            If None and the model is conditional, zero-conditioning is used.

        steps : int | None, optional
            Number of transport steps.

            If None, ``self.default_steps`` is used.

        sobol : bool, default=False
            Use Sobol low-discrepancy sampling instead of standard Gaussian
            sampling.

            Sobol sampling often improves latent-space coverage and may reduce
            variance for small sample counts.

        Returns
        -------
        torch.Tensor
            Generated samples.

            Shape:

            ``[num_samples, in_dim]``
        """

        if not steps: steps = self.default_steps
        if sobol:
            x = sample_base(self.sobol,num_samples,self.device)
        else:
            x = torch.randn((num_samples,self.in_dim),device=self.device)
        return self.to_target(x,condition,steps=steps)
    def constrained_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        condition : Optional[torch.Tensor] = None,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 0.0,
        sampler_steps = None
    ) -> torch.Tensor:
        """
        Generate samples satisfying arbitrary differentiable constraints.

        The optimization is performed in latent space rather than data space.
        Latent vectors are adjusted using LBFGS while balancing:

        - Constraint satisfaction
        - Prior probability preservation
        - Optional mode-seeking behavior

        This method is particularly useful for inverse design problems where
        generated samples must satisfy user-defined objectives.

        Parameters
        ----------
        constraint : Callable[[torch.Tensor], torch.Tensor]
            Differentiable constraint function.

            Receives generated samples:

            ``[batch_size, in_dim]``

            Returns a scalar loss.

        num_samples : int
            Number of samples to generate.

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        noise_scale : float, default=0.0
            Langevin-style noise added after optimization steps.

            Small values can improve exploration.

            Typical range:

            ``0.0 - 0.05``

        steps : int, default=2
            Number of latent optimization iterations.

        lr : float, default=1
            LBFGS learning rate.

        mode_closeness_weight : float, default=0.0
            Additional penalty encouraging solutions closer to latent-space
            modes.

            Large values may cause mode collapse.

        sampler_steps : int | None, optional
            Number of flow integration steps used during optimization.

            If None, ``self.default_steps`` is used.

        Returns
        -------
        torch.Tensor
            Generated samples satisfying the constraint as closely as possible.
        """
        model = self
        model.eval()
        self.freeze()
        device = self.device
        # Initialize z from standard normal distribution
        z = sample_base(self.sobol,num_samples,device=device).requires_grad_(True)
        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        self._iteration = Iteration()
        
        def closure():
            optimizer.zero_grad()

            # Forward pass: x = M_inv(z)
            x = model.to_target(z,condition,steps=sampler_steps)

            # Balance original prior probability/vs likelihood maximization
            L_prior = (z * z).mean()
            L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            it = self._iteration
            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            L_total.backward()
            with torch.no_grad():
                z.data += noise_scale * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(self._iteration.best_sample,condition,steps=sampler_steps)
        self.unfreeze()
        return final_x
    def constrained_optimize(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        data,
        condition : Optional[torch.Tensor] = None,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 0.0,
        sampler_steps = None
    ) -> torch.Tensor:
        """
        Optimize existing samples subject to a differentiable constraint.

        Unlike ``constrained_sample()``, this method starts from existing data,
        maps it into latent space, performs optimization there, and then maps
        the optimized latent vectors back into target space.

        The optimization attempts to preserve original sample probability while
        minimizing the supplied constraint.

        Parameters
        ----------
        constraint : Callable[[torch.Tensor], torch.Tensor]
            Differentiable objective function.

        data : torch.Tensor
            Initial samples.

            Shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        noise_scale : float, default=0.0
            Langevin-style exploration noise.

        steps : int, default=2
            Number of optimization iterations.

        lr : float, default=1
            LBFGS learning rate.

        mode_closeness_weight : float, default=0.0
            Additional mode-seeking regularization.

        sampler_steps : int | None, optional
            Number of transport steps used during optimization.

        Returns
        -------
        torch.Tensor
            Optimized samples in target space.
        """
        model = self
        model.eval()
        self.freeze()
        device = self.device
        # Move data to prior
        with torch.no_grad():
            z : torch.Tensor = self.to_prior(data,condition,steps=sampler_steps)
        z=z.requires_grad_(True)
        
        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        self._iteration = Iteration()
        
        def closure():
            optimizer.zero_grad()

            # Forward pass: x = M_inv(z)
            x = model.to_target(z,condition,steps=sampler_steps)

            # Balance original prior probability/vs likelihood maximization
            L_prior = (z * z).mean()
            L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            it = self._iteration
            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            L_total.backward()
            with torch.no_grad():
                z.data += noise_scale * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.to_target(self._iteration.best_sample,condition,steps=sampler_steps)
        self.unfreeze()
        return final_x
    def optimize(
        self, 
        data: torch.Tensor,
        condition : Optional[torch.Tensor] = None,
        lr: float = 1.0, 
        epochs: int = 1,
        columns_to_optimize: list[int] = None,
        random_directions=0
    ):
        """
        Optimize specific columns of data to maximize log probability.

        This method performs gradient-based optimization to adjust specific columns of the input
        data to increase their likelihood under the learned model distribution. It keeps other
        columns fixed while optimizing the specified ones.

        Args:
            data (torch.Tensor): Input tensor of shape [batch_size, input_dim] to optimize
            lr (float): Learning rate for the LBFGS optimizer (default: 1.0)
            epochs (int): Number of optimization epochs (default: 1)
            columns_to_optimize (list[int]): List of column indices to optimize (0-based).
                                           If None or empty, all columns will be optimized.
            random_directions: log-prob random directions approximation vectors
        Returns:
            tuple: A tuple containing:
                 - torch.Tensor: Optimized data tensor with the same shape as input
                 - torch.Tensor: Final loss value after optimization
        """
        batch_size, input_dim = data.shape
        self.freeze()
        # Handle default case - optimize all columns if none specified
        if columns_to_optimize is None or len(columns_to_optimize) == 0:
            columns_to_optimize = list(range(input_dim))

        # Validate column indices
        columns_to_optimize = [c for c in columns_to_optimize if 0 <= c < input_dim]
        if not columns_to_optimize:
            return data.clone(), -self.log_prob(data,condition=condition,random_directions=random_directions).sum().detach()

        # Identify fixed columns as those not in columns_to_optimize
        all_columns = list(range(input_dim))
        fixed_columns = [c for c in all_columns if c not in columns_to_optimize]

        # Create optimizable parameters for only the specified columns
        optimizable_data = data[:, columns_to_optimize].clone().detach().requires_grad_(True)

        # Fixed data doesn't need gradients
        if fixed_columns:
            fixed_data = data[:, fixed_columns].clone().detach()
        else:
            fixed_data = None

        # Define optimizer on only the optimizable part
        optimizer = torch.optim.LBFGS([optimizable_data], lr=lr, max_iter=20)

        class IterationData:
            best_loss = 1e8
            best_optimizable_data = optimizable_data.clone().detach()

        iteration = IterationData()
        self._iteration = iteration
        # Reconstruct full tensor by combining optimizable and fixed parts
        self._current_data = torch.zeros_like(data)

        def closure():
            optimizer.zero_grad(True)

            iteration = self._iteration
            current_data = self._current_data.detach()

            # Fill in the optimizable columns
            current_data[:, columns_to_optimize] = optimizable_data

            # Fill in fixed columns if any exist
            if fixed_columns:
                current_data[:, fixed_columns] = fixed_data

            # Compute loss on the full tensor
            loss = -self.log_prob(current_data,condition=condition,random_directions=random_directions).sum()

            if loss<iteration.best_loss:
                iteration.best_loss=loss
                iteration.best_optimizable_data=optimizable_data.detach().clone()

            loss.backward()

            return loss

        # Run optimization
        for i in range(epochs):
            loss = optimizer.step(closure)

        # Create final result by combining optimized and fixed parts
        result = torch.zeros_like(data)
        result[:, columns_to_optimize] = iteration.best_optimizable_data

        # Add back fixed columns if any exist
        if fixed_columns:
            result[:, fixed_columns] = fixed_data
        self.unfreeze()
        return result, iteration.best_loss
    def log_prob(
        self, 
        data, 
        condition:Optional[torch.Tensor] = None,
        eps=1e-3,
        random_directions=0,
        return_prior=False,
        steps=None):
        """
        Estimate log-probability under the learned distribution.

        Density estimation is performed by transporting samples into latent
        space and estimating the inverse-flow Jacobian determinant.

        Supports conditional densities when condition vectors are supplied.

        Parameters
        ----------
        data : torch.Tensor
            Samples for density evaluation.

            Shape:

            ``[batch_size, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors.

        eps : float, default=1e-3
            Finite-difference step size used for Jacobian estimation.

        random_directions : int, default=0
            Number of random projection directions used for Jacobian
            approximation.

            Values:

            - 0 : exact directional evaluation
            - >0 : stochastic approximation

            Larger values are typically faster for high-dimensional problems.

        return_prior : bool, default=False
            Forwarded to ``log_prob_inverse``.

        Returns
        -------
        torch.Tensor
            Estimated log-probabilities.

        Notes
        -----
        For conditional models the returned density corresponds to:

        .. math::

            \\log p(x \\mid c)

        rather than the unconditional density.
        """
        
        to_prior = lambda xt:self.to_prior(xt,condition,steps=steps)
        return log_prob_inverse(to_prior,data.to(self.device),eps,random_directions=random_directions,return_prior=return_prior)
    
    def freeze(self):
        """
        Disables grad on model weights
        """
        for p in self.parameters():
            p.requires_grad_(False)
    def unfreeze(self):
        """
        Enables grad on model weights
        """
        for p in self.parameters():
            p.requires_grad_(True)
    def interpolate(
        self,
        A:torch.Tensor,
        B:torch.Tensor,
        t:torch.Tensor|float,
        A_condition : Optional[torch.Tensor] = None,
        B_condition : Optional[torch.Tensor] = None):
        """
        Interpolate between samples through the learned latent space.

        Samples are first mapped into latent space, interpolated linearly,
        and then mapped back into target space.

        Compared to direct interpolation in data space, latent interpolation
        often produces smoother and more realistic trajectories.

        Conditional interpolation is also supported.

        When both condition vectors are supplied, conditions are interpolated
        together with latent representations.

        Parameters
        ----------
        A : torch.Tensor
            Starting samples.

            Shape:

            ``[batch_size, in_dim]``

        B : torch.Tensor
            Ending samples.

            Shape:

            ``[batch_size, in_dim]``

        t : float | torch.Tensor
            Interpolation locations.

            Values must lie in:

            ``[0, 1]``

            Examples:

            .. code-block:: python
                t = 0.5
                t = torch.linspace(0, 1, 128)

        A_condition : torch.Tensor | None, optional
            Conditions associated with A.

        B_condition : torch.Tensor | None, optional
            Conditions associated with B.

        Returns
        -------
        torch.Tensor
            Interpolated samples.

        Notes
        -----
        The interpolation is performed as:

        1. A -> latent space
        2. B -> latent space
        3. Linear interpolation in latent space
        4. Transport back into target space

        If both conditions are supplied:

        .. math::

            c_t = (1-t)c_A + tc_B

        is used during decoding.
        """
        
        if isinstance(t,float):t=torch.tensor([t])

        if A.ndim==1:A=A.unsqueeze(0)
        if B.ndim==1:B=B.unsqueeze(0)
        if t.ndim==1:t=t.unsqueeze(1).unsqueeze(1)
        if A_condition is not None:
            if A_condition.ndim==1:A_condition=A_condition.unsqueeze(0)
        
        if B_condition is not None:
            if B_condition.ndim==1:B_condition=B_condition.unsqueeze(0)

        with torch.no_grad():
            A_prior = self.to_prior(A,A_condition)
            B_prior = self.to_prior(B,B_condition)

            prior_interp = torch.lerp(A_prior,B_prior,t)
            if A_condition is not None and B_condition is not None:
                condition_interp = torch.lerp(A_condition,B_condition,t)
            else:
                condition_interp=None
            AB_interp = self.to_target(prior_interp,condition_interp)
        
        return AB_interp