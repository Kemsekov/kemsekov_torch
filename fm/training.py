"""
FlowModel1dTrainingMixin: fit / reflow training loops and the captured
CUDA-graph training steps (_get_fit_graph, _get_reflow_graph).
Optimizations applied to fit(): defer .item() to end of epoch,
cached sobol prior + rotation instead of randperm, r2 computed once
per epoch outside the captured graph.
"""
import gc
from copy import deepcopy
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from kemsekov_torch.common_modules import Prod, Residual
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.fm.core import (LossNormalizer1d, get_fm_optim_groups,
                                    zero_module)
from kemsekov_torch.fm.samplers import sample_base
from kemsekov_torch.fm.cuda_graph import _CudaGraph


class FlowModel1dTrainingMixin:
    def fit(
        self,
        data: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_dropout: float = 0.5,
        epochs: int = 512,
        batch_size: int = 512,
        contrastive_loss_weight=1.0,
        lr: float = 0.02,
        distribution_matching=0.0,
        debug: bool = False,
        scheduler = True,
    ):
        """
        Train the Flow Matching model.

        Training uses Contrastive Flow Matching (CFM) together with optional
        distribution matching and classifier-free conditioning.

        During training, random interpolation points are sampled between
        Gaussian prior samples and target data samples. The model learns to
        predict the transport direction connecting both domains.

        When the model is conditional, condition vectors may be randomly
        replaced with zeros according to ``condition_dropout``. This implements
        classifier-free conditioning and improves generalization.
        
        **Model is fitted enough when r2 metric is above 0.36 or so.**

        Parameters
        ----------
        data : torch.Tensor | numpy.ndarray
            Training dataset.

            Expected shape:

            ``[num_samples, in_dim]``

        condition : torch.Tensor | numpy.ndarray | None, optional
            Conditioning vectors.

            Required when training a conditional model.

            Expected shape:

            ``[num_samples, conditional_dim]``

            If None and the model is unconditional, no conditioning is used.

        condition_dropout : float, default=0.5
            Probability of replacing condition vectors with zeros during
            training.

            This implements classifier-free conditioning.

            Values between 0.1 and 0.5 typically work well.

        epochs : int, default=64
            Number of training epochs.

        batch_size : int, default=256
            Mini-batch size.

        contrastive_loss_weight : float, default=1.0
            Weight applied to the contrastive flow matching objective.

            Larger values increase separation from negative transport
            directions.

        lr : float, default=0.02
            Learning rate.

        distribution_matching : float, default=0.0
            Enables adaptive loss reweighting that encourages better matching
            of the target distribution.

            Recommended values:

            - 0.0: disabled
            - 0.05 - 0.25: mild correction
            - >0.5: aggressive distribution matching

        debug : bool, default=False
            Print training statistics.

        scheduler : bool, default=True
            Enable cosine annealing learning-rate scheduling.

        Returns
        -------
        None

        Notes
        -----
        Training history is stored in:

        ``self.fit_history["loss"]``

        ``self.fit_history["r2"]``

        The best model checkpoint is automatically restored at the end of
        training according to validation R² measured on transport direction
        prediction.
        """
        self.unfreeze()
        # these are optimal for cpu-training
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        except: pass
        gc.disable()
        
        data, condition = self._prepare_data(data, condition)
        
        model = self
        device = model.device
        batch_size = min(batch_size,data.shape[0])
        
        trainable_weights = list(model.parameters())
        loss_normalizer=None
        if distribution_matching>0:
            loss_normalizer = LossNormalizer1d(model.in_dim,model.hidden_dim).to(device)
            # loss_normalizer = torch.jit.trace(loss_normalizer,(torch.randn((1,self.in_dim),device=device),torch.randn((1,1),device=device)))
            trainable_weights=trainable_weights+list(loss_normalizer.parameters())
        

        assert data.shape[-1]==self.in_dim, f'Dataset dimension must match in_dim on model. data.shape[-1]({data.shape[-1]})!=model.in_dim({self.in_dim})'
        model.train()
        
        # fused AdamW requires float32 parameters
        optim = torch.optim.AdamW(get_fm_optim_groups(model,loss_normalizer), lr=lr,fused=self._param_dtype()==torch.float32)
        
        best_loss = float("inf")
        best_r2 = -1e8
        
        best_trained_model = deepcopy(model)
        
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs)
        
        prior_batch=torch.randn((batch_size,self.in_dim),device=device)
        time = torch.rand(batch_size,device=device)
        
        perm = torch.zeros(n, device=device,dtype=torch.int32)
        rot_idx = torch.arange(batch_size, device=device)
        # model_trace = torch.jit.trace(model,example_inputs=(torch.randn((1,self.in_dim),device=device),torch.randn((1),device=device)))
        model_trace=model
        self.fit_history = {
            'loss':[],
            'r2':[]
        }
        
        # capture the per-batch training step (forward + loss + backward) as
        # CUDA graphs, one per distinct batch size
        use_train_graphs = self._graphs_available() and torch.is_grad_enabled()
        epoch_t = torch.zeros(1, device=device)
        last_pred = None
        last_target = None
        
        try:
            for epoch in range(epochs):
                if use_train_graphs:
                    epoch_t.fill_(epoch)
                if debug and improved:
                    print(f"Epoch {epoch}: best_loss={best_loss:0.3f}\tbest r2={best_r2:0.3f}")
                improved = False
                
                # shuffle each epoch
                torch.randperm(n, device=device,out=perm)
                data_shuf = data[perm]
                condition_shuf = condition[perm]

                losses = 0
                r2s = 0
                for start in slices:
                    if use_train_graphs:
                        # CUDA-graph training step: draw all randomness eagerly
                        # (in the same order as the eager path), copy inputs
                        # into static buffers, replay the captured forward+loss
                        # +backward graph, then step the optimizer
                        B = min(batch_size, n - start)
                        zero_mask = (torch.rand(batch_size,device=device)<condition_dropout)[:B].unsqueeze(-1)
                        prior_batch = sample_base(self.sobol,batch_size,device)
                        time.uniform_()
                        rot_idx.add_(1).remainder_(B)
                        idx = rot_idx[:B]
                        tg = self._get_fit_graph(B, model, condition_dropout, distribution_matching, epochs, device, loss_normalizer, contrastive_loss_weight)
                        if tg is not None:
                            tg.inputs[0].copy_(data_shuf[start : start + B])
                            tg.inputs[1].copy_(condition_shuf[start : start + B])
                            tg.inputs[2].copy_(zero_mask)
                            tg.inputs[3].copy_(prior_batch[:B])
                            tg.inputs[4].copy_(time[:B])
                            tg.inputs[5].copy_(epoch_t)
                            tg.inputs[6].copy_(idx)
                            tg.replay()
                            # expose the shared gradient buffers as p.grad
                            # (matches eager: unused params keep None grads)
                            for p, buf in tg.grad_buffers.items():
                                p.grad = buf
                            for p in tg.unused_params:
                                p.grad = None
                            loss = tg.outputs[0].detach()
                            last_pred = tg.outputs[1]
                            last_target = tg.outputs[2]
                            optim.step()
                            losses+=loss
                            self.fit_history['loss'].append(loss.detach().clone())
                            continue
                        use_train_graphs = False
                    optim.zero_grad(set_to_none=True)  # set_to_none saves mem and can be faster [web:399]
                    
                    batch = data_shuf[start : start + batch_size]
                    B = batch.shape[0]
                    
                    zero_mask = (torch.rand(batch_size,device=device)<condition_dropout)[:B].unsqueeze(-1)
                    condition_batch = condition_shuf[start : start + batch_size]*zero_mask
                    
                    model_inference = lambda xt,t: model_trace(xt,t,condition_batch)
                    
                    # prior_batch.normal_()
                    prior_batch = sample_base(self.sobol,batch_size,device)
                    time.uniform_()
                    
                    # if epoch/epochs<0.5:
                    # prior_batch=match_approximate_sliced(prior_batch,batch)
                    
                    pred_dir,target_dir,contrast_dir,t = \
                        model.fm.contrastive_flow_matching_pair(
                            model_inference,
                            prior_batch[:B],
                            batch,
                            time=time[:B]
                        )
                    # model predicts in the model dtype; losses are computed in float32
                    pred_dir = pred_dir.float()
                    
                    pred_loss = F.mse_loss(pred_dir,target_dir,reduction='none')+1
                    contrastive_loss = F.mse_loss(pred_dir,contrast_dir,reduction='none')
                    
                    contrastive_loss_det = contrastive_loss.detach()
                    pred_loss_det = pred_loss.detach()
                    # make it negative
                    contrastive_loss-=contrastive_loss_det.max()+1e-4
                    contrastive_loss=contrastive_loss/contrastive_loss_det.abs().mean()*pred_loss_det.abs().mean()
                    
                    # scale it
                    contrastive_loss = contrastive_loss_weight*contrastive_loss
                    
                    # sample-wise loss
                    sample_loss = pred_loss-contrastive_loss
                    
                    dm = (1-(1+epoch)/epochs)*distribution_matching
                    # dm=distribution_matching
                    if distribution_matching>0:
                        with torch.no_grad():  # Stop-gradient via detach
                            sg_log_losses = pred_loss_det.log()
                            target_log_w = -sg_log_losses  # log(1/L)
                        # dm = distribution_matching
                        weights = loss_normalizer(target_dir, t) # it equals to log(1/loss)
                        loss_weighted = (weights.detach()*dm).exp() # it equals to 1/loss
                        aux_loss = F.mse_loss(weights, target_log_w)
                    else:
                        loss_weighted=1
                        aux_loss=0
                    
                    #scale loss by it's prediction
                    weighed_loss = (loss_weighted*sample_loss).mean()
                    # print(r2_score(weights, (1/sample_loss).log()))
                    # print(weighed_loss)
                    loss = weighed_loss+dm*aux_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1,
                        norm_type=2.0,
                    )
                        
                    optim.step()
                    
                    loss=loss.detach()
                    
                    r2 = r2_score(pred_dir,target_dir)
                    losses+=loss
                    r2s+=r2
                    
                    self.fit_history['loss'].append(loss.item())
                    self.fit_history['r2'].append(r2.item())
                    
                if scheduler: sch.step()
                
                if last_pred is not None:
                    r2s = r2_score(last_pred, last_target)
                    mean_r2 = r2s.item()
                    self.fit_history['r2'].append(r2s.item())
                else:
                    mean_r2 = (r2s/len(slices)).item()
                mean_loss = (losses/len(slices)).item()
                if mean_r2 > best_r2:
                    best_loss = mean_loss
                    model_state_dict = model.state_dict()
                    best_trained_model = {key:model_state_dict[key].clone() for key in model_state_dict}
                    best_r2 = mean_r2
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        finally:
            gc.enable()
            gc.collect()
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}\tbest_r2={best_r2:0.3f}")
        
        # update current model with best checkpoint
        model.load_state_dict(best_trained_model)
        model.eval()
        self.fit_history['loss'] = [x.item() if isinstance(x, torch.Tensor) else x for x in self.fit_history['loss']]
        self.fit_history['r2'] = [x.item() if isinstance(x, torch.Tensor) else x for x in self.fit_history['r2']]
    def _prepare_data(self, data, condition):
        if not isinstance(data,torch.Tensor):
            data = torch.tensor(data,dtype=torch.float32,device=self.device)
        else:
            data = data.to(self.device).float()
        if data.ndim==1:
            data = data.unsqueeze(0)
        assert data.shape[-1]==self.in_dim,f"Data input shape must equal (BATCH,{self.in_dim}), got {data.shape}"
        
        if self.conditional_dim is not None and condition is not None:
            if not isinstance(condition,torch.Tensor):
                condition = torch.tensor(condition,dtype=torch.float32,device=self.device)
            else:
                condition = condition.to(self.device).float()
            if condition.ndim==1:
                condition=condition.unsqueeze(0)
            if condition.shape[0]==1:
                condition=condition[[0]*len(data)]
            assert len(condition)==len(data),'Dataset length and condition length must match'
            assert condition.shape[-1]==self.conditional_dim, f'Condition dimension must match conditional_dim on model. condition.shape[-1]({data.shape[-1]})!=model.conditional_dim({self.conditional_dim})'
        if condition is None:
            condition = torch.zeros((len(data),1),device=self.device)
        return data,condition
    def reflow(
            self,
            data : torch.Tensor,
            condition : Optional[torch.Tensor] = None,
            epochs = 2048,
            steps : Literal[1,2] = 1,
            batch_size=512,
            debug = False,
            lr = 0.01,
            weight_decay=0.01,
            distribution_matching = 0,
            grad_clip_max_norm : float|None=1,
            base_model : nn.Module|None = None,
            freeze_integrator = False
        ) -> None:
        """
        Distill a multi-step flow into a fast one-step or two-step generator.

        ReFlow trains the current model to directly approximate the transport
        map learned by a slower flow model. The resulting model can generate
        samples using only one or two transport evaluations while preserving
        the distribution learned by the teacher model.

        The procedure operates in both directions:

        - Prior -> Target
        - Target -> Prior

        allowing the distilled model to retain generation, inversion and
        density-estimation capabilities.

        During training a synthetic dataset is created by transporting samples
        through a teacher model. The distilled model is then trained to
        reproduce those mappings using a reduced number of integration steps.

        For conditional models the same conditioning vectors used by the
        teacher are propagated through the distillation process.

        Parameters
        ----------
        data : torch.Tensor
            Dataset sampled from the target distribution.

            Shape:

            ``[num_samples, in_dim]``

        condition : torch.Tensor | None, optional
            Conditioning vectors associated with the dataset.

            Required for conditional models.

            Shape:

            ``[num_samples, conditional_dim]``

        epochs : int, default=512
            Number of ReFlow optimization iterations.

        steps : {1, 2}, default=1
            Target generator complexity.

            - ``1``: distill into a one-step generator
            - ``2``: distill into a two-step generator

            After successful training:

            ``self.default_steps = steps``

        batch_size : int, default=256
            Mini-batch size.

        debug : bool, default=False
            Print optimization statistics.

        lr : float, default=1e-2
            Learning rate.

        weight_decay : float, default=0.01
            Weight decay used by the optimizer.

        distribution_matching : float, default=0
            Enables adaptive loss reweighting.

            Larger values focus training on regions where the distilled model
            performs poorly.

            Typical values:

            - 0.0 : disabled
            - 0.05 - 0.25 : mild correction

        grad_clip_max_norm : float | None, default=1
            Maximum gradient norm.

            If None, gradient clipping is disabled.

        base_model : nn.Module | None, optional
            Teacher model.

            The teacher must implement:

            - ``to_target()``
            - ``to_prior()``

            If None, the current model is used as the teacher
            (self-distillation).

        freeze_integrator : bool, default=False
            Whether to freeze learned integration coefficients during ReFlow.

            If True:

            - one_weights
            - one_weights_inv
            - rk2_weights
            - rk2_weights_inv

            remain fixed.

            If False, the integrator coefficients are optimized jointly with
            the neural network.

        Returns
        -------
        None

        Notes
        -----
        ReFlow performs online dataset generation using the teacher model.

        The generated training set contains two components:

        1. Real dataset samples mapped into latent space.
        2. Synthetic samples generated by the teacher model from randomly
        sampled latent vectors.

        Mixing both sources improves coverage of latent space and reduces
        failures in poorly represented prior regions.

        Unlike standard Flow Matching training, ReFlow optimizes direct
        transport mappings:

        .. math::

            x \\rightarrow y

        and

        .. math::

            y \\rightarrow x

        rather than velocity-field prediction.

        Training statistics are stored in:

        .. code-block:: python

            self.reflow_history["loss"]
            self.reflow_history["forward_r2"]
            self.reflow_history["inverse_r2"]

        Examples
        --------
        Distill a trained flow into a one-step generator:

        >>> model.fit(data)
        >>> model.reflow(data, steps=1)

        Conditional ReFlow:

        >>> model.fit(data, condition)
        >>> model.reflow(data, condition, steps=1)

        Create a two-step distilled model:

        >>> model.reflow(data, steps=2)
        """

        self.unfreeze()
        gc.disable()
        try:
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        except: pass
        if not isinstance(data,torch.Tensor):
            data = torch.tensor(data,dtype=torch.float32,device=self.device)
        self.to(self.device)
        
        data = data.to(self.device)
        if base_model is None: base_model=self
        if self.conditional_dim is not None:
            assert condition is not None,'Cannot reflow conditional model with None condition'
        if condition is None:
            condition = torch.zeros((len(data),self.conditional_dim or 1))
        condition = condition.to(self.device)
        
        with torch.no_grad():
            x = base_model.to_prior(data,condition,steps=32)
            y = data
            
            # balance generated and original dataset 50/50
            # the thing is that dataset latent space may be too limited
            # to reach all edge-case samples from some subspaces of prior
            # and reflowed model may struggle to transport these subspace prior
            # samples to target distribution, so, we also include generated from base model
            # samples to reflow model training, this step empirically helps a lot
            # with reflowed model quality
            
            # x_gen = sample_base(self.sobol,len(x),self.device)
            x_gen = torch.randn_like(x)
            cond_gen = torch.zeros_like(condition)
            y_gen = base_model.to_target(x_gen,cond_gen,steps=32)
            
            x = torch.concat([x,x_gen],0)
            y = torch.concat([y,y_gen],0)
            cond = torch.concat([condition,cond_gen],0)
            
        assert steps in [1,2],"steps parameter must be one of [1,2]"
        self.fm.reset_weights()
        # reset_weights() replaces the integration-weight tensors, so any
        # captured graphs referencing the old ones must be dropped
        self._invalidate_graphs()
        self.train()
        self.default_steps=steps
        
        if freeze_integrator:
            self.fm.freeze()

        
        loss_normalizer = nn.Sequential(
            nn.Linear(self.in_dim*2,self.hidden_dim),
            Residual([
                nn.SiLU(),
                zero_module(nn.Linear(self.hidden_dim,self.hidden_dim)),
            ],init_at_zero=False),
            nn.RMSNorm(self.hidden_dim),
            Prod([
                nn.Linear(self.hidden_dim,self.hidden_dim),
                # nn.RMSNorm(self.hidden_dim),
                nn.Tanh(),
            ]),
            nn.SiLU(),
            nn.Linear(self.hidden_dim,2),
        ).to(self.device)
        
        device=self.device
        # loss_normalizer = torch.jit.trace(loss_normalizer,torch.randn((1,self.in_dim*2),device=self.device))
        # opt = torch.optim.AdamW(get_fm_optim_groups(self,loss_normalizer,weight_decay=weight_decay),lr=lr,fused=True)
        # fused AdamW requires float32 parameters
        opt = torch.optim.AdamW(list(self.parameters())+list(loss_normalizer.parameters()),weight_decay=weight_decay,lr=lr,fused=self._param_dtype()==torch.float32)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        mse = torch.nn.functional.mse_loss
        
        self.reflow_history = {
            'loss':[],
            'forward_r2':[],
            'inverse_r2':[],
        }
        # running_r2 = 0
        # best_r2 = 0
        # best_model = None
        # check_each=16
        # capture the per-iteration distillation step (forward + losses +
        # backward + clip) as a CUDA graph
        use_train_graphs = self._graphs_available() and torch.is_grad_enabled()
        # rolling window through a precomputed permutation: reshuffle once per
        # full pass (2*len(data)/batch_size iterations) instead of a randperm
        # on every iteration
        perm_roll = torch.randperm(x.shape[0], device=device)
        roll_i = 0
        try:
            for i in range(epochs):
                ind = perm_roll[roll_i:roll_i + batch_size]
                roll_i += batch_size
                if roll_i + batch_size > x.shape[0]:
                    roll_i = 0
                    perm_roll = torch.randperm(x.shape[0], device=device)
                if use_train_graphs:
                    rg = self._get_reflow_graph(batch_size, opt, loss_normalizer, grad_clip_max_norm, distribution_matching, device)
                    if rg is not None:
                        rg.inputs[0].copy_(x[ind])
                        rg.inputs[1].copy_(y[ind])
                        rg.inputs[2].copy_(cond[ind])
                        rg.replay()
                        loss, forward_r2, inverse_r2, prediction_loss, forward_weight, inverse_weight, forward_loss, inverse_loss = rg.outputs
                        # expose the shared gradient buffers as p.grad
                        for p, buf in rg.grad_buffers.items():
                            p.grad = buf
                        for p in rg.unused_params:
                            p.grad = None
                        opt.step()
                        sch.step()
                        self.reflow_history['loss'].append(loss)
                        self.reflow_history['forward_r2'].append(forward_r2)
                        self.reflow_history['inverse_r2'].append(inverse_r2)
                        if debug and (i+1)%32==0:
                            loss_pred_r2 = (r2_score(forward_weight,forward_loss.log())+r2_score(inverse_weight,inverse_loss.log()))/2
                            print(f"Iteration={(str(i)+" "*6)[:4]} loss={str(prediction_loss.detach().item())[:8]} forward_r2={str(forward_r2.item())[:6]} inverse_r2={str(inverse_r2.item())[:6]} loss_pred_r2={str(loss_pred_r2.item())[:6]}")
                        continue
                    use_train_graphs = False
                opt.zero_grad(True)
                xbatch = x[ind]
                ybatch = y[ind]
                condbatch = cond[ind]
                y_pred = self.to_target(xbatch,condbatch)
                x_pred = self.to_prior(ybatch,condbatch)
                
                forward_loss = mse(ybatch,y_pred,reduction='none').mean(-1)
                forward_loss-=forward_loss.min().detach()-1e-2
                inverse_loss = mse(xbatch,x_pred,reduction='none').mean(-1)
                inverse_loss-=inverse_loss.min().detach()-1e-2
                
                prediction_loss = forward_loss.mean()+inverse_loss.mean()
                
                forward_weight,inverse_weight = loss_normalizer(torch.concat([xbatch,ybatch],-1)).chunk(2,-1)
                forward_weight, inverse_weight = forward_weight[...,0], inverse_weight[...,0]
                normalizer_loss = mse(forward_weight,forward_loss.detach().log())+mse(inverse_weight,inverse_loss.detach().log())
                
                fw = (-forward_weight*distribution_matching).detach().exp()
                iw = (-inverse_weight*distribution_matching).detach().exp()
                loss = (fw*forward_loss).mean()+(iw*inverse_loss).mean()+normalizer_loss*distribution_matching
                
                forward_r2 = r2_score(ybatch,y_pred)
                inverse_r2 = r2_score(xbatch,x_pred)
                # running_r2 += (forward_r2+inverse_r2)/2
                # if (i+1)%check_each==0:
                #     if running_r2>best_r2:
                #         if debug: print("Save")
                #         best_r2=running_r2
                #         best_model=deepcopy(self.state_dict())
                #     running_r2=0
                    
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=grad_clip_max_norm,
                        norm_type=2.0,
                    )
                opt.step()
                sch.step()
                

                self.reflow_history['loss'].append(loss.detach())
                self.reflow_history['forward_r2'].append(forward_r2.detach())
                self.reflow_history['inverse_r2'].append(inverse_r2.detach())
                if debug and (i+1)%32==0:
                    loss_pred_r2 = (r2_score(forward_weight,forward_loss.log())+r2_score(inverse_weight,inverse_loss.log()))/2
                    print(f"Iteration={(str(i)+" "*6)[:4]} loss={str(prediction_loss.detach().item())[:8]} forward_r2={str(forward_r2.item())[:6]} inverse_r2={str(inverse_r2.item())[:6]} loss_pred_r2={str(loss_pred_r2.item())[:6]}")
        except KeyboardInterrupt as e:
            print("Stop reflowing...")
        finally:
            gc.enable()
            gc.collect()
            
        # self.load_state_dict(best_model)
        for _k in self.reflow_history:
            self.reflow_history[_k] = [v.item() if isinstance(v, torch.Tensor) else v for v in self.reflow_history[_k]]
        self.eval()
    

    def _get_fit_graph(self, B, model, condition_dropout, distribution_matching, epochs, device, loss_normalizer, contrastive_loss_weight):
        """
        Returns (capturing on first call) a CUDA graph of one fit() training
        step for batch size B: forward + loss + gradients + clip. Gradients
        are computed with torch.autograd.grad and copied into shared eager
        buffers (see the docstring on _CudaGraph for why backward() cannot be
        captured), then exposed as p.grad by the caller.

        The optimizer step and all RNG draws stay eager: the learning rate is
        scheduled by the caller and the RNG must not be frozen by the graph.
        Returns None (and disables graphs) if the capture fails.
        """
        key = ('fit', B)
        tg = self._train_graphs.get(key)
        if tg is not None:
            return tg
        # shared gradient buffers: every captured graph copies its freshly
        # computed gradients into these eager, stable addresses (loss.backward
        # would accumulate into per-capture .grad allocations that the
        # autograd engine reallocates on each new capture, invalidating the
        # addresses captured by earlier graphs)
        params = list(self.parameters())
        used_ids = set()
        bx = torch.empty(B, self.in_dim, device=device)
        bc = torch.empty(B, self.conditional_dim or 1, device=device)
        bm = torch.empty(B, 1, device=device)
        bp = torch.empty(B, self.in_dim, device=device)
        bt = torch.empty(B, device=device)
        be = torch.zeros(1, device=device)
        bi = torch.arange(B, device=device)
        def fn(bx, bc, bm, bp, bt, be, bi):
            condition_batch = bc * bm
            model_inference = lambda xt, t: self._eager_forward(xt, t, condition_batch)
            pred_dir, target_dir, contrast_dir, t_exp = \
                model.fm.contrastive_flow_matching_pair(model_inference, bp, bx, time=bt, idx=bi)
            pred_dir = pred_dir.float()
            pred_loss = F.mse_loss(pred_dir, target_dir, reduction='none') + 1
            contrastive_loss = F.mse_loss(pred_dir, contrast_dir, reduction='none')
            contrastive_loss_det = contrastive_loss.detach()
            pred_loss_det = pred_loss.detach()
            # NOTE: the eager path does this with an in-place `-=` on
            # contrastive_loss, which also shifts the shared-storage detach
            # view. In-place ops cannot be captured (the replay's backward
            # would read the post-op value from the saved tensor address), so
            # the exact same arithmetic is replicated without in-place ops.
            contrastive_loss = contrastive_loss - contrastive_loss_det.max() - 1e-4
            contrastive_loss_det = contrastive_loss.detach()
            contrastive_loss = contrastive_loss / contrastive_loss_det.abs().mean() * pred_loss_det.abs().mean()
            # scale it
            contrastive_loss = contrastive_loss_weight * contrastive_loss
            # sample-wise loss
            sample_loss = pred_loss - contrastive_loss
            dm = ((1 - (1 + be) / epochs) * distribution_matching).squeeze()
            if distribution_matching > 0:
                with torch.no_grad():  # Stop-gradient via detach
                    sg_log_losses = pred_loss_det.log()
                    target_log_w = -sg_log_losses  # log(1/L)
                weights = loss_normalizer(target_dir, t_exp)  # it equals to log(1/loss)
                loss_weighted = (weights.detach() * dm).exp()  # it equals to 1/loss
                aux_loss = F.mse_loss(weights, target_log_w)
            else:
                loss_weighted = 1
                aux_loss = 0
            # scale loss by it's prediction
            weighed_loss = (loss_weighted * sample_loss).mean()
            loss = weighed_loss + dm * aux_loss
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            used = [(p, g) for p, g in zip(params, grads) if g is not None]
            for p, _ in used:
                used_ids.add(id(p))
            # clip the captured grad tensors in place. The norm uses the same
            # two-level formula as the eager clip_grad_norm_ (bit-identical),
            # computed with one fused foreach kernel; the scale multiply is
            # also one fused foreach kernel. No per-param copies: the fit loop
            # points p.grad directly at these pool tensors, whose addresses
            # are stable across replays.
            # fused foreach norm + scale kernels replace ~70 per-param
            # launches (results differ from eager at ULP level, which is fine)
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([g for _, g in used], 2.0)), 2.0)
            scale = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
            torch._foreach_mul_([g for _, g in used], scale)
            fn.grads = used
            return loss, pred_dir, target_dir
        tg = _CudaGraph.capture(fn, [bx, bc, bm, bp, bt, be, bi])
        if tg is None:
            self.cuda_graphs = False
            return None
        tg.grad_buffers = dict(fn.grads)
        tg.unused_params = [p for p in params if id(p) not in used_ids]
        self._train_graphs[key] = tg
        return tg
    def _get_reflow_graph(self, batch_size, opt, loss_normalizer, grad_clip_max_norm, distribution_matching, device):
        """
        Returns (capturing on first call) a CUDA graph of one reflow()
        distillation iteration: zero_grad + forward transports + losses +
        backward + clip. The optimizer step and batch sampling stay eager.
        Returns None (and disables graphs) if the capture fails.
        """
        key = 'reflow'
        rg = self._train_graphs.get(key)
        if rg is not None:
            return rg
        # shared gradient buffers (see _get_fit_graph): reflow's loss involves
        # both the model and the loss_normalizer
        params = list(self.parameters()) + list(loss_normalizer.parameters())
        used_ids = set()
        bx = torch.empty(batch_size, self.in_dim, device=device)
        by = torch.empty(batch_size, self.in_dim, device=device)
        bc = torch.empty(batch_size, self.conditional_dim or 1, device=device)
        def fn(bx, by, bc):
            y_pred = self._eager_to_target(bx, bc)
            x_pred = self._eager_to_prior(by, bc)
            forward_loss = F.mse_loss(by, y_pred, reduction='none').mean(-1)
            # no in-place ops inside the capture (see _get_fit_graph); note the
            # eager path does `-= (min - 1e-2)`, i.e. subtracts min and ADDS 1e-2
            forward_loss = forward_loss - forward_loss.min().detach() + 1e-2
            inverse_loss = F.mse_loss(bx, x_pred, reduction='none').mean(-1)
            inverse_loss = inverse_loss - inverse_loss.min().detach() + 1e-2
            prediction_loss = forward_loss.mean() + inverse_loss.mean()
            forward_weight, inverse_weight = loss_normalizer(torch.concat([bx, by], -1)).chunk(2, -1)
            forward_weight, inverse_weight = forward_weight[..., 0], inverse_weight[..., 0]
            normalizer_loss = F.mse_loss(forward_weight, forward_loss.detach().log()) + F.mse_loss(inverse_weight, inverse_loss.detach().log())
            fw = (-forward_weight * distribution_matching).detach().exp()
            iw = (-inverse_weight * distribution_matching).detach().exp()
            loss = (fw * forward_loss).mean() + (iw * inverse_loss).mean() + normalizer_loss * distribution_matching
            forward_r2 = r2_score(by, y_pred)
            inverse_r2 = r2_score(bx, x_pred)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            used = [(p, g) for p, g in zip(params, grads) if g is not None]
            for p, _ in used:
                used_ids.add(id(p))
            if grad_clip_max_norm is not None:
                # capture-safe clip (see _get_fit_graph): one fused norm
                # kernel + one fused scale kernel, no per-param copies
                norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([g for _, g in used], 2.0)), 2.0)
                scale = torch.clamp(grad_clip_max_norm / (norm + 1e-6), max=1.0)
                torch._foreach_mul_([g for _, g in used], scale)
            fn.grads = used
            return (loss, forward_r2, inverse_r2, prediction_loss,
                    forward_weight, inverse_weight, forward_loss, inverse_loss)
        rg = _CudaGraph.capture(fn, [bx, by, bc])
        if rg is None:
            self.cuda_graphs = False
            return None
        rg.grad_buffers = dict(fn.grads)
        rg.unused_params = [p for p in params if id(p) not in used_ids]
        self._train_graphs[key] = rg
        return rg