import gc
import os
from typing import List, Dict
import accelerate
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import json
from accelerate import Accelerator, load_checkpoint_and_dispatch
import math
import shutil
import time
import tabulate
from ema_pytorch.ema_pytorch import EMA

def _print_red(text: str) -> None:
    """Print bold red text."""
    print(f"\033[1;31m{text}\033[0m")

def _print_green(text: str) -> None:
    """Print bold green text."""
    print(f"\033[1;32m{text}\033[0m")

def _print_blue(text: str) -> None:
    """Print bold blue text."""
    print(f"\033[1;34m{text}\033[0m")

def train(
        model,
        train_loader,
        test_loader,
        compute_loss_and_metric,
        save_results_dir="runs/train",
        load_checkpoint_dir = None,
        num_epochs = 100,
        checkpoints_count = 1,
        save_on_metric_improve : str | List[str] = 'any',
        optimizer = None,
        scheduler = None,
        model_wrapper = None,
        accelerator : Accelerator = None,
        accelerate_args : None | Dict = None,
        ema_args :  None | Dict = None,
        gradient_clipping_max_norm = None,
        tie_weights=False, 
        cast_batch_to_mixed_precision_dtype = False,
        on_epoch_end = None,
        on_train_batch_end = None,
        on_test_batch_end = None,
        save_plots = False
    ):
    """
    Train and evaluate a model, saving checkpoints, plots, and metric history during the training process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.

    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset, providing batched data.

    test_loader : torch.utils.data.DataLoader or None
        DataLoader for the test dataset, providing batched data. If None, the testing step is skipped.

    compute_loss_and_metric : Callable
        A function that computes the loss and performance metric for each batch. It should accept `model,batch`
        as input and return a tuple of `(loss, metric)`, where `metric` is a dictionary like `{'r2': 0.1, 'iou': 0.4, ...}`.
        
        loss value can be list or tuple as well, in such case, backward will be called on all of these losses separately.
        You can use it when you have multiple optimizers/schedulers.

    save_results_dir : str
        Directory to save model checkpoints, metric plots, and training history.

    load_checkpoint_dir : str, optional
        Directory path for loading a saved training checkpoint, if available. Default is None.

    num_epochs : int, optional
        Number of epochs for training. Default is 10.

    checkpoints_count : int, optional
        Number of last best checkpoints to save. Default is 1.

    save_on_metric_improve: 'any', 'all', list[metric_name]
        When `'any'` a checkpoint will be saved when any of the metrics improve. If one or more metrics show improvement, a new checkpoint will be saved.
        
        When `'all'` a checkpoint will be saved only if all of the metrics improves.
        
        When `list[metric_name]` the checkpoint will only be saved if all of the metrics in the list show improvement.
    
    optimizer : torch.optim.Optimizer, List[torch.optim.Optimizer]
        Optimizer for updating model weights. Multiple optimizers can be passed to separately train differnt parts of network

    scheduler : torch.optim.lr_scheduler.LRScheduler, List[torch.optim.lr_scheduler.LRScheduler], None
        Learning rate scheduler to update during the training loop. Default is None.
        Multiple schedulers can be used at the same time.

    model_wrapper : torch.nn.DataParallel, optional
        Wrapper for the model (e.g., for multi-GPU training). Default is None.

    accelerator : accelerate.Accelerator, optional
        An instance of `Accelerator` from the `accelerate` library for distributed training and mixed-precision
        optimizations. If None, a default instance will be created. Default is None.

    accelerate_args: dict 
        contains arguments passed to accelerator. Default is None. For available arguments see https://huggingface.co/docs/accelerate/en/package_reference/accelerator\n
        `device_placement`: bool = True,\n
        `split_batches`: bool = _split_batches,\n
        `mixed_precision`: PrecisionType | str | None = None,\n
        `gradient_accumulation_steps`: int = 1,\n
        `cpu`: bool = False,\n
        `dataloader_config`: DataLoaderConfiguration | None = None,\n
        `deepspeed_plugin`: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None,\n
        `fsdp_plugin`: FullyShardedDataParallelPlugin | None = None,\n
        `megatron_lm_plugin`: MegatronLMPlugin | None = None,\n
        `rng_types`: list[str | RNGType] | None = None,\n
        `log_with`: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,\n
        `project_dir`: str | PathLike | None = None,\n
        `project_config`: ProjectConfiguration | None = None,\n
        `gradient_accumulation_plugin`: GradientAccumulationPlugin | None = None,\n
        `step_scheduler_with_optimizer`: bool = True,\n
        `kwargs_handlers`: list[KwargsHandler] | None = None,\n
        `dynamo_backend`: DynamoBackend | str | None = None,\n
        `deepspeed_plugins`: DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None\n
    ema_args: dict
        contains arguments passed to the EMA wrapper. If set to `None`, EMA is not used.\n
        **USE EMA WHEN YOU HAVE UNSTABLE TRAINING SETUP (GAN,VAE,DIFFUSION, FLOW).**\n
        `update_model_with_ema_every`: int | None = None\n
            Every N batches, blend EMA weights back into the online model. If `None` EMA will be merged to model after each epoch.\n
            (used for continual learning or "hare-to-tortoise" training).\n
        `update_every`: int = 1\n
            Frequency (in steps) to update EMA weights. A value > 1 reduces compute cost.\n
        `beta`: float = 0.9999\n
            Target EMA decay factor. Larger values make updates smoother but slower to adapt.\n
        `update_after_step`: int = 100\n
            Number of optimizer steps to wait before starting EMA updates.\n
            Helps avoid averaging unstable early weights.\n
        `power`: float = 2/3\n
            Exponent for EMA warmup schedule. Common values:\n
            - 1.0 = simple average\n
            - 2/3 (default) = for very long runs (~1M steps)\n
            - 3/4 = for shorter runs (~100k steps)\n
        `ema_model_device`: str = None\n
            Where to store EMA model. If `None` same device as online model is used.\n
            It can be for example `'cpu'` to decrease memory usage\n
            (weights will be moved automatically during updates).\n
        `inv_gamma`: float = 1.0\n
            Inverse gamma factor for EMA warmup schedule. Controls how quickly decay ramps up.\n
        `min_value`: float = 0.0\n
            Minimum allowed EMA decay rate (clamps the schedule).\n
        `param_or_buffer_names_no_ema`: set[str] = set()\n
            Names of parameters or buffers that should not be EMA-smoothed \n
            (they will be copied directly instead). Example: BN statistics.\n
        `ignore_names`: set[str] = set()\n
            Exact parameter/buffer names to exclude from EMA.\n
        `ignore_startswith_names`: set[str] = set()\n
            Exclude parameters/buffers whose names start with any prefix in this set.\n
        `include_online_model`: bool = True\n
            If True, the online model is included inside the EMA module and saved in state_dict.\n
            If False, the online model must be managed externally.\n
        
        `use_foreach`: bool = False\n
            If True and supported by PyTorch, uses fused foreach ops (`_foreach_lerp_`, `_foreach_copy_`)\n
            for faster EMA updates.\n
        `update_model_with_ema_beta`: float = 0.0\n
            Interpolation factor when updating online model with EMA:\n
            - 0.0 = full replacement with EMA weights\n
            - (0,1) = partial blend\n
        `forward_method_names`: tuple[str, ...] = ()\n
            Names of forward methods from the EMA model to expose directly on this wrapper.\n
        `move_ema_to_online_device`: bool = False\n
            If True, EMA model will be moved to the same device as the online model during updates.\n
        `coerce_dtype`: bool = False\n
            If True, casts EMA tensors to the same dtype as online model tensors during updates.\n
        `lazy_init_ema`: bool = False\n
            If True, defers EMA model initialization until the first update \n
            (useful if the model contains lazy modules like LazyLinear).\n
    
    gradient_clipping_max_norm: when not set to None will perform gradient clipping by limiting gradient norm to specified value

    tie_weights : bool, optional
        If True, calls `model.tie_weights()` at the start of training, useful for models with shared weights.
        Default is False.

    cast_batch_to_mixed_precision_dtype : bool, optional
        If True, explicitly casts input batches to the model's mixed-precision data type to ensure compatibility
        with mixed-precision training. Default is False.

    on_epoch_end : Callable(int, torch.nn.Module), optional
        Method that is called at the end of each epoch. The epoch index and model are passed to this method.
        Default is None.

    on_train_batch_end : Callable(model, batch, loss, metric), optional
        Method that is called at the end of each training batch. The model, batch, loss, and metric are passed
        to this method. Default is None.

    on_test_batch_end : Callable(model, batch, loss, metric), optional
        Method that is called at the end of each test batch. The model, batch, loss, and metric are passed to
        this method. Default is None.

    save_plots: create plots of metrics for saved checkpoints
    
    Returns
    -------
    None
        Saves model checkpoints, training performance plots, and metric history in `save_results_dir`.

    Notes
    -----
    - Creates a directory structure under `save_results_dir` to store model checkpoints, loss, and metric plots.
    - Supports mixed-precision and distributed training through `accelerator`.
    - Checkpoints the best model based on the test metric; if the test metric matches the best score, it also
      considers the training metric for improvement.
    - Generates plots for loss and metric history at the end of training, saving them as "loss_history.png" and
      "{metric_name}_history.png" in the plot directory.
    - Periodically saves the training state and model in `checkpoints` based on performance improvement.

    Example
    -------
    >>> train(
    >>>     model=model,
    >>>     train_loader=train_loader,
    >>>     test_loader=test_loader,
    >>>     optimizer=optimizer,
    >>>     compute_loss_and_metric=compute_loss_and_metric,
    >>>     save_results_dir="runs",
    >>>     load_checkpoint_dir="runs/checkpoints/epoch-10",
    >>>     num_epochs=10,
    >>>     accelerator=my_accelerator,
    >>>     cast_batch_to_mixed_precision_dtype=True
    >>> )

    Procedure
    ---------
    1. **Prepare Model and Data:** The model, data loaders, optimizer, and scheduler are prepared with the `accelerator`.
    2. **Checkpoint Loading (Optional):** If `load_checkpoint_dir` is specified, attempts to load training state and
       metric history. Adjusts the starting epoch and the best observed metric based on the checkpointed values.
    3. **Epoch Training Loop:** For each epoch, loops over batches in `train_loader`.
       - If `cast_batch_to_mixed_precision_dtype` is True, casts inputs to the specified dtype.
       - Computes loss and metric for each batch, backpropagates, and updates weights.
       - Accumulates batch loss and metric to compute the epoch's averages.
    4. **Evaluation:** After each epoch, evaluates the model on `test_loader` if provided. Calculates the test metric
       and loss, appending them to the history.
    5. **Checkpointing and Plotting:** Saves the model checkpoint if the test metric improves. At the end of training,
       generates and saves loss and metric plots in the `plots` folder.
    """
    if checkpoints_count==0:
        _print_red("WARNING!!! (checkpoints_count==0) No checkpoints will be saved!")
    # only if we use deepspeed
    if  accelerate_args is not None and 'deepspeed_plugins' in accelerate_args.keys():
        optimizer = accelerate.utils.DummyOptim(model.parameters())
        scheduler = accelerate.utils.DummyScheduler(optimizer)
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
    total_parameters = sum([v.numel() for v in model.parameters()])/1000/1000
    _print_blue(f"Total model parameters {total_parameters:0.2f} M")
    save_last_dir = os.path.join(save_results_dir,"last")
    plot_dir = os.path.join(save_last_dir, "plots")
    report_path = os.path.join(save_last_dir,"report.json")
    state_dir = os.path.join(save_last_dir,"state")

    checkpoints_dir = os.path.join(save_results_dir,"checkpoints")
    
    os.makedirs(save_last_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_train_metric = {}
    best_test_metric = {}
    train_metric_history = {}
    test_metric_history = {}
    
    train_time_history = []   
    start_epoch=0

    loss_history = []
    test_loss_history = []
    
    if on_epoch_end is None: on_epoch_end             = lambda x,y: None
    if on_train_batch_end is None: on_train_batch_end = lambda x,y,w,z: None
    if on_test_batch_end is None: on_test_batch_end   = lambda x,y,w,z: None
    
    if accelerate_args is None: accelerate_args = {}
    acc = accelerator if accelerator is not None else Accelerator(**accelerate_args)
    _print_blue("Using device "+str(acc.device))
    
    if load_checkpoint_dir is not None and os.path.exists(load_checkpoint_dir):
        try:
            load_report_path = os.path.join(load_checkpoint_dir,"report.json")
            if os.path.exists(load_report_path):
                with open(load_report_path,'r') as f:
                    saved_state = json.loads(f.read())
                    if "train_time_history" in saved_state.keys():
                        train_time_history = saved_state['train_time_history']
                    
                    loss_history = saved_state['loss_history']
                    test_loss_history = saved_state['test_loss_history']
                    start_epoch = int(saved_state['epochs'])
                    
                    #train_metric_history is dict like {'r2':0.5,'iou':0.2,'f1':0.6 ... }
                    train_metric_history = saved_state['train_metric_history']
                    test_metric_history = saved_state['test_metric_history']
                    
                    # from each metric history metric get best metrics
                    load_best_metric_from_history(best_train_metric, train_metric_history)
                    load_best_metric_from_history(best_test_metric, test_metric_history)
            _print_green("Loaded training history")
        except Exception as e:
            _print_red("Failed to load training history\n"+str(e))
            _print_blue("Ignoring training history loading...")
        if start_epoch>=num_epochs:
            return model
    
    if not isinstance(optimizer,list) and not isinstance(optimizer,tuple):
        optimizer = [optimizer]
    if not isinstance(scheduler,list) and not isinstance(scheduler,tuple):
        scheduler = [scheduler]
        
    model_acc, train_loader, test_loader,*remaining = acc.prepare(model,train_loader,test_loader,*optimizer, *scheduler)
    optimizer_acc = remaining[:len(optimizer)]
    scheduler_acc = remaining[-len(scheduler):]
    
    if load_checkpoint_dir is not None and os.path.exists(load_checkpoint_dir):
        try:
            load_state_dir = os.path.join(load_checkpoint_dir,"state")
            acc.load_state(load_state_dir)
            _print_green(f"Loaded training state from {load_state_dir}")
        except Exception as e:
            _print_red("Failed to load state with error\n"+str(e))
            _print_blue("Ignoring state loading...")

    if acc.is_main_process:
      try:
          _print_blue("Trying to capture model architecture...")
          model_script = torch.jit.script(model)
          model_save_path=os.path.join(save_results_dir,"model.pt")
          model_script.save(model_save_path)
          _print_green(f"Saved model architecture at {model_save_path}. You can torch.load it and update it's weights with checkpoint")
      except Exception as e:
          print(f"Failed to compile model:\n{e}")
          model_script = None
    model=model_acc

    def backward_loss(acc,loss):
        """Returns true if computed backwards"""
        is_nan = False
        loss_is_iterable = isinstance(loss,list) or isinstance(loss,tuple)
        if loss_is_iterable:
            for l in loss:
                if torch.isnan(l).any() or torch.isinf(l).any():
                    is_nan = True
                else:
                    acc.backward(l)
        else:
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                is_nan = True
            else:
                acc.backward(loss)
        return not is_nan
    
    if tie_weights:
        model.tie_weights()
            
    if model_wrapper is not None:
        model = model_wrapper(model)
    
    update_ema = lambda : 0
    update_model_with_ema = lambda : 0
    if ema_args is not None:
        _print_green("Using EMA for training")
        ema = EMA(model,**ema_args)
        update_ema = lambda: ema.update()
        if 'update_model_with_ema_every' not in ema_args:
            update_model_with_ema = lambda : ema.update_model_with_ema()
    
    mixed_precision = dtype_map[acc.mixed_precision]

    is_testing = test_loader is not None and len(test_loader)
    
    def limit_grad():
        if acc.sync_gradients: 
            # Only clip when gradients are synced
            acc.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_max_norm)
    
    if gradient_clipping_max_norm is not None:
        gradient_clipping_max_norm = float(gradient_clipping_max_norm)
        grad_norm = limit_grad
    else:
        grad_norm = lambda : 0
    
    try:
        for epoch in range(start_epoch,num_epochs):
            if acc.is_main_process:
                print(f'\nEpoch {epoch+1}/{num_epochs}')
            # Training
            running_loss = 0.0
            metric = {}
            for opt in optimizer_acc:
                opt.zero_grad()  # Reset gradients before accumulation
            pbar = tqdm(train_loader,desc=f"train {acc.process_index}")
            start = time.time()
            NANS_COUNT = 0
            model.train()
            for batch in pbar:
                with acc.accumulate(model):
                    for opt in optimizer_acc:
                        opt.zero_grad()
                    
                    # sometimes accelerate autocast fails to do it's job
                    # and we need manually cast batch to required dtype
                    if cast_batch_to_mixed_precision_dtype:
                        batch = cast_to_dtype(batch,mixed_precision)

                    with acc.autocast():
                        loss, batch_metric = compute_loss_and_metric(model,batch)
                    
                    grad_norm()
                    
                    if not backward_loss(acc,loss):
                        NANS_COUNT+=1
                        gc.collect()
                        torch.cuda.empty_cache()
                        pbar.set_postfix(nan_count=NANS_COUNT)
                        # with torch.no_grad():
                        #     for p in model.parameters():
                        #         nan_mask = torch.isnan(p) | torch.isinf(p)
                        #         nan_params = p[nan_mask]
                        #         p[nan_mask]=torch.randn_like(nan_params)
                        continue
                    
                    add_batch_metric(metric, batch_metric)
                    for opt in optimizer_acc:
                        opt.step()
                    
                    for sch in scheduler_acc:
                        if sch is None: continue
                        sch.step()
                    
                    if isinstance(loss,list) or isinstance(loss,tuple):
                        loss = sum([l.detach() for l in loss])/len(loss)
                    
                    batch_loss = loss.item()
                    running_loss += batch_loss
                    
                    metrics_render = {name: f"{batch_metric[name]:.4f}"[:6] for name in batch_metric}
                    if NANS_COUNT>0:
                        metrics_render['nan_count'] = NANS_COUNT
                    pbar.set_postfix(loss=f"{batch_loss:.4f}"[:6], **metrics_render)
                    
                    update_ema()
                    
                    on_train_batch_end(model,batch,loss,batch_metric)
            
            update_model_with_ema()
            running_time = time.time()-start
            train_time_history.append(running_time)
            running_loss /= len(train_loader)
            loss_history.append(round(running_loss,5))
            
            train_metric = update_metric(train_loader, best_train_metric, train_metric_history, metric)

            # Evaluation on test set
            test_loss = 0.0
            test_metric = None
            if is_testing:
                model.eval()
                with torch.no_grad():
                    metric = {}
                    for batch in test_loader:
                        if cast_batch_to_mixed_precision_dtype:
                            batch = cast_to_dtype(batch,mixed_precision)
                        with acc.autocast():
                            loss, batch_metric = compute_loss_and_metric(model,batch)
                        if isinstance(batch_metric,torch.Tensor):
                            batch_metric = batch_metric.detach().cpu()
                        add_batch_metric(metric, batch_metric)
                        loss_is_iterable = isinstance(loss,list) or isinstance(loss,tuple)
                        if loss_is_iterable:
                            loss = sum([l.detach() for l in loss])/len(loss)
                    
                        test_loss += loss.item()
                        on_test_batch_end(model,batch,loss,batch_metric)

                test_loss /= len(test_loader)
                test_loss_history.append(round(test_loss,5))
                test_metric = update_metric(test_loader, best_test_metric, test_metric_history, metric)
            
            metrics = train_metric.keys()
            for d in [train_metric_history,test_metric_history,best_test_metric,best_train_metric]:
                if d is None: continue
                for m in list(d):
                    if m not in metrics:
                        d.pop(m)
            
            if acc.is_main_process:
                table_data = []
                loss_row = ["loss",round(running_loss,5)]
                if is_testing: 
                    loss_row.append(round(test_loss,5))
                table_data.append(loss_row)
                for name in train_metric:
                    row = [name, f'{train_metric[name]:.4f}']
                    if is_testing:
                        row.append(f'{test_metric[name]:.4f}')
                    table_data.append(row)
                
    
                # Define the headers
                headers = ['', 'Train']
                if is_testing:
                    headers.append('Test')

                # Print the table
                print(tabulate.tabulate(table_data, headers=headers, tablefmt='pretty'))

            # print('save')
            
            # create history plots in plots folder
            # Update to save loss and metric plots in separate files and only for the last epoch
            if acc.is_main_process:
                if save_plots:
                    # Loss plot
                    plt.figure()
                    plt.plot(loss_history, label="Train Loss")
                    if is_testing:
                        plt.plot(test_loss_history, label=f"Test Loss")
                    plt.title("Loss History")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, "loss_history.png"))
                    plt.close()

                    # Metric plot
                    save_plot_metric_history(plot_dir, train_metric_history,test_metric_history if is_testing else None,'train')
                    
                    # time plot
                    plt.figure()
                    plt.plot(train_time_history)
                    plt.title(f"Epoch execution time")
                    plt.xlabel("Epoch")
                    plt.ylabel("Seconds")
                    plt.savefig(os.path.join(plot_dir, f"train_time_history.png"))
                    plt.close()

                results = {
                    "loss_history"          : loss_history,
                    "test_loss_history"     : test_loss_history,
                    "train_metric_history"  : train_metric_history,
                    "test_metric_history"   : test_metric_history,
                    "train_time_history"    : [round(v,3) for v in train_time_history],
                    "epochs"                : epoch+1
                }

                results = json.dumps(results)
                with open(report_path,'w') as f:
                    f.write(results) 

            # Check if the test metric improves based on the specified condition
            test_improvements = (test_metric is not None) and (
                (save_on_metric_improve == 'any' and any(test_metric[m] >= best_test_metric.get(m, -1e10) for m in test_metric)) or
                (save_on_metric_improve == 'all' and all(test_metric[m] >= best_test_metric.get(m, -1e10) for m in test_metric)) or
                (isinstance(save_on_metric_improve, list) and all(test_metric.get(m, -1e10) >= best_test_metric.get(m, -1e10) for m in save_on_metric_improve))
            )

            # Check if the train metric improves, using similar logic to `test_improvements`
            train_improvements = (train_metric is not None) and (
                (save_on_metric_improve == 'any' and any(train_metric[m] > best_train_metric.get(m, -1e10) for m in train_metric)) or
                (save_on_metric_improve == 'all' and all(train_metric[m] > best_train_metric.get(m, -1e10) for m in train_metric)) or
                (isinstance(save_on_metric_improve, list) and all(train_metric.get(m, -1e10) >= best_train_metric.get(m, -1e10) for m in save_on_metric_improve))
            )
            
            if checkpoints_count>0 and (test_improvements or (not is_testing and train_improvements)):
                acc.save_state(state_dir)
                best_test_metric = test_metric
                if acc.is_main_process and checkpoints_count>1:
                    # keep total count of saved checkpoints constant
                    checkpoints = os.listdir(checkpoints_dir)
                    checkpoints=sorted(checkpoints,key=lambda x: int(x.split('-')[-1]))
                    if len(checkpoints)>=checkpoints_count:
                        for c in checkpoints[:-checkpoints_count+1]:
                            c_dir = os.path.join(checkpoints_dir,c)
                            shutil.rmtree(c_dir,ignore_errors=True)
                    if checkpoints_count==1:
                        for c in checkpoints:
                            c_dir = os.path.join(checkpoints_dir,c)
                            shutil.rmtree(c_dir,ignore_errors=True)
                        
                    checkpoints_dir_with_epoch=os.path.join(checkpoints_dir,f"epoch-{epoch+1}")
                    # for each improvement save training state and model
                    # copy current saved state from last to checkpoint
                    _print_green(f"saved epoch-{epoch+1}")
                    shutil.copytree(save_last_dir, checkpoints_dir_with_epoch,dirs_exist_ok=True)

            on_epoch_end(epoch,model)
    except KeyboardInterrupt:
        _print_red("Interrupt training")
    gc.collect()
    try:
        torch.cuda.empty_cache()
        _print_green("Empty cuda cache complete")
    except: pass
    return model

def save_plot_metric_history(plot_dir, train_metric_history,test_metric_history,source):
    metrics_count = len(train_metric_history.keys())
    
    plt.figure(figsize=(6*metrics_count,6))
    for i,metric_name in enumerate(train_metric_history):
        plt.subplot(1,metrics_count,i+1)
        plt.plot(train_metric_history[metric_name], label='train')
        if test_metric_history is not None:
            plt.plot(test_metric_history[metric_name], label='test')
        
        plt.xlabel("Epoch")
        plt.ylabel('Metric value')
        plt.title(f"{metric_name}")
        plt.legend()
    plt.suptitle(f'Metrics')
    plt.savefig(os.path.join(plot_dir, f"metrics.png"))
    plt.close()

def update_metric(train_loader, best_train_metric, train_metric_history, metric):
    
    train_metric =  {name : metric[name]/ len(train_loader) for name in metric}
    for m in train_metric:
        if m not in train_metric_history.keys():
            train_metric_history[m]=[]
        if m not in best_train_metric:
            best_train_metric[m]=-1e10
        train_metric_history[m].append(round(float(train_metric[m]),5))
            # update train metric we see improvements
        if train_metric[m]>best_train_metric[m]:
            best_train_metric[m]=train_metric[m]
    return train_metric

def add_batch_metric(metric, batch_metric):
    for m in batch_metric:
        metric_val = batch_metric[m]
        if not isinstance(metric_val,torch.Tensor):
            metric_val = torch.tensor(metric_val)
        v = metric_val.detach().cpu()
        if m not in metric.keys():
            metric[m]=0
        is_value = (not torch.isnan(v).any()) and (not torch.isinf(v).any())
        if is_value:
            v=v.numpy()
            batch_metric[m] = v
            metric[m] += v

def load_best_metric_from_history(best_train_metric, train_metric_history):
    for metric_name in train_metric_history:
        metric_history = train_metric_history[metric_name]
        best_train_metric[metric_name] = -1e-10
        non_nan = [v for v in metric_history if not math.isnan(v)]
        if len(non_nan)>0:
            best_train_metric[metric_name] = max(non_nan)

def cast_to_dtype(inputs,dtype):
    """Casts tensors, lists and dicts tensors to given dtype"""
    allowed_dtypes = [torch.float32,torch.float64]
    if isinstance(inputs,torch.Tensor) and inputs.dtype in allowed_dtypes:
        return torch.as_tensor(inputs,dtype=dtype)
    
    if isinstance(inputs,list):
        return [cast_to_dtype(v,dtype) for v in inputs]
    
    if isinstance(inputs,tuple):
        return (cast_to_dtype(v,dtype) for v in inputs)
    
    if isinstance(inputs,dict):
        return {v:cast_to_dtype(inputs[v],dtype) for v in inputs}
    
    return inputs
    

dtype_map = {
    'float32': torch.float32,
    'no': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'int32': torch.int32,
    'int64': torch.int64,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
}

def load_best_checkpoint(model,base_path,log = True,device_map=None):
    return load_checkpoint(model,base_path,-1,log,device_map=device_map)

def load_last_checkpoint(model,base_path,log=True,device_map=None):
    checkpoint = os.path.join(base_path,'last','state')
    if log:
        print("loading",checkpoint)
    model = load_checkpoint_and_dispatch(model,checkpoint,device_map=device_map)
    return model

def load_checkpoint(model,base_path,checkpoint_index,log=True,device_map=None):
    checkpoints = os.path.join(base_path,'checkpoints')
    c = os.listdir(checkpoints)
    best = sorted(c,key=lambda x: int(x.split('-')[-1]))[checkpoint_index]
    checkpoint = os.path.join(checkpoints,best,'state')
    if log:
        print("loading",checkpoint)
    model = load_checkpoint_and_dispatch(model,checkpoint,device_map=device_map)
    return model

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from kemsekov_torch.utils import BinBySizeDataset
def split_dataset(dataset,test_size=0.05,batch_size=8,num_workers = 16,prefetch_factor=2,startify=None,bin_by_size = False,shuffle=True,random_state=123, pin_memory = True):
    """
    Splits dataset into train and test parts
    
    dataset: dataset to split
    
    test_size: size of test dataset relative to train
    
    batch_size: batch size for train/test dataloaders
    
    num_workers: CPU's count to prepare batch
    
    prefetch_factor: how many batches to prepare in advance
    
    startify: If not None, data is split in a stratified fashion, using this as the class labels. 
    
    bin_by_size: Analyzes whole dataset and bins elements with sample shape into chunks of size `batch_size`, allowing dataloader to load data with different shapes
    
    shuffle: shuffle train/test dataset
    
    random_state: random state for dataset shuffle
    
    Returns:
        train_dataset,test_dataset,train_loader, test_loader
    """
    # split dataset
    if test_size>0:
        train_idx, test_idx = train_test_split(
            list(range(len(dataset))),
            test_size=test_size,
            stratify=startify,
            shuffle=shuffle,
            random_state=random_state
        )

        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)
        if bin_by_size:
            train_data=BinBySizeDataset(train_data,batch_size,batch_size//8,max_workers=num_workers)
            test_data=BinBySizeDataset(test_data,batch_size,batch_size//8,max_workers=num_workers)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,pin_memory=pin_memory)
        print("Train items",len(train_data))
        print("Test items",len(test_data))
        return train_data,test_data,train_loader, test_loader
    else:
        if shuffle:
            g = torch.Generator()
            g.manual_seed(random_state)
            train_idx = torch.randperm(len(dataset),generator=g)
            dataset = Subset(dataset, train_idx)
        if bin_by_size:
            dataset=BinBySizeDataset(dataset,batch_size,batch_size//8,max_workers=num_workers)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor,pin_memory=pin_memory)
        print("Train items",len(dataset))
        return dataset,[],train_loader,[]
        

class EpochSplittedDataloader:
    def __init__(self, dataset, batch_size, num_parts, shuffle=False, **dataloader_kwargs):
        """
        A wrapper around dataset to split a dataset into multiple parts for training across epochs.
        So you can split dataset into 3 parts, and it will iterate over full dataset in three epochs.
        
        Args:
            dataset (Dataset): PyTorch dataset.
            batch_size (int): Number of samples per batch.
            num_parts (int): Number of parts to split the dataset into.
            shuffle (bool): Whether to shuffle indices for each part.
            **dataloader_kwargs: Additional arguments for the DataLoader.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.dataloader_kwargs = dataloader_kwargs
        
        # Create indices for the dataset
        self.indices = list(range(len(self.dataset)))
        
        # Shuffle indices if required
        if self.shuffle:
            torch.manual_seed(42)  # For reproducibility
            self.indices = torch.randperm(len(self.dataset)).tolist()
        
        # Split indices into parts
        self.split_indices = torch.chunk(torch.tensor(self.indices), self.num_parts)
        self.current_part = 0

    def __iter__(self):
        """Return the DataLoader for the current part."""
        current_indices = self.split_indices[self.current_part].tolist()
        subset = Subset(self.dataset, current_indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, **self.dataloader_kwargs)
        self.current_part = (self.current_part + 1) % self.num_parts  # Move to the next part for the next epoch
        return iter(dataloader)

    def __len__(self):
        """Return the number of batches in the current part."""
        current_indices = self.split_indices[self.current_part]
        return len(current_indices) // self.batch_size + int(len(current_indices) % self.batch_size > 0)

