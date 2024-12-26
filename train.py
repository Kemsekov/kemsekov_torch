import os
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import json
from accelerate import Accelerator
import math
import shutil
import time
from accelerate import load_checkpoint_and_dispatch

def train(
        model,
        train_loader,
        test_loader,
        optimizer,
        compute_loss_and_metric,
        metric_name,
        save_results_dir,
        load_checkpoint_dir = None,
        num_epochs = 10,
        accelerator : Accelerator = None,
        tie_weights=False, 
        cast_batch_to_mixed_precision_dtype = False,
        scheduler = None,
        checkpoints_count = 5,
        model_wrapper = None
    ):
    """
    Train and evaluate a model, saving checkpoints, plots, and metric history 
    during the training process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.

    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset, providing batched data.

    test_loader : torch.utils.data.DataLoader or None
        DataLoader for the test dataset, providing batched data. If None, the 
        testing step is skipped.

    optimizer : torch.optim.Optimizer
        Optimizer for updating model weights.

    compute_loss_and_metric : Callable
        A function that computes the loss and performance metric for each batch. 
        It should accept `model` and `batch` as input and return a tuple of 
        `(loss, metric)`.

    metric_name : str
        Name of the evaluation metric used (e.g., "accuracy"), for logging and plot titles.

    save_results_dir : str
        Directory to save model checkpoints, metric plots, and training history.

    load_checkpoint_dir : str, optional
        Directory path for loading a saved training checkpoint, if available.

    num_epochs : int
        Number of epochs for training.

    accelerator : accelerate.Accelerator, optional
        An instance of `Accelerator` from the `accelerate` library for distributed 
        training and mixed-precision optimizations. If None, a default instance 
        will be created.

    tie_weights : bool, default=False
        If True, calls `model.tie_weights()` at the start of training, useful for 
        models with shared weights.

    cast_batch_to_mixed_precision_dtype : bool, default=False
        If True, explicitly casts input batches to the model’s mixed-precision data type 
        to ensure compatibility with mixed-precision training.

    scheduler : optional
        Learning rate scheduler to update during the training loop.
    
    checkpoints_count: int, default=3
        How many last best checkpoints to save
    
    model_wrapper: torch.nn.DataParallel, default=None
        How to wrap model.
        
    Returns
    -------
    None
        Saves model checkpoints and training performance plots in `save_results_dir`.

    Notes
    -----
    - Creates a directory structure under `save_results_dir` to store model checkpoints, 
      loss, and metric plots.
    - Supports mixed-precision and distributed training through `accelerator`.
    - Checkpoints the best model based on the test metric; if the test metric matches 
      the best score, it also considers the training metric for improvement.
    - Generates plots for loss and metric history at the end of training, saving them as 
      "loss_history.png" and "{metric_name}_history.png" in the plot directory.
    - Periodically saves the training state and model in `checkpoints` based on 
      performance improvement.

    Example
    -------
    >>> train(
    >>>     model=model,
    >>>     train_loader=train_loader,
    >>>     test_loader=test_loader,
    >>>     optimizer=optimizer,
    >>>     compute_loss_and_metric=compute_loss_and_metric,
    >>>     metric_name="accuracy",
    >>>     save_results_dir="runs",
    >>>     load_checkpoint_dir="runs/checkpoints/epoch-10",
    >>>     num_epochs=10,
    >>>     accelerator=my_accelerator,
    >>>     cast_batch_to_mixed_precision_dtype=True
    >>> )

    Procedure
    ---------
    1. **Prepare Model and Data:** The model, data loaders, optimizer, and scheduler are 
       prepared with the `accelerator`.

    2. **Checkpoint Loading (Optional):** If `load_checkpoint_dir` is specified, attempts 
       to load training state and metric history. Adjusts the starting epoch and the 
       best observed metric based on the checkpointed values.

    3. **Epoch Training Loop:** For each epoch, loops over batches in `train_loader`.
       - If `cast_batch_to_mixed_precision_dtype` is True, casts inputs to the specified 
         dtype.
       - Computes loss and metric for each batch, backpropagates, and updates weights.
       - Accumulates batch loss and metric to compute the epoch's averages.

    4. **Evaluation:** After each epoch, evaluates the model on `test_loader` if provided. 
       Calculates the test metric and loss, appending them to the history.

    5. **Checkpointing and Plotting:** Saves the model checkpoint if the test metric 
       improves. At the end of training, generates and saves loss and metric plots 
       in the `plots` folder.
    """

    
    save_last_dir = os.path.join(save_results_dir,"last")
    plot_dir = os.path.join(save_last_dir, "plots")
    report_path = os.path.join(save_last_dir,"report.json")
    state_dir = os.path.join(save_last_dir,"state")

    checkpoints_dir = os.path.join(save_results_dir,"checkpoints")
    
    os.makedirs(save_last_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_train_metric = -1e10
    best_test_metric = -1e10
    train_metric_history = []   
    train_time_history = []   
    test_metric_history = []
    start_epoch=0

    loss_history = []
    test_loss_history = []

    acc = accelerator if accelerator is not None else Accelerator()
    if acc.is_main_process:
      try:
          print("trying to capture model architecture...")
          model_script = torch.jit.script(model)
          model_save_path=os.path.join(save_results_dir,"model.pt")
          model_script.save(model_save_path)
          print(f"Saved model architecture at {model_save_path}. You can torch.load it and update it's weights with checkpoint")
      except Exception as e:
          print(f"failed to compile model: {e}")
          model_script = None
    
    model, train_loader, test_loader,optimizer, scheduler = acc.prepare(model,train_loader,test_loader,optimizer, scheduler)

    if load_checkpoint_dir is not None and os.path.exists(load_checkpoint_dir):
        try:
            load_state_dir = os.path.join(load_checkpoint_dir,"state")
            acc.load_state(load_state_dir)
            print(f"loaded training state from {load_state_dir}")
            
            load_report_path = os.path.join(load_checkpoint_dir,"report.json")
            if os.path.exists(load_report_path):
                with open(load_report_path,'r') as f:
                    saved_state = json.loads(f.read())
                    if saved_state['metric_name']==metric_name:
                        train_metric_history = saved_state['train_metric_history']
                        test_metric_history = saved_state['test_metric_history']
                        if "train_time_history" in saved_state.keys():
                            train_time_history = saved_state['train_time_history']
                        
                        loss_history = saved_state['loss_history']
                        test_loss_history = saved_state['test_loss_history']
                        start_epoch = int(saved_state['epochs']) - 1
                        
                        train_metric_not_nan = [v for v in train_metric_history if not math.isnan(v)]
                        test_metric_not_nan = [v for v in test_metric_history if not math.isnan(v)]
                        if len(train_metric_not_nan)>0:
                            best_train_metric = max(train_metric_not_nan)
                        if len(test_metric_not_nan)>0:
                            best_test_metric = max(test_metric_not_nan)
        except Exception as e:
            print("Failed to load state with error",e)
            print("Ignoring state loading...")
    
    if tie_weights:
        model.tie_weights()
            
    if model_wrapper is not None:
        model = model_wrapper(model)
        
    mixed_precision = dtype_map[acc.mixed_precision]

    is_testing = test_loader is not None and len(test_loader)

    for epoch in range(start_epoch,num_epochs):
        if acc.is_main_process:
            print(f'\nEpoch {epoch+1}/{num_epochs}')
        # Training
        running_loss = 0.0
        metric = 0
        optimizer.zero_grad()  # Reset gradients before accumulation
        train_metric = 0.0
        pbar = tqdm(train_loader,desc=f"train {acc.process_index}")
        start = time.time()
        
        model.train()
        for batch in pbar:
            with acc.accumulate(model):
                # sometimes accelerate autocast fails to do it's job
                # and we need manually cast batch to required dtype
                if cast_batch_to_mixed_precision_dtype:
                    batch = cast_to_dtype(batch,mixed_precision)

                with acc.autocast():
                    loss, batch_metric = compute_loss_and_metric(model,batch)

                acc.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                batch_loss = loss.item()
                
                running_loss += batch_loss
                metric += batch_metric
                
                pbar.set_postfix(loss=f"{batch_loss:.4f}", **{metric_name: f"{batch_metric:.4f}"})
        running_time = time.time()-start
        train_time_history.append(running_time)
        running_loss /= len(train_loader)
        train_metric = metric / len(train_loader)
        train_metric_history.append(train_metric)
        loss_history.append(running_loss)
        
        # update train metric we see improvements
        if train_metric>best_train_metric:
            best_train_metric = train_metric

        # Evaluation on test set
        test_metric = 0.0
        test_loss = 0.0
        if is_testing:
            model.eval()
            with torch.no_grad():
                metric = 0
                for batch in test_loader:
                    if cast_batch_to_mixed_precision_dtype:
                        batch = cast_to_dtype(batch,mixed_precision)
                    with acc.autocast():
                        loss, batch_metric = compute_loss_and_metric(model,batch)
                    metric += batch_metric
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_metric = metric / len(test_loader)          
            test_metric_history.append(test_metric)
            test_loss_history.append(test_loss)

        if acc.is_main_process:
            if train_metric!=0:
                print(f'\tTrain {metric_name}: {train_metric:.4f}')
            if is_testing and test_metric!=0:
                print(f'\tTest  {metric_name}: {test_metric:.4f}')
            print(f'\tTrain Loss: {running_loss:.4f}')
            if is_testing:
                print(f'\tTest  Loss: {test_loss:.4f}')
               
        acc.save_state(state_dir)
        
        # create history plots in plots folder
        # Update to save loss and metric plots in separate files and only for the last epoch
        if acc.is_main_process:
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
            plt.figure()
            plt.plot(train_metric_history, label=f"Train {metric_name}")
            if is_testing:
                plt.plot(test_metric_history, label=f"Test {metric_name}")
            plt.title(f"{metric_name} History")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"{metric_name}_history.png"))
            plt.close()
            
            # time plot
            plt.figure()
            plt.plot(train_time_history)
            plt.title(f"Epoch execution time")
            plt.xlabel("Epoch")
            plt.ylabel("Seconds")
            plt.savefig(os.path.join(plot_dir, f"train_time_history.png"))
            plt.close()

            results = {
                "loss_history": [round(v,4) for v in loss_history],
                "test_loss_history" : [round(v,4) for v in test_loss_history],
                "train_metric_history" : [round(v,4) for v in train_metric_history],
                "test_metric_history" : [round(v,4) for v in test_metric_history],
                "train_time_history" : [round(v,4) for v in train_time_history],
                "metric_name" : metric_name,
                "epochs": epoch+1
            }

            results = json.dumps(results)
            with open(report_path,'w') as f:
                f.write(results) 
        # Save state if test metric improves or if test metric is same but train metric improved
        # note, metric always must suggest that the larger it is, the better model is performing
        if test_metric > best_test_metric or (test_metric==best_test_metric and train_metric>=best_train_metric):
            best_test_metric = test_metric
            if acc.is_main_process:
                # keep total count of saved checkpoints constant
                checkpoints = os.listdir(checkpoints_dir)
                checkpoints=sorted(checkpoints)
                if len(checkpoints)>=checkpoints_count:
                    for c in checkpoints[:-checkpoints_count+1]:
                        c_dir = os.path.join(checkpoints_dir,c)
                        shutil.rmtree(c_dir,ignore_errors=True)
                
                checkpoints_dir_with_epoch=os.path.join(checkpoints_dir,f"epoch-{epoch+1}")
                # for each improvement save training state and model
                # copy current saved state from last to checkpoint
                shutil.copytree(save_last_dir, checkpoints_dir_with_epoch,dirs_exist_ok=True)
                # update base model
                if model_script is not None:
                    model_script=load_last_checkpoint(model_script,save_results_dir,log=False)
                    model_script.save(model_save_path)


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

def load_best_checkpoint(model,base_path,log = True):
    checkpoints = os.path.join(base_path,'checkpoints')
    c = os.listdir(checkpoints)
    best = sorted(c,key=lambda x: int(x.split('-')[-1]))[-1]
    checkpoint = os.path.join(checkpoints,best,'state')
    if log:
        print("loading",checkpoint)
    model = load_checkpoint_and_dispatch(model,checkpoint)
    return model

def load_last_checkpoint(model,base_path,log=True):
    checkpoint = os.path.join(base_path,'last','state')
    if log:
        print("loading",checkpoint)
    model = load_checkpoint_and_dispatch(model,checkpoint)
    return model
        


def split_dataset(dataset,test_size=0.05,batch_size=8,num_workers = 16,prefetch_factor=2,random_state=123,startify=None):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    # split dataset
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=startify,
        random_state=random_state
    )

    train_data = Subset(dataset, train_idx)
    test_data = Subset(dataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,num_workers=num_workers,prefetch_factor=prefetch_factor)
    return train_data,test_data,train_loader, test_loader
