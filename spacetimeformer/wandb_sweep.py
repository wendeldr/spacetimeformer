import argparse
import multiprocessing
import wandb

import numpy as np
import pandas as pd
import spacetimeformer as stf
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import itertools
import random

from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)


default_config = {
    # fixed but maybe change...
    'batch_size':2000,
    'workers':2,
    'init_lr':1e-10,
    'base_lr':0.0005,
    'context_points':10,
    'target_points':1,
    'd_model':25,
    'd_qk':15,
    'd_v':15,
    'd_ff':100,
    'n_heads':1,
    'enc_layers':1,
    'dec_layers':1,
    'global_self_attn':'full',
    'local_self_attn':'full',
    'global_cross_attn':'full',
    'local_cross_attn':'full',
    'no_val':False,
    'no_time':False,
    'no_space':False,
    'no_given':False,

    # directly set parameters
    'gpus':[0],
    # 'gpus':None,
    'strategy':'dp',
    "time_resolution": 1,
    "start_token_len": 0,
    "attn_factor": 5,
    "dropout_emb": 0.2,
    "dropout_attn_out": 0,
    "dropout_attn_matrix": 0,
    "dropout_qkv": 0,
    "dropout_ff": 0.2,
    "pos_emb_type": 'abs',
    "no_final_norm": False,
    "performer_kernel": 'relu',
    "performer_redraw_interval": 100,
    "attn_time_windows": 1,
    "use_shifted_time_windows": False,
    "norm": 'batch',
    "activation": 'gelu',
    "warmup_steps": 10,
    "decay_factor": 0.25,
    "initial_downsample_convs": 0,
    "intermediate_downsample_convs": 0,
    "embed_method": 'spatio-temporal',
    "l2_coeff": 0.000001,
    "loss": 'mse',
    "class_loss_imp": 0.1,
    "recon_loss_imp": 0,
    "time_emb_dim": 6,
    "null_value": None,
    "pad_value": None,
    "linear_window": 0,
    "use_revin": False,
    "linear_shared_weights": False,
    "use_seasonal_decomp": False,
    "recon_mask_skip_all": 1,
    "recon_mask_max_seq_len": 5,
    "recon_mask_drop_seq": 0.2,
    "recon_mask_drop_standard": 0.1,
    "recon_mask_drop_full": 0.05,
}

sweep_config = {
    'method': 'random',  # Choose 'random' to randomly select values, other options include 'grid' for exhaustive search
    'parameters': {
        'base_lr': {
            'distribution': 'log_uniform',
            'min': -10,  # Use log scale, corresponds to 10^-10
            'max': -2,   # Use log scale, corresponds to 10^-2
        },
    },
    'metric': {
        'goal': 'minimize',  # Assuming you want to minimize the validation metric, adjust as necessary
        'name': 'val/forecast_loss'  # Replace 'val/smape' with the actual metric name you are using for evaluation
    },
}

def create_dataset(config):
    seed_everything(42, workers=True)

    size = 1000
    # np.random.seed(42)
    # random.seed(42)
    x1  = np.zeros(size)
    x2  = np.zeros(size)
    x3  = np.zeros(size)
    x4  = np.zeros(size)
    x5  = np.zeros(size)

    x1_seq = np.array([1,2,3])

    # embed sequence randomly several times in  x1. overlap is ok.
    for i in random.sample(range(0, len(x1)-len(x1_seq)), 75):
        x1[i:i+len(x1_seq)] = x1_seq

    # Define the numbers and the maximum length of the noise sequences
    numbers = [0, 1, 2, 3]
    max_length = 3

    # Generate all possible sequences of lengths 1 to max_length
    all_sequences = [seq for i in range(1, max_length + 1) for seq in itertools.product(numbers, repeat=i)]

    # Convert to numpy arrays and filter out the (1, 2, 3) sequence
    noise_sequences = [np.array(seq) for seq in all_sequences if seq != (1, 2, 3)]

    # Number of times to embed noise
    num_noise_embeddings = 50

    # Randomly embed noise sequences
    for _ in range(num_noise_embeddings):
        # Choose a random noise sequence
        noise_seq = random.choice(noise_sequences)
        # Choose a random start position; overlap is ok
        start_pos = random.randint(0, len(x1) - len(noise_seq))
        # Embed the noise sequence
        x1[start_pos:start_pos+len(noise_seq)] = noise_seq

    #### X2 ####
    # x2 responds to x1 and has the unique value of 5.  The response is 2 indexs after x1 produces 1,2,3. 
    responsetime = 2
    for i in range(len(x1) - (len(x1_seq) + responsetime)):
        if np.array_equal(x1[i:i+len(x1_seq)], x1_seq):
            s = i + len(x1_seq) + responsetime # start at the end of the sequence and add the response time
            x2[s] = 5

    #### X3 ####
    # second source
    x3_seq = np.array([4, 3, 2])

    # Embed x3_seq randomly in x3, allowing overlaps.
    for i in random.sample(range(0, len(x3) - len(x3_seq)), 75):
        x3[i:i+len(x3_seq)] = x3_seq

    # Generate noise for x3, similar to x1, excluding the sequence [4, 3, 2].
    numbers = [0, 1, 2, 3, 4]  # Including 4 since it's part of x3's unique sequence.
    max_length = 3
    all_sequences = [seq for i in range(1, max_length + 1) for seq in itertools.product(numbers, repeat=i)]
    noise_sequences = [np.array(seq) for seq in all_sequences if seq != tuple(x3_seq)]
    num_noise_embeddings = 50

    for _ in range(num_noise_embeddings):
        noise_seq = random.choice(noise_sequences)
        start_pos = random.randint(0, len(x3) - len(noise_seq))
        x3[start_pos:start_pos+len(noise_seq)] = noise_seq

    #### X4 ####
    # x4 responds to x3 with the sequence [3, -3] two indices after [4, 3, 2].
    responsetime = 5
    for i in range(len(x3) - (len(x3_seq) + responsetime+1)):
        if np.array_equal(x3[i:i+len(x3_seq)], x3_seq):
            s = i + len(x3_seq) + responsetime # start at the end of the sequence and add the response time
            x4[s] = 3
            x4[s+1] = -3

    #### X5 ####
    # x5 is abs inverted x3
    x5 = np.abs(-x3)

    data = np.array([x1, x2, x3, x4, x5]).T

    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    PLOT_VAR_NAMES = np.arange(5) + 1
    PLOT_VAR_IDXS = np.arange(5)
    df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
    df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="s")

    dset = stf.data.CSVTimeSeries(
        data_path=None,
        raw_df=df,
        val_split=0.1,
        test_split=0.1,
        normalize=True,
        time_col_name="Datetime",
        time_features=["second", 'millisecond', 'microsecond'],
    )
    yc_dim = data.shape[1]
    yt_dim = data.shape[1]
    x_dim = dset.time_cols.shape[0]

    DATA_MODULE = stf.data.DataModule(
        datasetCls=stf.data.CSVTorchDset,
        dataset_kwargs={
            "csv_time_series": dset,
            "context_points": config['context_points'],
            "target_points": config['target_points'],
            "time_resolution": config['time_resolution'],
        },
        batch_size=config['batch_size'],
        workers=config['workers'],
        overfit=False,
    )
    INV_SCALER = dset.reverse_scaling
    SCALER = dset.apply_scaling
    NULL_VAL = None
    INV_SCALER = dset.reverse_scaling
    SCALER = dset.apply_scaling
    NULL_VAL = None
    return (
        DATA_MODULE,
        INV_SCALER,
        SCALER,
        NULL_VAL,
        PLOT_VAR_IDXS,
        PLOT_VAR_NAMES,
        PAD_VAL,
        x_dim,
        yc_dim,
        yt_dim
    )

def create_model(config, x_dim, yc_dim, yt_dim):
    seed_everything(42, workers=True)
    max_seq_len = config['context_points'] + config['target_points']

    forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
        d_x=x_dim,
        d_yc=yc_dim,
        d_yt=yt_dim,
        max_seq_len=max_seq_len,    
        start_token_len=config['start_token_len'],
        attn_factor=config['attn_factor'],
        d_model=config['d_model'],
        d_queries_keys=config['d_qk'],
        d_values=config['d_v'],
        n_heads=config['n_heads'],
        e_layers=config['enc_layers'],
        d_layers=config['dec_layers'],
        d_ff=config['d_ff'],
        dropout_emb=config['dropout_emb'],
        dropout_attn_out=config['dropout_attn_out'],
        dropout_attn_matrix=config['dropout_attn_matrix'],
        dropout_qkv=config['dropout_qkv'],
        dropout_ff=config['dropout_ff'],
        pos_emb_type=config['pos_emb_type'],
        use_final_norm=not config['no_final_norm'],
        global_self_attn=config['global_self_attn'],
        local_self_attn=config['local_self_attn'],
        global_cross_attn=config['global_cross_attn'],
        local_cross_attn=config['local_cross_attn'],
        performer_kernel=config['performer_kernel'],
        performer_redraw_interval=config['performer_redraw_interval'],
        attn_time_windows=config['attn_time_windows'],
        use_shifted_time_windows=config['use_shifted_time_windows'],
        norm=config['norm'],
        activation=config['activation'],
        init_lr=config['init_lr'],
        base_lr=config['base_lr'],
        warmup_steps=config['warmup_steps'],
        decay_factor=config['decay_factor'],
        initial_downsample_convs=config['initial_downsample_convs'],
        intermediate_downsample_convs=config['intermediate_downsample_convs'],
        embed_method=config['embed_method'],
        l2_coeff=config['l2_coeff'],
        loss=config['loss'],
        class_loss_imp=config['class_loss_imp'],
        recon_loss_imp=config['recon_loss_imp'],
        time_emb_dim=config['time_emb_dim'],
        null_value=config['null_value'],
        pad_value=config['pad_value'],
        linear_window=config['linear_window'],
        use_revin=config['use_revin'],
        linear_shared_weights=config['linear_shared_weights'],
        use_seasonal_decomp=config['use_seasonal_decomp'],
        use_val=not config['no_val'],
        use_time=not config['no_time'],
        use_space=not config['no_space'],
        use_given=not config['no_given'],
        recon_mask_skip_all=config['recon_mask_skip_all'],
        recon_mask_max_seq_len=config['recon_mask_max_seq_len'],
        recon_mask_drop_seq=config['recon_mask_drop_seq'],
        recon_mask_drop_standard=config['recon_mask_drop_standard'],
        recon_mask_drop_full=config['recon_mask_drop_full'],
    )
    return forecaster


def train(config=None):
    seed_everything(42, workers=True)
    with wandb.init(config=config,project="sweep"):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        # place values from default_config into config if not already set
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]

        (data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
        pad_val,
        x_dim,
        yc_dim,
        yt_dim) = create_dataset(config)

        forecaster = create_model(config, x_dim=x_dim, yc_dim=yc_dim, yt_dim=yt_dim)
        forecaster.set_inv_scaler(inv_scaler)
        forecaster.set_scaler(scaler)
        forecaster.set_null_value(null_val)

        wandb_logger = WandbLogger()
    
        trainer = pl.Trainer(
            gpus=config.gpus, # which gpus to use
            logger=wandb_logger, # W&B integration
            strategy=config.strategy, # DataParallel (dp) or DDP
            callbacks=[pl.callbacks.LearningRateMonitor()], # various callbacks
            deterministic=True, # reproducibility
            # accumulate_grad_batches=1, # number of batches to accumulate gradients. batch_size * accumulate_grad_batches = actual batch size
            # gradient_clip_val=0,  # control gradient clipping. 0 means don't clip.
            # gradient_clip_algorithm="norm",
            # overfit_batches=0,
            # sync_batchnorm=False, # we use batch norm but we are not running across gpus so we don't need to sync I think. Default is true though so try without
            # limit_val_batches=1,
            max_epochs=25,
            log_every_n_steps=1,
            val_check_interval=1,

        )

        # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        trainer.test(datamodule=data_module, ckpt_path="best")

        wandb_logger.finalize("success")



# def train_nowand(config=default_config):   
#     # place values from default_config into config if not already set
#     for key in default_config:
#         if key not in config:
#             config[key] = default_config[key]

#     (data_module,
#     inv_scaler,
#     scaler,
#     null_val,
#     plot_var_idxs,
#     plot_var_names,
#     pad_val,
#     x_dim,
#     yc_dim,
#     yt_dim) = create_dataset(config)

#     forecaster = create_model(config, x_dim=x_dim, yc_dim=yc_dim, yt_dim=yt_dim)
#     forecaster.eval()

#     cp = config['context_points']
#     tp = config['target_points']
#     N_batch = config['batch_size']
#     N_time_representations = 1

#     # test_samples = next(iter(data_module.test_dataloader()))
#     # for x in test_samples:
#     #     print(x.shape)

#      # x_c, y_c, x_t, y_t,
#     # forecaster(torch.randn(N_batch, cp, N_time_representations),
#     #            torch.randn(N_batch, cp, yc_dim),
#     #            torch.randn(N_batch, tp, N_time_representations),
#     #            torch.randn(N_batch, tp, yc_dim))

#     # torch.onnx.export(forecaster,
#     #                   # x_c, y_c, x_t, y_t,
#     #              (torch.randn(1, cp, 1),torch.randn(1, cp, yc_dim),torch.randn(1, tp, 1),torch.randn(1, tp, yc_dim)),
#     #              "stf.onnx",
#     #              verbose=False,
#     #              export_params=True,
#     #              )
    
#     forecaster.set_inv_scaler(inv_scaler)
#     forecaster.set_scaler(scaler)
#     forecaster.set_null_value(null_val)

#     trainer = pl.Trainer(
#         gpus=config['gpus'],
#         strategy=config['strategy'],
#         accelerator=config['strategy'],
#         gradient_clip_val=0,
#         gradient_clip_algorithm="norm",
#         overfit_batches=0,
#         accumulate_grad_batches=1,
#         sync_batchnorm=False,
#         limit_val_batches=1,
#         max_epochs=15,
#         log_every_n_steps=1,
#         val_check_interval=1,
#         callbacks=[pl.callbacks.LearningRateMonitor()],
#         deterministic=True,
#     )

#     # Train
#     trainer.fit(forecaster, datamodule=data_module)
#     trainer.test(datamodule=data_module, ckpt_path="best")

# train_nowand()


def run_gpu_agent(sweep_id, gpu_id, run_count=None):
    # default run count of none lets agent run until no more runs are available. Since each agent is a process, this will run until all runs are complete.
    seed_everything(42, workers=True)

    def train_with_specific_gpu():
        seed_everything(42, workers=True)
        # Initialize or update the configuration with the specific GPU to use
        # Ensure 'config' is accessible and modifiable here, possibly passed as an argument or global
        modified_config = default_config.copy()
        modified_config['gpus'] = [gpu_id]  # Set to use a specific GPU

        # Call the original training function with the updated configuration
        train(config=modified_config)

    # Start a W&B agent with the modified training function
    wandb.agent(sweep_id=sweep_id, function=train_with_specific_gpu, count=run_count)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    wandb.login()
    project = 'learning_rate_sweep'
    sweep_id = wandb.sweep(sweep_config, project=project)

    # Number of GPUs available
    avaiable_gpus = [0, 1, 2, 3]
    agents_per_gpu = 8

    # Create a process for each GPU
    processes = []
    for gpu_id in avaiable_gpus:
        for agent_id in range(agents_per_gpu):
            p = multiprocessing.Process(target=run_gpu_agent, args=(sweep_id, gpu_id, None))
            p.start()
            processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()



