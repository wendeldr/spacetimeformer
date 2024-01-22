import argparse
import multiprocessing
import wandb



import numpy as np
import pandas as pd
import spacetimeformer as stf
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch


default_config = {
    # fixed but maybe change...
    'batch_size':100,
    'workers':1,
    'init_lr':1e-10,
    'base_lr':0.0005,
    'context_points':10,
    'target_points':1,
    'd_model':15,
    'd_qk':15,
    'd_v':15,
    'd_ff':60,
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
    "dropout_ff": 0.3,
    "pos_emb_type": 'abs',
    "no_final_norm": False,
    "performer_kernel": 'relu',
    "performer_redraw_interval": 100,
    "attn_time_windows": 1,
    "use_shifted_time_windows": False,
    "norm": 'batch',
    "activation": 'gelu',
    "warmup_steps": 0,
    "decay_factor": 0.25,
    "initial_downsample_convs": 0,
    "intermediate_downsample_convs": 0,
    "embed_method": 'spatio-temporal',
    "l2_coeff": 0.000001,
    "loss": 'mse',
    "class_loss_imp": 0.1,
    "recon_loss_imp": 0,
    "time_emb_dim": 3,
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
    'method': 'random', #grid, random
    'parameters': {
        'base_lr':   {'distribution': 'log_uniform_values', 'min': 1e-10, 'max': 0.01},
    },
    # only for bayes sweeps
    'metric': {
        'goal': 'minimize',
        'name': 'val/smape'
    },
}

def create_dataset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    fs = 2048  # Sampling rate (Hz)
    T = 10  # Length of epochs (s)
    f = 100  # Frequency of sinusoids (Hz)
    t = np.arange(0, T, 1 / fs)  # Time array
    A = 1  # Amplitude
    sigma = 0.1  # Gaussian noise variance

    # Damping/growth factor
    k = 4

    # Number of repetitions
    N = 6  # Replace with desired number of repetitions

    # Initializing the data array
    data = []

    # Phase differences for the sine waves
    phase_differences = [0, np.pi]
    names = ['0 (0°)', 'π (180°)']

    # Time variable for the repeating dampened wave
    repeating_t = t % (T / N)

    # Append dampened sine wave (repeating)
    dampened_wave = A * np.exp(-k * repeating_t) * np.sin(2 * np.pi * f * repeating_t)
    data.append(dampened_wave)

    # Append standard and phase-shifted sine waves
    for ps in phase_differences:
        # Create the sine wave with phase shift
        sig = A*np.sin(2 * np.pi * f * t - ps)
        data.append(sig)

    data = np.array(data).T
    # Update names for the new waves
    names.insert(0, "Dampened")

    PLOT_VAR_NAMES = names
    PLOT_VAR_IDXS = np.arange(0, len(names))

    df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
    df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="ms")

    dset = stf.data.CSVTimeSeries(
        data_path=None,
        raw_df=df,
        target_cols=PLOT_VAR_NAMES,
        ignore_cols=[],
        val_split=0.2,
        test_split=0.2,
        normalize=False,
        time_col_name="Datetime",
        time_features=["millisecond"],
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

def set_seed(seed=42):
    torch.manual_seed(seed)
    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config=None):
    with wandb.init(config=config,project="sweep"):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        # place values from default_config into config if not already set
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]

        set_seed(42)
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
            gpus=config.gpus,
            logger=wandb_logger,
            strategy=config.strategy,
            accelerator=config.strategy,
            gradient_clip_val=0,
            gradient_clip_algorithm="norm",
            overfit_batches=0,
            accumulate_grad_batches=1,
            sync_batchnorm=False,
            limit_val_batches=1,
            max_epochs=10,
            log_every_n_steps=1,
            val_check_interval=1,
            callbacks=[pl.callbacks.LearningRateMonitor()],
            deterministic=True,
        )

        # Train
        trainer.fit(forecaster, datamodule=data_module)


def train_nowand(config=default_config):   
    # place values from default_config into config if not already set
    for key in default_config:
        if key not in config:
            config[key] = default_config[key]

    set_seed(42)
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
    forecaster.eval()

    cp = config['context_points']
    tp = config['target_points']
    N_batch = config['batch_size']
    N_time_representations = 1

    # test_samples = next(iter(data_module.test_dataloader()))
    # for x in test_samples:
    #     print(x.shape)

     # x_c, y_c, x_t, y_t,
    forecaster(torch.randn(N_batch, cp, N_time_representations),
               torch.randn(N_batch, cp, yc_dim),
               torch.randn(N_batch, tp, N_time_representations),
               torch.randn(N_batch, tp, yc_dim))

    # torch.onnx.export(forecaster,
    #                   # x_c, y_c, x_t, y_t,
    #              (torch.randn(1, cp, 1),torch.randn(1, cp, yc_dim),torch.randn(1, tp, 1),torch.randn(1, tp, yc_dim)),
    #              "stf.onnx",
    #              verbose=False,
    #              export_params=True,
    #              )
    
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    trainer = pl.Trainer(
        gpus=config.gpus,
        strategy=config.strategy,
        accelerator=config.strategy,
        gradient_clip_val=0,
        gradient_clip_algorithm="norm",
        overfit_batches=0,
        accumulate_grad_batches=1,
        sync_batchnorm=False,
        limit_val_batches=1,
        max_epochs=10,
        log_every_n_steps=1,
        val_check_interval=1,
        callbacks=[pl.callbacks.LearningRateMonitor()],
        deterministic=True,
    )

    # Train
    trainer.fit(forecaster, datamodule=data_module)

train_nowand()


def run_agent(sweep_id,run_count=1):
    wandb.agent(sweep_id=sweep_id, function=train, count=run_count)




# if __name__ == "__main__":
#     set_seed(42)
    # wandb.login()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('project', type=str)
#     parser.add_argument('--agent_count', type=int, default=1)
#     parser.add_argument('--run_count', type=int, default=1)

#     args = parser.parse_args()

#     project= args.project
#     count = args.agent_count
#     run_count = args.run_count
#     # project = "baselr_sweep"
#     # count = 1

#     sweep_id = wandb.sweep(sweep_config, project=project)

#     processes = []
#     for i in range(count):
#         # Create a separate process for each agent
#         p = multiprocessing.Process(target=run_agent, args=(sweep_id,run_count))
#         processes.append(p)
#         p.start()

#     # Wait for all processes to complete
#     for p in processes:
#         p.join()



