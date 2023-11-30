import wandb
wandb.login()


import numpy as np
import pandas as pd
import spacetimeformer as stf
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch


def create_dataset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None
    fs = 2048  # sampling rate (Hz)
    T = 2  # length of epochs (s)
    f = 100  # frequency of sinusoids (Hz)
    t = np.arange(0, T, 1 / fs)
    A = 1  # Amplitude
    sigma = 0.1  # Gaussian noise variance

    # Damping/growth factor
    k = 0.1

    # Initializing the data array
    data = []

    # Phase differences for the sine waves
    phase_differences = [0, np.pi]
    names = ['0 (0°)', 'π (180°)']

    # Append dampened sine wave
    dampened_wave = A * np.exp(-k * t) * np.sin(2 * np.pi * f * t)
    data.append(dampened_wave)

    # Append standard and phase-shifted sine waves
    for ps in phase_differences:
        # Create the sine wave with phase shift
        sig = np.sin(2 * np.pi * f * t - ps)
        data.append(sig)

    data = np.array(data).T
    # Update names for the new waves
    names.insert(0, "Dampened")

    PLOT_VAR_NAMES = names
    PLOT_VAR_IDXS = np.arange(0, len(names))

    df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
    df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="D")

    dset = stf.data.CSVTimeSeries(
        data_path=None,
        raw_df=df,
        target_cols=PLOT_VAR_NAMES,
        ignore_cols=[],
        val_split=0.2,
        test_split=0.2,
        normalize=True,
        time_col_name="Datetime",
        time_features=["day"],
    )
    yc_dim = data.shape[1]
    yt_dim = data.shape[1]
    x_dim = dset.time_cols.shape[0]

    DATA_MODULE = stf.data.DataModule(
        datasetCls=stf.data.CSVTorchDset,
        dataset_kwargs={
            "csv_time_series": dset,
            "context_points": config.context_points,
            "target_points": config.target_points,
            "time_resolution": config.time_resolution,
        },
        batch_size=config.batch_size,
        workers=config.workers,
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
    max_seq_len = config.context_points + config.target_points

    forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
        d_x=x_dim,
        d_yc=yc_dim,
        d_yt=yt_dim,
        max_seq_len=max_seq_len,
        start_token_len=config.start_token_len,
        attn_factor=config.attn_factor,
        d_model=config.d_model,
        d_queries_keys=config.d_qk,
        d_values=config.d_v,
        n_heads=config.n_heads,
        e_layers=config.enc_layers,
        d_layers=config.dec_layers,
        d_ff=config.d_ff,
        dropout_emb=config.dropout_emb,
        dropout_attn_out=config.dropout_attn_out,
        dropout_attn_matrix=config.dropout_attn_matrix,
        dropout_qkv=config.dropout_qkv,
        dropout_ff=config.dropout_ff,
        pos_emb_type=config.pos_emb_type,
        use_final_norm=not config.no_final_norm,
        global_self_attn=config.global_self_attn,
        local_self_attn=config.local_self_attn,
        global_cross_attn=config.global_cross_attn,
        local_cross_attn=config.local_cross_attn,
        performer_kernel=config.performer_kernel,
        performer_redraw_interval=config.performer_redraw_interval,
        attn_time_windows=config.attn_time_windows,
        use_shifted_time_windows=config.use_shifted_time_windows,
        norm=config.norm,
        activation=config.activation,
        init_lr=config.init_lr,
        base_lr=config.base_lr,
        warmup_steps=config.warmup_steps,
        decay_factor=config.decay_factor,
        initial_downsample_convs=config.initial_downsample_convs,
        intermediate_downsample_convs=config.intermediate_downsample_convs,
        embed_method=config.embed_method,
        l2_coeff=config.l2_coeff,
        loss=config.loss,
        class_loss_imp=config.class_loss_imp,
        recon_loss_imp=config.recon_loss_imp,
        time_emb_dim=config.time_emb_dim,
        null_value=config.null_value,
        pad_value=config.pad_value,
        linear_window=config.linear_window,
        use_revin=config.use_revin,
        linear_shared_weights=config.linear_shared_weights,
        use_seasonal_decomp=config.use_seasonal_decomp,
        use_val=not config.no_val,
        use_time=not config.no_time,
        use_space=not config.no_space,
        use_given=not config.no_given,
        recon_mask_skip_all=config.recon_mask_skip_all,
        recon_mask_max_seq_len=config.recon_mask_max_seq_len,
        recon_mask_drop_seq=config.recon_mask_drop_seq,
        recon_mask_drop_standard=config.recon_mask_drop_standard,
        recon_mask_drop_full=config.recon_mask_drop_full,
    )
    return forecaster


def set_seed(seed=42):
    torch.manual_seed(seed)
    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def train(config=None):
    with wandb.init(config=config,project="sweep"):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
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


        # wandb_logger.watch(module.net)
    
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


sweep_config = {
    'method': 'bayes', #grid, random
    'parameters': {
        # random search
        # 'init_lr':{'distribution': 'uniform', 'min': 1e-10, 'max': 1e-5},
        'base_lr':   {'distribution': 'uniform', 'min': 1e-10, 'max': 1},
        # grid search
        # 'no_val':{"values": [True,False]},
        # 'no_time':{"values": [True,False]},
        # 'no_space':{"values": [True,False]},
        # 'no_given':{"values": [True,False]},
        # 'init_lr':{"value": 1e-10},
        # 'base_lr':{"value": 0.0005},

        # fixed but maybe change...
        'batch_size':{"value": 2000},
        'workers':{"value": 10},
        'init_lr':{"value": 1e-10},
        # 'base_lr':{"value": 0.0005},
        'context_points':{"value": 32},
        'target_points':{"value": 2},
        'd_model':{"value": 100},
        'd_qk':{"value": 100},
        'd_v':{"value": 100},
        'd_ff':{"value": 400},
        'n_heads':{"value": 1},
        'enc_layers':{"value": 1},
        'dec_layers':{"value": 1},
        'global_self_attn':{"value": 'full'},
        'local_self_attn':{"value": 'full'},
        'global_cross_attn':{"value": 'full'},
        'local_cross_attn':{"value": 'full'},
        'no_val':{"value": False},
        'no_time':{"value": True},
        'no_space':{"value": True},
        'no_given':{"value": True},

        # directly set parameters
        'gpus':{"value": [0]},
        'strategy':{"value": 'dp'},
        "time_resolution": {"value": 1},
        "start_token_len": {"value": 0},
        "attn_factor": {"value": 5},
        "dropout_emb": {"value": 0.2},
        "dropout_attn_out": {"value": 0},
        "dropout_attn_matrix": {"value": 0},
        "dropout_qkv": {"value": 0},
        "dropout_ff": {"value": 0.3},
        "pos_emb_type": {"value": 'abs'},
        "no_final_norm": {"value": False},
        "performer_kernel": {"value": 'relu'},
        "performer_redraw_interval": {"value": 100},
        "attn_time_windows": {"value": 1},
        "use_shifted_time_windows": {"value": False},
        "norm": {"value": 'batch'},
        "activation": {"value": 'gelu'},
        "warmup_steps": {"value": 0},
        "decay_factor": {"value": 0.25},
        "initial_downsample_convs": {"value": 0},
        "intermediate_downsample_convs": {"value": 0},
        "embed_method": {"value": 'spatio-temporal'},
        "l2_coeff": {"value": 0.000001},
        "loss": {"value": 'mse'},
        "class_loss_imp": {"value": 0.1},
        "recon_loss_imp": {"value": 0},
        "time_emb_dim": {"value": 6},
        "null_value": {"value": None},
        "pad_value": {"value": None},
        "linear_window": {"value": 0},
        "use_revin": {"value": False},
        "linear_shared_weights": {"value": False},
        "use_seasonal_decomp": {"value": False},
        "recon_mask_skip_all": {"value": 1},
        "recon_mask_max_seq_len": {"value": 5},
        "recon_mask_drop_seq": {"value": 0.2},
        "recon_mask_drop_standard": {"value": 0.1},
        "recon_mask_drop_full": {"value": 0.05},
        "null_value": {"value": None},
        "pad_value": {"value": None},
    },
    # only for bayes sweeps
    'metric': {
        'goal': 'minimize',
        'name': 'val/smape'
    },
}

sweep_id=wandb.sweep(sweep_config, project="baselr_sweep")
# wandb.agent(sweep_id=sweep_id, function=train, count=5)
# wandb.agent(sweep_id=sweep_id, function=train)
wandb.agent(sweep_id=sweep_id, function=train, count=1)




