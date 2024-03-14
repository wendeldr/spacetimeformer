from argparse import ArgumentParser
from contextlib import contextmanager
import random
import sys
import warnings
import os
import uuid
import pandas as pd
import numpy as np
import mne

from scipy.stats import chi2, norm, beta, gamma
import math
import pytorch_lightning as pl
import torch

import spacetimeformer as stf


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout  # Save the original stdout
    try:
        sys.stdout = SuppressOutput()  # Suppress printing
        yield
    finally:
        sys.stdout = original_stdout  # Restore the original stdout

class SuppressOutput:
    def write(self, _):
        pass
    def flush(self):
        pass 

from connection_complexity.data.raw_data.EDF.edf_helpers import read_edf

# from pytorch_lightning import Trainer, seed_everything
# seed_everything(42, workers=True)

_MODELS = ["spacetimeformer", "mtgnn", "heuristic", "lstm", "lstnet", "linear", "s4"]

_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "exchange",
    "precip",
    "toy2",
    "solar_energy",
    "syn",
    "mnist",
    "cifar",
    "copy",
    "cont_copy",
    "m4",
    "wiki",
    "ettm1",
    "weather",
    "monash",
    "hangzhou",
    "traffic",
    "eeg",
    "clean_phaseshifted",
    "three_simple_waves",
    "contemporaneous_dep_S1",
    "time_dep_S2",
    "both_dep_S3",
    "EDF",
    "q_s1",
    "q_s1ar",
    "eeg_with_s1",
    "eeg_with_s2"
]


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    # Throw error now before we get confusing parser issues
    assert (
            model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset == "EDF":
        parser.add_argument("--data_path", type=str, default=None)
        parser.add_argument("--channels", type=str, default=None)

        parser.add_argument(
            "--context_points",
            type=int,
            default=128,
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default=32,
            help="number of future timesteps to predict",
        )
    elif (dset == "precip"):
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
    elif dset == "metr-la" or dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
    elif dset == "syn":
        stf.data.synthetic.SyntheticData.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset == "mnist":
        stf.data.image_completion.MNISTDset.add_cli(parser)
    elif dset == "cifar":
        stf.data.image_completion.CIFARDset.add_cli(parser)
    elif dset == "copy":
        stf.data.copy_task.CopyTaskDset.add_cli(parser)
    elif dset == "cont_copy":
        stf.data.cont_copy_task.ContCopyTaskDset.add_cli(parser)
    elif dset == "m4":
        stf.data.m4.M4TorchDset.add_cli(parser)
    elif dset == "wiki":
        stf.data.wiki.WikipediaTorchDset.add_cli(parser)
    elif dset == "monash":
        stf.data.monash.MonashDset.add_cli(parser)
    elif dset == "hangzhou":
        stf.data.metro.MetroData.add_cli(parser)
    else:
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    stf.data.DataModule.add_cli(parser)

    if model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif model == "heuristic":
        stf.heuristic_model.Heuristic_Forecaster.add_cli(parser)
    elif model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
    elif model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)
    elif model == "s4":
        stf.s4_model.S4_Forecaster.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_samples", type=int, default=8)
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--no_earlystopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--scaling_factor", type=float, default=1.0)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser


def create_model(config, x_dim=None, yc_dim=None, yt_dim=None):
    if config.dset == "metr-la":
        x_dim = 2
        yc_dim = 207
        yt_dim = 207
    elif config.dset == "pems-bay":
        x_dim = 2
        yc_dim = 325
        yt_dim = 325
    elif config.dset == "precip":
        x_dim = 2
        yc_dim = 49
        yt_dim = 49
    elif config.dset == "asos":
        x_dim = 6
        yc_dim = 6
        yt_dim = 6
    elif config.dset == "solar_energy":
        x_dim = 6
        yc_dim = 137
        yt_dim = 137
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "syn":
        x_dim = 5
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "mnist":
        x_dim = 1
        yc_dim = 28
        yt_dim = 28
    elif config.dset == "cifar":
        x_dim = 1
        yc_dim = 3
        yt_dim = 3
    elif config.dset == "copy" or config.dset == "cont_copy":
        x_dim = 1
        yc_dim = config.copy_vars
        yt_dim = config.copy_vars
    elif config.dset == "m4":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "wiki":
        x_dim = 2
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "monash":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "ettm1":
        x_dim = 4
        yc_dim = 7
        yt_dim = 7
    elif config.dset == "weather":
        x_dim = 3
        yc_dim = 21
        yt_dim = 21
    elif config.dset == "hangzhou":
        x_dim = 4
        yc_dim = 160
        yt_dim = 160
    elif config.dset == "traffic":
        x_dim = 2
        yc_dim = 862
        yt_dim = 862
    elif config.dset == "eeg":
        x_dim = 3
        yc_dim = 6
        yt_dim = 6

    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    elif config.model == "heuristic":
        forecaster = stf.heuristic_model.Heuristic_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            loss=config.loss,
            method=config.method,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_revin=config.use_revin,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
    elif config.model == "spacetimeformer":
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
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
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )
    elif config.model == "s4":
        forecaster = stf.s4_model.S4_Forecaster(
            context_points=config.context_points,
            target_points=config.target_points,
            d_state=config.d_state,
            d_model=config.d_model,
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            layers=config.layers,
            time_emb_dim=config.time_emb_dim,
            channels=config.channels,
            dropout_p=config.dropout_p,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    return forecaster


def create_dset(config, x_dim=None, yc_dim=None, yt_dim=None):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    if config.dset == "metr-la" or config.dset == "pems-bay":
        if config.dset == "pems-bay":
            assert (
                    "pems_bay" in config.data_path
            ), "Make sure to switch to the pems-bay file!"
        data = stf.data.metr_la.METR_LA_Data(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metr_la.METR_LA_Torch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "hangzhou":
        data = stf.data.metro.MetroData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metro.MetroTorch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "precip":
        dset = stf.data.precip.GeoDset(dset_dir=config.dset_dir, var="precip")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.precip.CONUS_Precip,
            dataset_kwargs={
                "dset": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
    elif config.dset == "syn":
        dset = stf.data.synthetic.SyntheticData(config.data_path)
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
    elif config.dset in ["mnist", "cifar"]:
        if config.dset == "mnist":
            config.target_points = 28 - config.context_points
            datasetCls = stf.data.image_completion.MNISTDset
            PLOT_VAR_IDXS = [18, 24]
            PLOT_VAR_NAMES = ["18th row", "24th row"]
        else:
            config.target_points = 32 * 32 - config.context_points
            datasetCls = stf.data.image_completion.CIFARDset
            PLOT_VAR_IDXS = [0]
            PLOT_VAR_NAMES = ["Reds"]
        DATA_MODULE = stf.data.DataModule(
            datasetCls=datasetCls,
            dataset_kwargs={"context_points": config.context_points},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.copy_task.CopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "mask_prob": config.copy_mask_prob,
                "include_lags": config.copy_include_lags,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "cont_copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.cont_copy_task.ContCopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "include_lags": config.copy_include_lags,
                "magnitude_matters": config.copy_mag_matters,
                "freq_shift": config.copy_freq_shift,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "m4":
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.m4.M4TorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "resolutions": args.resolutions,
                "max_len": args.max_len,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            collate_fn=stf.data.m4.pad_m4_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0

    elif config.dset == "wiki":
        DATA_MODULE = stf.data.DataModule(
            stf.data.wiki.WikipediaTorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "forecast_duration": args.forecast_duration,
                "max_len": args.max_len,
            },
            batch_size=args.batch_size,
            workers=args.workers,
            collate_fn=stf.data.wiki.pad_wiki_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0
        SCALER = stf.data.wiki.WikipediaTorchDset.scale
        INV_SCALER = stf.data.wiki.WikipediaTorchDset.inverse_scale
    elif config.dset == "monash":
        root_dir = config.root_dir
        DATA_MODULE = stf.data.monash.monash_dloader.make_monash_dmodule(
            root_dir=root_dir,
            max_len=config.max_len,
            include=config.include,
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=config.overfit,
        )
        NULL_VAL = -64.0
        PAD_VAL = -64.0
    elif config.dset == "ettm1":
        target_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split=4.0 / 20,  # from informer
            test_split=4.0 / 20,  # from informer
            time_col_name="date",
            time_features=["month", "day", "weekday", "hour"],
        )
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        # PAD_VAL = -32.0
        PLOT_VAR_NAMES = target_cols
        PLOT_VAR_IDXS = [i for i in range(len(target_cols))]
    elif config.dset == "weather":
        data_path = config.data_path
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=[],
            ignore_cols=[],
            # paper says 7:1:2 split
            val_split=1.0 / 10,
            test_split=2.0 / 10,
            time_col_name="date",
            time_features=["day", "hour", "minute"],
        )
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        PLOT_VAR_NAMES = ["OT", "p (mbar)", "raining (s)"]
        PLOT_VAR_IDXS = [20, 0, 15]

    elif config.dset == "eeg":
        dset = stf.data.CSVTimeSeries(
            data_path="/media/dan/Data/git/spacetimeformer/realdata.csv",
            val_split=0.2,
            test_split=0.2,
            time_col_name="datetime",
            time_features=["minute", "second", "millisecond"],
        )
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "clean_phaseshifted":
        # clean_phaseshifted s1 for debugging
        fs = 2048  # sampling rate (Hz)
        T = 10  # length of epochs (s)
        f = 100  # frequency of sinusoids (Hz)
        t = np.arange(0, T, 1 / fs)
        A = 1  # noise amplitude
        sigma = 0.1  # Gaussian noise variance

        data = []
        # phase diffs around unit circle
        phase_differences = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
        names = ['0 (0°)', 'π/4 (45°)', 'π/2 (90°)', '3π/4 (135°)', 'π (180°)', '5π/4 (225°)', '3π/2 (270°)',
                 '7π/4 (315°)']
        for i, ps in enumerate(phase_differences):
            # set random seed for reproducibility
            # np.random.seed(i)
            # sig = np.sin(2 * np.pi * f * t - ps) + A * np.random.normal(0, sigma, size=t.shape)
            sig = np.sin(2 * np.pi * f * t - ps)
            data.append(sig)

        data = np.array(data).T
        yc_dim = data.shape[1]
        yt_dim = data.shape[1]

        PLOT_VAR_NAMES = names
        PLOT_VAR_IDXS = np.arange(0, len(names))

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)

        # add datetime column
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
            time_features=["year", "day"],
        )
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

        x_dim = dset.time_cols.shape[0]

    elif config.dset == "three_simple_waves":
        # # wave 1 = flat
        # # wave 2 = linear increase
        # # wave 3 = exponential decrease
        # # time is same sampling for all waves
        # length = 10000
        # time = np.arange(0, length, 1)
        # wave1 = np.zeros(time.shape)
        # wave2 = time
        # wave3 = np.exp(-time)
        # data = np.vstack((wave1, wave2, wave3)).T
        # yc_dim = data.shape[1]
        # yt_dim = data.shape[1]
        # names = ['flat', 'linear', 'exp-decrease']
        fs = 2048  # Sampling rate (Hz)
        T = 10  # Length of epochs (s)
        f = 100  # Frequency of sinusoids (Hz)
        t = np.arange(0, T, 1 / fs)  # Time array
        A = 1  # Amplitude
        sigma = 0.1  # Gaussian noise variance

        # Damping/growth factor
        k = 2

        # Number of repetitions
        N = 10  # Replace with desired number of repetitions

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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "contemporaneous_dep_S1":

        # error_scale = config.scaling_factor

        # fs = 2048  # Sampling rate (Hz)
        # T = 150  # Length of epochs (s)

        # # Set the seed for reproducibility
        # np.random.seed(0)

        # # Define the number of iterations for the simulation
        # n_iterations = fs * T

        # # Preallocate the arrays for the x variables
        # x1 = np.zeros(n_iterations)
        # x2 = np.zeros(n_iterations)
        # x3 = np.zeros(n_iterations)
        # x4 = np.zeros(n_iterations)
        # x5 = np.zeros(n_iterations)

        # # Define the rate lambda for the exponential distribution
        # lambda_rate = 2

        # # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
        # e2 = chi2.rvs(df=1, size=n_iterations)
        # e3 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1
        # e4 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1
        # e5 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1

        # for t in range(1, n_iterations):
        #     # Generate the x variables based on the given equations
        #     x1[t] = e1[t]
        #     x2[t] = e2[t]
        #     x3[t] = 0.8 * x2[t] + e3[t]
        #     x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
        #     x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        # # After the loop, x1t, x2t, x3t, x4t, and x5t contain the simulation data.

        # PLOT_VAR_NAMES = np.arange(5) + 1
        # PLOT_VAR_IDXS = np.arange(5)

        # data = np.array([x1, x2, x3, x4, x5]).T

        # df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        # df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="ms")

        # dset = stf.data.CSVTimeSeries(
        #     data_path=None,
        #     raw_df=df,
        #     val_split=0.1,
        #     test_split=0.1,
        #     normalize=True,
        #     time_col_name="Datetime",
        #     time_features=["minute", 'second', 'millisecond'],
        # )
        # yc_dim = data.shape[1]
        # yt_dim = data.shape[1]
        # x_dim = dset.time_cols.shape[0]

        # DATA_MODULE = stf.data.DataModule(
        #     datasetCls=stf.data.CSVTorchDset,
        #     dataset_kwargs={
        #         "csv_time_series": dset,
        #         "context_points": config.context_points,
        #         "target_points": config.target_points,
        #         "time_resolution": config.time_resolution,
        #     },
        #     batch_size=config.batch_size,
        #     workers=config.workers,
        #     overfit=args.overfit,
        # )
        # INV_SCALER = dset.reverse_scaling
        # SCALER = dset.apply_scaling
        # NULL_VAL = None

        m=0
        fs = 256  # Sampling rate (Hz)
        T = 30  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
        # e2 = chi2.rvs(df=1, size=n_iterations)

        e1 = norm.rvs(scale=1, size=n_iterations)
        e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e4 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) *m# Gaussian with mean 0, std 1

        for t in range(6, n_iterations):
            # Generate the x variables based on the given equations
            x1[t] = 0.05 * x1[t-5] - 0.15 * x1[t-4] + 0.25 * x1[t-3] - 0.35 * x1[t-2] + 0.45 * x1[t-1] + e1[t]
            # x1[t] = e1[t]

            x2[t] = - 0.25 * x2[t-2] + .5 * x2[t-1] + e2[t]
            x3[t] = 0.8 * x2[t] + e3[t]
            x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
            x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        # After the loop, x1t, x2t, x3t, x4t, and x5t contain the simulation data.

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T
        # cut off first few values so system is in steady state
        data = data[100:]

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="s")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["hour", 'minute', 'second'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "eeg_with_s2":
        # generate N scalar values 
        def generate_scalars(x=20):
            # Generate x-1 random numbers, values between -1 and 1
            values = np.random.uniform(-1, 1, x - 1)
            sum_values = np.sum(values)
            # Subtract this sum from 1 to get the x+1 value
            last_value = 1 - sum_values
            # Append the last value to the list
            values = np.append(values, last_value)
            # Shuffle the array to not have the balancing value at the end
            np.random.shuffle(values)
            return values


        m  = 3
        fs = 512  # Sampling rate (Hz)
        T  = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        with suppress_stdout():
            eeg1 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/FC_OvertNaming.EDF", preload=True)
            eeg2 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/PC_OvertNaming.EDF", preload=True)

            sf1 = eeg1.info['sfreq']
            sf2 = eeg2.info['sfreq']

            # resample to 512 Hz
            eeg1 = eeg1.resample(fs)
            eeg2 = eeg2.resample(fs)


            # filter 60 Hz noise and harmonics with zerophase notch filter
            eeg1 = eeg1.notch_filter(np.arange(60, fs//2, 60), fir_design='firwin',verbose=False).get_data(picks=eeg1.info['ch_names'][0]).squeeze()
            eeg2 = eeg2.notch_filter(np.arange(60, fs//2, 60), fir_design='firwin',verbose=False).get_data(picks=eeg2.info['ch_names'][1]).squeeze()

            # # z normalize data
            eeg1 = (eeg1 - np.mean(eeg1)) / np.std(eeg1)
            eeg2 = (eeg2 - np.mean(eeg2)) / np.std(eeg2)

            # quantize data
            eeg1 = np.round(eeg1, m)
            eeg2 = np.round(eeg2, m)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = eeg1[:n_iterations]
        x2 = eeg2[:n_iterations]
        # print(x1.shape)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = norm.rvs(scale=1, size=n_iterations)
        # e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) * 0.01 
        e3 = np.round(e3, m)
        e4 = norm.rvs(scale=1, size=n_iterations) * 0 # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) * 0# Gaussian with mean 0, std 1

        poly = generate_scalars(15)
        start = 21
        for i, t in enumerate(range(start, n_iterations)):
            # Generate the x variables based on the given equations
            x4[t] = 0.7 * np.sin(x1[t-20]) * (math.pow(x1[t-2], 2) - 1) + e4[t]

            for q, p in enumerate(poly):
                x3[t] += p * x1[t-(q+1)]
            x5[t] = 0.3 * x2[t-7] + 0.5 * math.pow(x2[t-1], 2) + 0.05* np.tan(x2[t-20]) + e5[t]

        x3 = np.round(x3, m)
        x4 = np.round(x4, m)
        x5 = np.round(x5, m)

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T
        data = data[start:,:]

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="us")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["minute", 'second', 'microsecond'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "eeg_with_s1":
        m  = 3
        fs = 2048  # Sampling rate (Hz)
        T  = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        with suppress_stdout():
            eeg1 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/FC_OvertNaming.EDF", preload=True)
            eeg2 = mne.io.read_raw_edf("/home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/PC_OvertNaming.EDF", preload=True)

            sf1 = eeg1.info['sfreq']
            sf2 = eeg2.info['sfreq']

            # filter 60 Hz noise and harmonics with zerophase notch filter
            eeg1 = eeg1.notch_filter(np.arange(60, sf1//2, 60), fir_design='firwin',verbose=False).get_data(picks=eeg1.info['ch_names'][0]).squeeze()
            eeg2 = eeg2.notch_filter(np.arange(60, sf2//2, 60), fir_design='firwin',verbose=False).get_data(picks=eeg2.info['ch_names'][1]).squeeze()

            # # z normalize data
            eeg1 = (eeg1 - np.mean(eeg1)) / np.std(eeg1)
            eeg2 = (eeg2 - np.mean(eeg2)) / np.std(eeg2)

            # quantize data
            eeg1 = np.round(eeg1, m)
            eeg2 = np.round(eeg2, m)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = eeg1[:n_iterations]
        x2 = eeg2[:n_iterations]
        # print(x1.shape)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = norm.rvs(scale=1, size=n_iterations)
        # e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) * 0.01 
        e3 = np.round(e3, m)
        e4 = norm.rvs(scale=1, size=n_iterations) * 0 # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) * 0# Gaussian with mean 0, std 1

        for t in range(6, n_iterations):
            # Generate the x variables based on the given equations
            x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]

            x3[t] = 0.8 * x2[t] + e3[t]
            x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        x3 = np.round(x3, m)
        x4 = np.round(x4, m)
        x5 = np.round(x5, m)

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="us")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["minute", 'second', 'microsecond'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "q_s1":
        m=0
        sigfig = 2
        fs = 2048  # Sampling rate (Hz)
        T = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        e1 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
        e2 = chi2.rvs(df=1, size=n_iterations)
        # e1 = norm.rvs(scale=1, size=n_iterations)
        # e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e4 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) *m# Gaussian with mean 0, std 1

        e1 = np.round(e1, sigfig)
        e2 = np.round(e2, sigfig)
        e3 = np.round(e3, sigfig)
        e4 = np.round(e4, sigfig)
        e5 = np.round(e5, sigfig)

        for t in range(6, n_iterations):
            x1[t] = e1[t]
            x2[t] = e2[t]
            x3[t] = 0.8 * x2[t] + e3[t]
            x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
            x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        x3 = np.round(x3, sigfig)
        x4 = np.round(x4, sigfig)
        x5 = np.round(x5, sigfig)

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T
        data= data[100:]
        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="s")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["hour", 'minute', 'second'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "q_s1ar":
        m=0
        sigfig = 2
        fs = 2048  # Sampling rate (Hz)
        T = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
        # e2 = chi2.rvs(df=1, size=n_iterations)
        e1 = norm.rvs(scale=1, size=n_iterations)
        e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e4 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) *m# Gaussian with mean 0, std 1

        e1 = np.round(e1, sigfig)
        e2 = np.round(e2, sigfig)
        e3 = np.round(e3, sigfig)
        e4 = np.round(e4, sigfig)
        e5 = np.round(e5, sigfig)

        for t in range(6, n_iterations):
            x1[t] = 0.05 * x1[t-5] - 0.15 * x1[t-4] + 0.25 * x1[t-3] - 0.35 * x1[t-2] + 0.45 * x1[t-1] + e1[t]
            x2[t] = - 0.01 * x2[t-6] + .09 * x2[t-5] - 0.2 * x2[t-4] + .27 * x2[t-3] - 0.35 * x2[t-2] + .4 * x2[t-1] + e2[t]
            x3[t] = 0.8 * x2[t] + e3[t]
            x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
            x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        x3 = np.round(x3, sigfig)
        x4 = np.round(x4, sigfig)
        x5 = np.round(x5, sigfig)

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T
        data= data[100:]
        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="s")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["hour", 'minute', 'second'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "q_s2ar":
        m=0
        sigfig = 2
        fs = 2048  # Sampling rate (Hz)
        T = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        np.random.seed(42)

        # Define the number of iterations for the simulation
        n_iterations = fs * T
        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        # e1 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
        # e2 = chi2.rvs(df=1, size=n_iterations)
        e1 = norm.rvs(scale=1, size=n_iterations)
        e2 = norm.rvs(scale=1, size=n_iterations)
        e3 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e4 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
        e5 = norm.rvs(scale=1, size=n_iterations) *m# Gaussian with mean 0, std 1

        e1 = np.round(e1, sigfig)
        e2 = np.round(e2, sigfig)
        e3 = np.round(e3, sigfig)
        e4 = np.round(e4, sigfig)
        e5 = np.round(e5, sigfig)

        for t in range(6, n_iterations):
            x1[t] = 0.05 * x1[t-5] - 0.15 * x1[t-4] + 0.25 * x1[t-3] - 0.35 * x1[t-2] + 0.45 * x1[t-1] + e1[t]
            x2[t] = - 0.01 * x2[t-6] + .09 * x2[t-5] - 0.2 * x2[t-4] + .27 * x2[t-3] - 0.35 * x2[t-2] + .4 * x2[t-1] + e2[t]
            x3[t] = 0.8 * x2[t] + e3[t]
            x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
            x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

        x3 = np.round(x3, sigfig)
        x4 = np.round(x4, sigfig)
        x5 = np.round(x5, sigfig)

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T
        data= data[100:]
        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="s")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            time_col_name="Datetime",
            time_features=["hour", 'minute', 'second'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "time_dep_S2":

        error_scale = config.scaling_factor

        fs = 2048  # Sampling rate (Hz)
        T = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        # np.random.seed(0)

        # Define the number of iterations for the simulation
        n_iterations = fs * T

        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        e1 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1
        e2 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations) * error_scale
        e3 = beta.rvs(a=1, b=2, size=n_iterations) * error_scale
        e4 = beta.rvs(a=2, b=1, size=n_iterations) * error_scale
        e5 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1

        for t in range(3, n_iterations):
            # Generate the x variables based on the given equations
            x1[t] = 0.7 * x1[t - 1] + e1[t]
            x2[t] = 0.3 * np.power(x1[t - 2], 2) + e2[t]
            x3[t] = 0.4 * x1[t-3] - 0.3 * x3[t-2] + e3[t]
            x4[t] = 0.7 * x4[t-1] - 0.3 * x5[t-1] * np.exp((-math.pow(x5[t-1], 2)) / 2) + e4[t]
            x5[t] = 0.5 * x4[t-1] + 0.2 * x5[t-2] + e5[t]

        # After the loop, x1t, x2t, x3t, x4t, and x5t contain the simulation data.

        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="ms")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.2,
            test_split=0.2,
            normalize=True,
            time_col_name="Datetime",
            time_features=["minute", 'second', 'millisecond'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "both_dep_S3":
        error_scale = config.scaling_factor


        fs = 2048  # Sampling rate (Hz)
        T = 150  # Length of epochs (s)

        # Set the seed for reproducibility
        # np.random.seed(0)

        # Define the number of iterations for the simulation
        n_iterations = fs * T

        # Preallocate the arrays for the x variables
        x1 = np.zeros(n_iterations)
        x2 = np.zeros(n_iterations)
        x3 = np.zeros(n_iterations)
        x4 = np.zeros(n_iterations)
        x5 = np.zeros(n_iterations)

        # Define the rate lambda for the exponential distribution
        lambda_rate = 2

        # Generate the noise processes e1t, e2t, e3t, e4t, e5t
        e1 = norm.rvs(scale=1, size=n_iterations) * error_scale # Gaussian with mean 0, std 1
        e2 = beta.rvs(a=1, b=2, size=n_iterations) * error_scale
        e3 = beta.rvs(a=1, b=2, size=n_iterations) * error_scale
        e4 = norm.rvs(scale=1, size=n_iterations) * error_scale
        e5 = gamma.rvs(a=16, rate=1/0.25, size=n_iterations) * error_scale

        for t in range(3, n_iterations):
            # Generate the x variables based on the given equations
            x1[t] = 0.6 * x1[t - 2] + e1[t]
            x2[t] = x1[t] + 0.3 * x2[t-1] + e2[t]
            x3[t] = 0.3 + x3[t-1] + np.sin(x2[t-1]) + e3[t]
            x4[t] = 0.4 * x3[t-2] + e4[t]
            x5[t] = -3.2 * 0.5 * math.pow(x3[t-1], 2) + e5[t]

        # After the loop, x1t, x2t, x3t, x4t, and x5t contain the simulation data.
        PLOT_VAR_NAMES = np.arange(5) + 1
        PLOT_VAR_IDXS = np.arange(5)

        data = np.array([x1, x2, x3, x4, x5]).T

        df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
        df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="ms")

        dset = stf.data.CSVTimeSeries(
            data_path=None,
            raw_df=df,
            val_split=0.2,
            test_split=0.2,
            normalize=True,
            time_col_name="Datetime",
            time_features=["minute", 'second', 'millisecond'],
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
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "EDF":
        edf = read_edf(args.data_path, preload=True)
        df = edf.to_data_frame(picks=args.channels, time_format='datetime')

        pd.set_option('display.max_columns', 500)
        # print data length and NAN count:
        print(df.describe())

        PLOT_VAR_NAMES = args.channels
        PLOT_VAR_IDXS = np.arange(len(PLOT_VAR_NAMES))
        dset = stf.data.CSVTimeSeries(
            raw_df=df,
            val_split=0.2,
            test_split=0.2,
            time_col_name="time",
            normalize=True,
            time_features=["second", "millisecond", "microsecond"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": 1,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    else:
        time_col_name = "Datetime"
        data_path = config.data_path
        time_features = ["year", "month", "day", "weekday", "hour", "minute"]
        if config.dset == "asos":
            if data_path == "auto":
                data_path = "./data/temperature-v1.csv"
            target_cols = ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"]
        elif config.dset == "solar_energy":
            if data_path == "auto":
                data_path = "./data/solar_AL_converted.csv"
            target_cols = [str(i) for i in range(137)]
        elif "toy" in config.dset:
            if data_path == "auto":
                if config.dset == "toy2":
                    data_path = "./data/toy_dset2.csv"
                else:
                    raise ValueError(f"Unrecognized toy dataset {config.dset}")
            target_cols = [f"D{i}" for i in range(1, 21)]
        elif config.dset == "exchange":
            if data_path == "auto":
                data_path = "./data/exchange_rate_converted.csv"
            target_cols = [
                "Australia",
                "United Kingdom",
                "Canada",
                "Switzerland",
                "China",
                "Japan",
                "New Zealand",
                "Singapore",
            ]
        elif config.dset == "traffic":
            if data_path == "auto":
                data_path = "./data/traffic.csv"
            target_cols = [f"Lane {i}" for i in range(862)]
            time_col_name = "FakeTime"
            time_features = ["month", "day"]

        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols="all",
            time_col_name=time_col_name,
            time_features=time_features,
            val_split=0.2,
            test_split=0.2,
        )
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
            overfit=args.overfit,
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


def create_callbacks(config, save_dir):
    filename = f"{config.run_name}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(save_dir, filename)
    config.model_ckpt_dir = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]

    if not config.no_earlystopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=config.patience,
            )
        )

    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.time_mask_end,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


# noinspection PyTypeChecker
def main(args):
    # read edf file. Should probably make a class like the other datasets but this is a quick fix
    if args.dset == 'EDF':
        # check data_path argument exists
        assert args.data_path is not None or args.data_path == '', "Please provide a path to the EDF file using the --data_path argument"

        # check that edf file exists
        assert os.path.exists(args.data_path), f"EDF file '{args.data_path}' does not exist!"

        edf = read_edf(args.data_path, )

        # check if channels exist. If not use all channels.
        if args.channels is None or args.channels == '':
            args.channels = edf.ch_names
            potential_list = args.channels

        else:
            # strip [,] from string
            potential_list = args.channels.strip('[]')

            # check if channels are valid
            # check if list of channels or single channel provided.
            # if single channel, convert to list
            # list will be either python list format or just csv string
            if ',' in potential_list:
                potential_list = potential_list.split(',')
            else:
                potential_list = [potential_list]

            not_overlapping = list(set(potential_list).difference(edf.ch_names))
            valid_string = ",".join(edf.ch_names)

            assert len(
                not_overlapping) == 0, f"Invalid channel name(s) provided.\nValid channels are: {valid_string}\nYou provided these which don't overlap: {not_overlapping}"
            args.channels = potential_list
        print(f"Processing channels from {args.data_path}")
        print(f"channels to use: {args.channels}")
        # check for data shape
        d = edf.get_data(picks=args.channels)
        print(d.shape)

    log_dir = os.getenv("STF_LOG_DIR")
    if log_dir is None:
        log_dir = "./data/STF_LOG_DIR"
        print(
            "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
        )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.wandb:
        import wandb

        if args.dset == "":
            project = "default"
        else:
            project = args.dset
        entity = "ponderingparameters"
        assert (
                project is not None and entity is not None
        ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
            your wandb user/organization name and project title, respectively."
        experiment = wandb.init(
            project=project,
            entity=entity,
            config=args,
            dir=log_dir,
            reinit=True,
        )
        config = wandb.config
        wandb.run.name = args.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(
            experiment=experiment,
            save_dir=log_dir,
        )

    if args.dset == 'EDF':
        x_dim = 3
        yc_dim = len(args.channels)
        yt_dim = len(args.channels)
    else:
        x_dim = None
        yc_dim = None
        yt_dim = None

    # Dset
    (
        data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
        pad_val,
        x_dim,
        yc_dim,
        yt_dim
    ) = create_dset(args, x_dim=x_dim, yc_dim=yc_dim, yt_dim=yt_dim)

    # Model
    args.null_value = null_val
    args.pad_value = pad_val
    forecaster = create_model(args, x_dim=x_dim, yc_dim=yc_dim, yt_dim=yt_dim)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    # Callbacks
    callbacks = create_callbacks(args, save_dir=log_dir)
    test_samples = next(iter(data_module.test_dataloader()))

    if args.wandb and args.plot:
        callbacks.append(
            stf.plot.PredictionPlotterCallback(
                test_samples,
                var_idxs=plot_var_idxs,
                var_names=plot_var_names,
                pad_val=pad_val,
                total_samples=min(args.plot_samples, args.batch_size),
            )
        )

    if args.wandb and args.dset in ["mnist", "cifar"] and args.plot:
        callbacks.append(
            stf.plot.ImageCompletionCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
                mode="left-right" if config.dset == "mnist" else "flat",
            )
        )

    if args.wandb and args.dset == "copy" and args.plot:
        callbacks.append(
            stf.plot.CopyTaskCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
            )
        )

    if args.wandb and args.model == "spacetimeformer" and args.attn_plot:
        callbacks.append(
            stf.plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(1, args.batch_size),
            )
        )

    if args.wandb:
        config.update(args)
        logger.log_hyperparams(config)

    if args.model == "spacetimeformer" and (args.attn_plot or args.plot) and not args.wandb:
        import wandb
        experiment = wandb.init(
            config=args,
            dir=log_dir,
            reinit=True,
            mode="offline",
        )
        config = wandb.config
        wandb.run.name = args.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(
            experiment=experiment,
            save_dir=log_dir,
        )
        if args.attn_plot:
            callbacks.append(
                stf.plot.AttentionMatrixCallback_SANSWANDB(
                    test_samples,
                    layer=0,
                    total_samples=min(args.plot_samples, args.batch_size),
                    context_points=args.context_points,
                    y_dim=forecaster.d_yc,
                    col_divider_width=1,
                    row_divider_width=1,
                    random_sample_output=False,
                )
            )
        if args.plot:
            callbacks.append(
                stf.plot.PredictionPlotterCallback(
                    test_samples,
                    var_idxs=plot_var_idxs,
                    var_names=plot_var_names,
                    pad_val=pad_val,
                    total_samples=min(args.plot_samples, args.batch_size),
                    log_to_wandb=False,
                    log_to_file=True,
                    start_idx=data_module.dataset_kwargs['csv_time_series'].test_idx_start,
                    random_sample_output=False,
                )
            )

    if args.val_check_interval <= 1.0:
        val_control = {"val_check_interval": args.val_check_interval}
    else:
        val_control = {"check_val_every_n_epoch": int(args.val_check_interval)}

    # init logger
    if args.wandb:
        touse = logger
    elif args.model == "spacetimeformer" and args.attn_plot:
        touse = logger
    else:
        touse = None

    trainer = pl.Trainer(
        gpus=args.gpus,
        callbacks=callbacks,
        logger=touse,
        accelerator="dp",
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=False,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        **val_control,
        # deterministic=True
    )

    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module, ckpt_path="best")

    # Predict (only here as a demo and test)
    # forecaster.to("cuda")
    # xc, yc, xt, _ = test_samples
    # yt_pred = forecaster.predict(xc, yc, xt)

    if args.wandb:
        experiment.finish()


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
