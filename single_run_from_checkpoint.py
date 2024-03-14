import numpy as np
import matplotlib.pyplot as plt

import spacetimeformer as stf
import pytorch_lightning as pl
import numpy as np
from scipy.stats import chi2, norm, beta, gamma
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

import sys
from contextlib import contextmanager


from collections import defaultdict
import re
import pickle
from tqdm.notebook import tqdm

import mne

import warnings
warnings.filterwarnings("ignore")


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


def gen_data(config, m=0, return_data=False):
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
    return DATA_MODULE


default_config = {
    # fixed but maybe change...
    'batch_size': 2000,
    'workers': 6,
    'init_lr': 1e-10,
    'base_lr': 0.0005,
    'context_points': 32,
    'target_points': 1,
    'd_model': 100,
    'd_qk': 100,
    'd_v': 100,
    'd_ff': 400,
    'n_heads': 1,
    'enc_layers': 1,
    'dec_layers': 1,
    'global_self_attn': 'full',
    'local_self_attn': 'full',
    'global_cross_attn': 'full',
    'local_cross_attn': 'full',
    'no_val': False,
    'no_time': False,
    'no_space': False,
    'no_given': False,

    # directly set parameters
    'gpus': [0],
    # 'gpus':None,
    'strategy': 'dp',
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
    "grad_clip_norm": 0.0,
    "accumulate": 1,
    "limit_val_batches": 1.0,
    "max_epochs": 10,
    "val_check_interval": 1.0,
}

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
        verbose=False,
    )
    return forecaster

trainer = pl.Trainer(
        gpus=default_config['gpus'],
        callbacks=[],

        accelerator="dp",
        gradient_clip_val=default_config['grad_clip_norm'],
        gradient_clip_algorithm="norm",
        overfit_batches= 0,
        accumulate_grad_batches=default_config['accumulate'],
        sync_batchnorm=False,
        limit_val_batches=default_config['limit_val_batches'],
        max_epochs=default_config['max_epochs'],
        log_every_n_steps=1,
        val_check_interval = default_config['val_check_interval'],
    )


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

# subgroups
# each filename in a group has either _m*epoch where * is a number or no _m* just _epoch (in this case the number is 1)
# subgroup based on the * number
def subgroup_files(groups):
    subgroups = {}
    m_value_pattern = re.compile(r'_m([\d.]+)')  # Pattern to find the m-value

    for group, filenames in groups.items():
        if group not in subgroups:
            subgroups[group] = {}
        for filename in filenames:
            # Extract m-value; default to 1 if not found
            match = m_value_pattern.search(filename)
            m_value = match.group(1) if match else '1'
            if m_value not in subgroups[group]:
                subgroups[group][m_value] = []
            subgroups[group][m_value].append(filename)
            
    return subgroups

# load run function, returns attn and mse if path is valid else returns None
def load_run(output_path):
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            attn, mse = pickle.load(f)
        return attn, mse
    else:
        return None

# for each subgroup and m-value, load the model and calculate the attention matrix. return the matrix 
def run(path,model,output_basepath):

    # extract m-value from the path
    m_value_pattern = re.compile(r'_m([\d.]+)')  # Pattern to find the m-value
    match = m_value_pattern.search(path)
    m_value = match.group(1) if match else '1'

    data_module = gen_data(default_config, m=float(m_value))

    with suppress_stdout():
        s1 = model.load_from_checkpoint(path)
    s1.to("cuda")

    # get the attention matrixs for a batch of data
    test_samples = next(iter(data_module.test_dataloader()))
    xc, yc, xt, y_true = test_samples
    yt_pred, attn = s1.predict(xc, yc, xt, scale_input=False, scale_output=False, output_attention_mat=True)
    attn = attn[0].detach().cpu().numpy()

    # get error
    yt_pred = yt_pred.detach().cpu().numpy()
    yt_pred = yt_pred.reshape(-1, 5)
    truth = y_true.detach().cpu().numpy()

    mse = stf.eval_stats.mse(truth, yt_pred)

    # save the attention matrix and the mse as a pickle file

    return attn, mse



basepath = "/home/wendeldr/git/spacetimeformer/spacetimeformer/data/STF_LOG_DIR/"


sub_directories = ['d_9_m3_20d4f4be']

# sub_directories = ["b_0_m3_08cfaf20",
#                    "b_1_m3_08b0635e"]

# find checkpoint files in each subdirectory
ckpt_files = []
for sub_dir in sub_directories:
    for root, dirs, files in os.walk(os.path.join(basepath, sub_dir)):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, file))


# sort into groups. check the first 2 characters of the file name
# all groups start with a letter followed by an underscore; otherwise they are in group 'R' for remaining

# get unique first characters:
first_chars = list(set([os.path.basename(f).split('_')[0] for f in ckpt_files]))
# disregard any chacters that are more then 1 character long:
first_chars = [f for f in first_chars if len(f) == 1]
groups = {}
for f in first_chars:
    files = [file for file in ckpt_files if os.path.basename(file).split('_')[0] == f]
    groups.update({f: files})
remaining = [file for file in ckpt_files if os.path.basename(file).split('_')[0] not in first_chars]
# dictionary to hold the groups
groups.update({"Remain": remaining})

subgroups = subgroup_files(groups)


data_module = gen_data(default_config)
model = create_model(default_config, 3, 5, 5)
model.set_inv_scaler(data_module.dataset_kwargs['csv_time_series'].reverse_scaling)
model.set_scaler(data_module.dataset_kwargs['csv_time_series'].apply_scaling)
model.set_null_value(None)


output_basepath = "/home/wendeldr/dump"


attn_mse = {}
# Loop through each group and its subgroups with tqdm for progress tracking
for group, filenames in tqdm(subgroups.items(), desc='Groups'):
    for m_value, files in tqdm(filenames.items(), desc=f'Subgroups in {group}'):
        for file in tqdm(files, desc=f'Files in m={m_value} of {group}'):
            try:
                attn, mse = run(file, model, output_basepath)
                attn_mse[file] = (attn, mse)
            except Exception as e:
                print(e)
                print(f"Error in {file}")
                continue
