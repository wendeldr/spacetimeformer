
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

def s1_data(config, m=1):
    fs = 2048  # Sampling rate (Hz)
    T = 150  # Length of epochs (s)

    # Set the seed for reproducibility
    np.random.seed(0)

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
    e3 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
    e4 = norm.rvs(scale=1, size=n_iterations) *m # Gaussian with mean 0, std 1
    e5 = norm.rvs(scale=1, size=n_iterations) *m# Gaussian with mean 0, std 1

    for t in range(0, n_iterations):
        # Generate the x variables based on the given equations
        x1[t] = e1[t]
        x2[t] = e2[t]
        x3[t] = 0.8 * x2[t] + e3[t]
        x4[t] = 0.7 * x1[t] * (math.pow(x1[t], 2) - 1) * np.exp((-math.pow(x1[t], 2)) / 2) + e4[t]
        x5[t] = 0.3 * x2[t] + 0.05 * math.pow(x2[t], 2) + e5[t]

    # After the loop, x1t, x2t, x3t, x4t, and x5t contain the simulation data.

    PLOT_VAR_NAMES = np.arange(5) + 1
    PLOT_VAR_IDXS = np.arange(5)

    data = np.array([x1, x2, x3, x4, x5]).T

    df = pd.DataFrame(data, columns=PLOT_VAR_NAMES)
    df["Datetime"] = pd.date_range(start="1/1/2020", periods=df.shape[0], freq="ms")

    dset = stf.data.CSVTimeSeries(
        data_path=None,
        raw_df=df,
        val_split=0.1,
        test_split=0.1,
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

def s2_data(config):
    fs = 2048  # Sampling rate (Hz)
    T = 150  # Length of epochs (s)

    # Set the seed for reproducibility
    np.random.seed(0)

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
    e1 = norm.rvs(scale=1, size=n_iterations)  # Gaussian with mean 0, std 1
    e2 = np.random.exponential(scale=1 / lambda_rate, size=n_iterations)
    e3 = beta.rvs(a=1, b=2, size=n_iterations)
    e4 = beta.rvs(a=2, b=1, size=n_iterations)
    e5 = norm.rvs(scale=1, size=n_iterations)  # Gaussian with mean 0, std 1

    for t in range(3, n_iterations):
        # Generate the x variables based on the given equations
        x1[t] = 0.7 * x1[t - 1] + e1[t]
        x2[t] = 0.3 * np.power(x1[t - 2], 2) + e2[t]
        x3[t] = 0.4 * x1[t - 3] - 0.3 * x3[t - 2] + e3[t]
        x4[t] = 0.7 * x4[t - 1] - 0.3 * x5[t - 1] * np.exp((-math.pow(x5[t - 1], 2)) / 2) + e4[t]
        x5[t] = 0.5 * x4[t - 1] + 0.2 * x5[t - 2] + e5[t]

    data = np.array([x1, x2, x3, x4, x5]).T
    PLOT_VAR_NAMES = np.arange(5) + 1
    PLOT_VAR_IDXS = np.arange(5)
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
            "context_points": config['context_points'],
            "target_points": config['target_points'],
            "time_resolution": config['time_resolution'],
        },
        batch_size=config['batch_size'],
        workers=config['workers'],
        overfit=False,
    )

    return DATA_MODULE

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



def get_sub_matrices(A, Q, indices):
    sub_matrices = []
    M = A.shape[0]  # Assuming A is a numpy array and square

    for idx in indices:
        i, j = idx
        if i*Q < M and j*Q < M:  # Check if indices are valid
            sub_matrix = A[i*Q:(i+1)*Q, j*Q:(j+1)*Q]
            sub_matrices.append(sub_matrix)
        else:
            sub_matrices.append(None)  # Or handle invalid index differently

    return sub_matrices

def insert_dividers(matrix, context_points, divider_width=1):
    total_rows, total_cols = matrix.shape

    # Insert row dividers
    insert_positions = np.arange(context_points, total_rows, context_points)
    insert_positions = np.repeat(insert_positions,
                                    divider_width)  # repeat the insert positions for each row divider
    matrix = np.insert(matrix, insert_positions, np.nan, axis=0)

    # Update total_rows and total_cols after row dividers insertion
    total_rows, total_cols = matrix.shape

    # Insert column dividers
    insert_positions = np.arange(context_points, total_cols, context_points)
    insert_positions = np.repeat(insert_positions,
                                    divider_width)  # repeat the insert positions for each column divider
    matrix = np.insert(matrix, insert_positions, np.nan, axis=1)

    return matrix

def calculate_tick_positions(matrix_size, context_points, divider_width):
    positions = np.arange(0, matrix_size, context_points + divider_width)
    adjusted_positions = positions + (context_points + divider_width) // 2 - divider_width
    return adjusted_positions  # Exclude the last position which is beyond the matrix

def reorder_attention_matrix(attention, current_order, target_order, context_points):
    """Reorder the attention matrix based on the target electrode order, considering sub-squares."""
    num_electrodes = len(current_order)
    reordered_attention = np.zeros_like(attention)

    # Create a mapping from electrode names to their indices
    current_indices = {electrode: i for i, electrode in enumerate(current_order)}
    target_indices = [current_indices[electrode] for electrode in target_order]

    # Reorder sub-squares in the attention matrix
    for i, target_idx in enumerate(target_indices):
        for j, target_jdx in enumerate(target_indices):
            source_i, source_j = int(target_idx * context_points), int(target_jdx * context_points)
            dest_i, dest_j = int(i * context_points), int(j * context_points)
            reordered_attention[dest_i:dest_i + context_points, dest_j:dest_j + context_points] =  attention[source_i:source_i + context_points, source_j:source_j + context_points]

    return reordered_attention

def combined_plot(attention_matrices, electrode_names, plot_titles, context_points, maxval=0.05, same_order=False, put_text=False, plot_size=(20, 10)):
    if same_order and len(attention_matrices) > 1 and len(electrode_names) > 1:
        attention_matrices[1] = reorder_attention_matrix(attention_matrices[1], electrode_names[1], electrode_names[0], context_points)
        electrode_names[1] = electrode_names[0]
        
    nrows = 1
    ncols = len(attention_matrices)
    fig, ax = plt.subplots(figsize=plot_size, dpi=100, nrows=nrows, ncols=ncols)
    
    # make sure ax is iterable
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for i, attention in enumerate(attention_matrices):
        attention_with_dividers = insert_dividers(attention, context_points)
        tick_positions = calculate_tick_positions(attention_with_dividers.shape[0], context_points, 1)

        ax[i].set_xticks(tick_positions)
        ax[i].set_yticks(tick_positions)
        ax[i].set_xticklabels(electrode_names[i], rotation=90)
        ax[i].set_yticklabels(electrode_names[i])
        ax[i].xaxis.tick_top()

        ax[i].imshow(attention_with_dividers, cmap='nipy_spectral', interpolation='none', vmin=0, vmax=maxval)

        ax[i].set_title(plot_titles[i])

        if put_text:
            # put text of values into each cell
            for x in range(attention_with_dividers.shape[0]):
                for y in range(attention_with_dividers.shape[1]):
                    if not np.isnan(attention_with_dividers[x, y]):
                        ax[i].text(y, x, '{:.2f}'.format(attention_with_dividers[x, y]),
                                horizontalalignment='center',
                                verticalalignment='center',
                                color='white', fontsize=6)

    # Adjust layout for colorbar
    plt.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])  # Adjust as needed
    fig.colorbar(ax[0].images[0], cax=cbar_ax, orientation='horizontal')
    ax[0].set_ylabel('Electrode')

    return fig, ax
    # return attention_matrices[0], attention_matrices[1]

def blow_up(matrix, N):
    # Repeat each element in each row N times
    repeated_rows = np.repeat(matrix, N, axis=1)
    # Repeat each row N times
    blown_up_matrix = np.repeat(repeated_rows, N, axis=0)
    return blown_up_matrix



# scan folder and subfolders for .ckpt files
from collections import defaultdict
import re


base_path = "~/git/spacetimeformer/spacetimeformer/data/STF_LOG_DIR/"
base_path = os.path.expanduser(base_path)
ckpt_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".ckpt"):
            ckpt_files.append(os.path.join(root, file))

# drop "EDF_" containing checkpoints
ckpt_files = [f for f in ckpt_files if "EDF_" not in f]
# drop "S2_" containing checkpoints
ckpt_files = [f for f in ckpt_files if "/S2_" not in f]

# sort into groups. check the first 2 characters of the file name
# group 1 begins with "G_"
# group 2 begins with "E_"
# group 3 begins with "L_"
# group 4 is the rest
group1 = [f for f in ckpt_files if os.path.basename(f).split("_")[0] == "G"]
group2 = [f for f in ckpt_files if os.path.basename(f).split("_")[0] == "E"]
group3 = [f for f in ckpt_files if os.path.basename(f).split("_")[0] == "L"]
group4 = [f for f in ckpt_files if f not in group1 and f not in group2 and f not in group3]
# dictionary to hold the groups
groups = {"G": group1, "E": group2, "L": group3, "R": group4}


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

subgroups = subgroup_files(groups)



data_module = s1_data(default_config)
model = create_model(default_config, 3, 5, 5)
model.set_inv_scaler(data_module.dataset_kwargs['csv_time_series'].reverse_scaling)
model.set_scaler(data_module.dataset_kwargs['csv_time_series'].apply_scaling)
model.set_null_value(None)


# for each subgroup and m-value, load the model and calculate the attention matrix. return the matrix 
def run(path,model):
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
    return attn, mse


test1 = run(subgroups['R']['1'][0],model)
test2 = run(subgroups['R']['1'][1],model)


path1 = "/home/wendeldr/git/spacetimeformer/S1~self_attn_0_0.npy"
attention1 = np.load(path1)
attention1 = attention1[:, 0, :, :].mean(axis=0)

path2 = "/home/wendeldr/git/spacetimeformer/S1_1~self_attn_0_0.npy"
attention2 = np.load(path2)
attention2 = attention2[:, 0, :, :].mean(axis=0)

titles = ["S1", "S1_1"]
names = ["1,2,3,4,5".split(',')]*2
combined_plot([attention1, attention2], names, titles, 32, same_order=True, plot_size=(10,5));


X = 0
Y = 3
a = get_sub_matrices(attention1, 256, [(X,Y)])[0]
plt.figure(figsize=(10,10))
plt.imshow(a, cmap='nipy_spectral', interpolation='none', vmin=0, vmax=0.006)
plt.xlabel(names[0][X])
plt.ylabel(names[0][Y])





