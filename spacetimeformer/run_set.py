import subprocess
from concurrent.futures import ThreadPoolExecutor
import os

dry_run = True

# Default parameters for the command
default_params = {
    'context_points': 32,
    'target_points': 1,
    'n_heads': 1,
    'enc_layers': 1,
    'dec_layers': 1,
    'd_model': 100,
    'd_qk': 100,
    'd_v': 100,
    'd_ff': 400,
    'global_self_attn': 'full',
    'local_self_attn': 'full',
    'global_cross_attn': 'full',
    'local_cross_attn': 'full',
    'batch_size': 2000,
    'attn_plot': False,
    'plot': False,
    # 'plot_samples': 1,
    'max_epochs': 20,
    'no_earlystopping': True,
    'wandb': True,
}


def run_command(dataset, run_name, gpu, custom_params):
    # Merge default and custom parameters
    params = default_params.copy()
    params.update(custom_params)
    params['gpus'] = gpu  # Set GPU

    # Construct the command
    cmd = ['python', 'train.py', 'spacetimeformer', dataset]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    cmd.extend(['--run_name', run_name])

    # Print the command for a dry run
    if dry_run:
        print(f"DRY RUN: Would execute on GPU {gpu}:\n=======\n{' '.join(cmd)}\n-------")
        return  # Skip actual execution

    # Execute the command with error handling
    try:
        result = subprocess.run(cmd, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Check for errors
        if result.returncode != 0:
            print(f"ERROR: Command {' '.join(cmd)} failed with error:\n{result.stderr.decode('utf-8')}")
        else:
            print(f"SUCCESS: Command {' '.join(cmd)} executed successfully.")
    except Exception as e:
        print(f"EXCEPTION: Failed to run command {' '.join(cmd)} due to {e}")


# Function to execute all commands for a specific GPU
def run_commands_for_gpu(commands, gpu):
    for dataset, run_name, custom_params in commands:
        run_command(dataset, run_name, gpu, custom_params)


# Main execution logic
if __name__ == "__main__":
    # List of tuples containing (run_name, gpu_id, custom_parameters)
    s1 = 'contemporaneous_dep_S1'
    # commands = [
    #     (s1, 'E_S1_m0', 0, {'scaling_factor': 0, 'max_epochs': 50}),
    #     (s1, 'E_S1_m0.0001', 0, {'scaling_factor': 0.0001, 'max_epochs': 50}),
    #     (s1, 'E_S1_1_m0.0001', 2, {'scaling_factor': 0.1, 'max_epochs': 50}),
    #     (s1, 'E_S1_m1', 2, {'scaling_factor': 1, 'max_epochs': 50}),
    #     (s1, 'E_S1_1_m0', 1, {'scaling_factor': 0, 'max_epochs': 50}),
    #     (s1, 'E_S1_1_m0.0001', 1, {'scaling_factor': 0.0001, 'max_epochs': 50}),
    #     (s1, 'E_S1_1_m0.1', 3, {'scaling_factor': 0.1, 'max_epochs': 50}),
    #     (s1, 'E_S1_1_m1', 3, {'scaling_factor': 1, 'max_epochs': 50}),

    #     (s1, 'G_S1_m0', 0, {'scaling_factor': 0, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_m0.0001', 0, {'scaling_factor': 0.0001, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_1_m0.0001', 2, {'scaling_factor': 0.1, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_m1', 2, {'scaling_factor': 1, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_1_m0', 1, {'scaling_factor': 0, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_1_m0.0001', 1, {'scaling_factor': 0.0001, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_1_m0.1', 3, {'scaling_factor': 0.1, 'local_self_attn': 'none'}),
    #     (s1, 'G_S1_1_m1', 3, {'scaling_factor': 1, 'local_self_attn': 'none'}),

    #     (s1, 'L_S1_m0', 0, {'scaling_factor': 0, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_m0.0001', 0, {'scaling_factor': 0.0001, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_1_m0.0001', 2, {'scaling_factor': 0.1, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_m1', 2, {'scaling_factor': 1, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_1_m0', 1, {'scaling_factor': 0, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_1_m0.0001', 1, {'scaling_factor': 0.0001, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_1_m0.1', 3, {'scaling_factor': 0.1, 'global_self_attn': 'none'}),
    #     (s1, 'L_S1_1_m1', 3, {'scaling_factor': 1, 'global_self_attn': 'none'}),
    # ]
    # commands = [
    # (s1, 'X_S1_m0', 0, {'scaling_factor': 0, 'max_epochs': 10}),
    # (s1, 'X_S1_1_m0', 1, {'scaling_factor': 0, 'max_epochs': 10}),
    # (s1, 'X_S1_1_m0', 1, {'scaling_factor': 0, 'max_epochs': 10})
    # ]
    # commands = [
    # (s1, 'A_S1_m0', 0, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 0, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'A_S1_1_m0', 1, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 0, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'A_S1_2_m0', 2, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 0, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'B_S1_m0', 0, {'base_lr': 0.05,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'B_S1_1_m0', 1, {'base_lr': 0.05,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'B_S1_2_m0', 2, {'base_lr': 0.05,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'C_S1_m0', 0, {'base_lr': 5e-4,"init_lr": 0.5, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'C_S1_1_m0', 1, {'base_lr': 5e-4,"init_lr": 0.5, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'C_S1_2_m0', 2, {'base_lr': 5e-4, "init_lr": 0.5, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'D_S1_m0', 0, {'base_lr': 0.005,"init_lr": 0.8, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'D_S1_1_m0', 1, {'base_lr': 0.005,"init_lr": 0.8, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'D_S1_2_m0', 2, {'base_lr': 0.005, "init_lr": 0.8, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'E_S1_m0', 3, {"warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'E_S1_1_m0', 3, { "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'E_S1_2_m0', 3, { "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # ]
    # commands = [

    # ('temp', 'F_S1_m0', 0, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # ('temp', 'F_S1_1_m0', 1, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # ('temp', 'F_S1_2_m0', 2, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'H_S1_m0', 0, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'H_S1_1_m0', 1, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # (s1, 'H_S1_2_m0', 2, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 100, "l2_coeff": 1e-4, 'max_epochs': 100, 'no_earlystopping': False}),
    # ]

    # s1 = 'eeg_with_s1'
    # designation = 'c'
    # commands = []
    # gpu = 0
    # for i in range(10):
    #     commands.append(
    #         (s1, f"{designation}_{i}_m3", gpu, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 20, 'dropout_ff': .2, "dropout_emb": .2, 'max_epochs': 150, 'no_earlystopping': False, }))
    #     gpu = (gpu + 1) % 4 # 4 is the number of gpus
    # commands = commands[:2] # select the first 2 commands
        
    s1 = 'eeg_with_s2'
    designation = 'd'
    commands = []
    gpu = 0
    for i in range(10):
        commands.append(
            (s1, f"{designation}_{i}_m3", gpu, {'base_lr': 5e-4,"init_lr": 1e-10, "warmup_steps": 20, 'dropout_ff': .2, "dropout_emb": .2, 'max_epochs': 150, 'no_earlystopping': False, }))
        gpu = (gpu + 1) % 4 # 4 is the number of gpus


    # Group commands by GPU
    commands_by_gpu = {}
    for command in commands:
        dataset, run_name, gpu, custom_params = command
        if gpu not in commands_by_gpu:
            commands_by_gpu[gpu] = []
        commands_by_gpu[gpu].append((dataset, run_name, custom_params))

    # Adjust the executor submission to pass GPU ID correctly
    with ThreadPoolExecutor(max_workers=len(commands_by_gpu)) as executor:
        futures = [executor.submit(run_commands_for_gpu, commands, gpu) for gpu, commands in commands_by_gpu.items()]
