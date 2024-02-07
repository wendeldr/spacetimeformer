#!/bin/bash

run_command() {
    local run_name=""
    local args=("$@")
    for i in "${!args[@]}"; do
	if [[ ${args[$i]} == "--run_name" ]]; then
	    run_name="${args[$((i + 1))]}"
	    break
	fi
    done

    if [ -z "$run_name" ]; then
	echo "Error: --run_name not provided."
        return 1
    fi

    python train.py spacetimeformer EDF "$@" > "${run_name}.log" 2>&1
}

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_1 --context_points 32 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 2000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP1,SP2,TO1,TO2,TO'1,TO'2" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_1_1 --context_points 32 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 2000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP1,SP2,TO1,TO2,TO'1,TO'2" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_2 --context_points 32 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 2000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP2,TO1,TO2,TO'1,TO'2,SP1" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_2_1 --context_points 32 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 2000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP2,TO1,TO2,TO'1,TO'2,SP1" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_3 --context_points 32 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 2000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP1,TO1,SP2,TO2,TO'1,TO'2" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_4 --context_points 256 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 1000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP1,SP2,TO1,TO2,TO'1,TO'2" --wandb --no_earlystopping

run_command --data_path /home/wendeldr/git/spacetimeformer/spacetimeformer/data/edf/OvertNaming.EDF --run_name EDF_TEST_5 --context_points 256 --target_points 1 --n_heads 1 --enc_layers 1 --dec_layers 1 --d_model 100 --d_qk 100 --d_v 100 --d_ff 400 --global_self_attn full --local_self_attn full --global_cross_attn full --local_cross_attn full --batch_size 1000 --attn_plot --plot --plot_samples 1 --max_epochs 10 --gpus 1 --channels "SP2,TO1,TO2,TO'1,TO'2,SP1" --wandb --no_earlystopping

