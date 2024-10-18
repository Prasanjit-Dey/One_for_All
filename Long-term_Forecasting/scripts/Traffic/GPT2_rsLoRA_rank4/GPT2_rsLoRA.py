import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rank = 4
seq_len = 512
model = "GPT2_rsLoRA"

for percent in [100]:
    for pred_len in [96, 192, 336, 720]:
            os.system(
                f"python /NeurIPS2024/Long-term_Forecasting/main.py \
                --root_path /NeurIPS2024/datasets/traffic/ \
                --data_path traffic.csv \
                --model_id traffic_{model}_{seq_len}_{pred_len}_{percent} \
                --data custom \
                --seq_len {seq_len} \
                --label_len 48 \
                --pred_len {pred_len} \
                --batch_size 2048 \
                --learning_rate 0.001 \
                --train_epochs 10 \
                --decay_fac 0.75 \
                --d_model 768 \
                --n_heads 4 \
                --d_ff 768 \
                --freq 0 \
                --rank {rank} \
                --patch_size 16 \
                --stride 8 \
                --percent {percent} \
                --gpt_layer 6 \
                --itr 3 \
                --model {model} \
                --patience 3 \
                --tmax 10 \
                --cos 1 \
                --is_gpt 1"
            )
