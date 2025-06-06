import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rank = 256
seq_len = 512
model = "GPT2_rsLoRA"

for percent in [100]:
    for pred_len in [96, 192, 336, 720]:
            os.system(
                f"python /NeurIPS2024/Long-term_Forecasting/main.py \
                --root_path /NeurIPS2024/datasets/weather/ \
                --data_path weather.csv \
                --model_id weather_{model}_{seq_len}_{pred_len}_{percent} \
                --data custom \
                --seq_len {seq_len} \
                --label_len 48 \
                --pred_len {pred_len} \
                --batch_size 512 \
                --lradj type3 \
                --learning_rate 0.0001 \
                --train_epochs 10 \
                --decay_fac 0.9 \
                --d_model 768 \
                --n_heads 4 \
                --d_ff 768 \
                --dropout 0.3 \
                --enc_in 7 \
                --c_out 7 \
                --freq 0 \
                --rank {rank} \
                --patch_size 16 \
                --stride 8 \
                --percent {percent} \
                --gpt_layer 6 \
                --itr 3 \
                --model {model} \
                --tmax 20 \
                --cos 1 \
                --is_gpt 1"
            )
