import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rank = 128
seq_len = 512
model = "GPT2_rsLoRA"

for percent in [5, 10]:
    for pred_len in [96, 192, 336, 720]:
            os.system(
                f"python /NeurIPS2024/Few-shot_Learning/main.py \
                --root_path /NeurIPS2024/datasets/ETT-small/ \
                --data_path ETTm2.csv \
                --model_id ETTm2_{model}_{seq_len}_{pred_len}_{percent} \
                --data ett_m \
                --seq_len {seq_len} \
                --label_len 48 \
                --pred_len {pred_len} \
                --batch_size 256 \
                --lradj type4 \
                --learning_rate 0.002 \
                --train_epochs 10 \
                --decay_fac 0.75 \
                --d_model 768 \
                --n_heads 4 \
                --rank {rank} \
                --d_ff 768 \
                --dropout 0.3 \
                --enc_in 7 \
                --c_out 7 \
                --freq 0 \
                --patch_size 16 \
                --stride 16 \
                --percent {percent} \
                --gpt_layer 6 \
                --itr 3 \
                --model {model} \
                --tmax 20 \
                --cos 1 \
                --is_gpt 1"
            )
