import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rank = 512
seq_len = 336
model = "GPT2_rsLoRA"

for percent in [100]:
    for pred_len in [96, 192, 336, 720]:
        for lr in [0.0001]:
            os.system(
                f"python /NeurIPS2024/Long-term_Forecasting/main.py \
                --root_path /NeurIPS2024/datasets/ETT-small/ \
                --data_path ETTh2.csv \
                --model_id ETTh2_{model}_{seq_len}_{pred_len}_{percent} \
                --data ett_h \
                --seq_len {seq_len} \
                --label_len 168 \
                --pred_len {pred_len} \
                --batch_size 256 \
                --lradj type4 \
                --learning_rate {lr} \
                --train_epochs 10 \
                --decay_fac 0.5 \
                --d_model 768 \
                --n_heads 4 \
                --d_ff 768 \
                --dropout 1 \
                --enc_in 7 \
                --c_out 7 \
                --freq 0 \
                --patch_size 16 \
                --stride 8 \
                --percent {percent} \
                --gpt_layer 6 \
                --rank {rank} \
                --itr 3 \
                --model {model} \
                --tmax 20 \
                --cos 1 \
                --is_gpt 1"
            )
