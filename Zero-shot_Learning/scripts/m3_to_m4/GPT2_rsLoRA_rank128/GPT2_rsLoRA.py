import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = "GPT2_rsLoRA"
seq_len = 16
pred_len = 8
test_seq_len = 16
test_pred_len = 8
source_data = "m3"
rank = 128

os.system(
    f"python /NeurIPS2024/Zero-shot_Learning/main_test.py \
        --root_path /NeurIPS2024/datasets/m3/ \
        --test_root_path /NeurIPS2024/datasets/m4/ \
        --data_path m3_quarterly_dataset.tsf \
        --test_data_path m4_quarterly_dataset.tsf \
        --model_id m3Quarterly_{model}_{seq_len}_{pred_len} \
        --data tsf_data \
        --seq_len {seq_len} \
        --pred_len {pred_len} \
        --test_seq_len {seq_len} \
        --test_pred_len {test_pred_len} \
        --label_len 0 \
        --batch_size 64 \
        --test_batch_size 128 \
        --learning_rate 0.002 \
        --train_epochs 10 \
        --decay_fac 1 \
        --d_model 768 \
        --n_heads 16 \
        --d_ff 512 \
        --loss_func smape \
        --percent 100 \
        --gpt_layer 6 \
        --itr 1 \
        --model {model} \
        --rank {rank} \
        --dropout 0 \
        --patch_size 2 \
        --stride 1 \
        --proj_hid 64 \
        --print_int 1000 \
        --itr 1 \
        --is_gpt 1"
    )

seq_len = 48
pred_len = 24
test_seq_len = 48
test_pred_len = 24
source_data = "m3"

os.system(
    f"python /NeurIPS2024/Zero-shot_Learning/main_test.py \
        --root_path /NeurIPS2024/datasets/m3/ \
        --test_root_path /NeurIPS2024/datasets/m4/ \
        --data_path m3_monthly_dataset.tsf \
        --test_data_path m4_monthly_dataset.tsf \
        --model_id m3Monthly_{model}_{seq_len}_{pred_len} \
        --data tsf_data \
        --seq_len {seq_len} \
        --pred_len {pred_len} \
        --test_seq_len {seq_len} \
        --test_pred_len {test_pred_len} \
        --label_len 10 \
        --batch_size 32 \
        --test_batch_size 128 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --decay_fac 0.5 \
        --d_model 768 \
        --n_heads 16 \
        --d_ff 512 \
        --loss_func smape \
        --percent 100 \
        --gpt_layer 6 \
        --itr 1 \
        --model {model} \
        --rank {rank} \
        --dropout 0 \
        --patch_size 1 \
        --stride 1 \
        --print_int 1000 \
        --itr 3 \
        --is_gpt 1"
    )


seq_len = 9
pred_len = 6
test_seq_len = 9
test_pred_len = 6
source_data = "m3"

os.system(
    f"python /NeurIPS2024/Zero-shot_Learning/main_test.py \
        --root_path /NeurIPS2024/datasets/m3/ \
        --test_root_path /NeurIPS2024/datasets/m4/ \
        --data_path m3_yearly_dataset.tsf \
        --test_data_path m4_yearly_dataset.tsf \
        --model_id m3Yearly_{model}_{seq_len}_{pred_len} \
        --data tsf_data \
        --seq_len {seq_len} \
        --pred_len {pred_len} \
        --test_seq_len {test_seq_len} \
        --test_pred_len {test_pred_len} \
        --label_len 0 \
        --batch_size 32 \
        --test_batch_size 128 \
        --learning_rate 0.002 \
        --train_epochs 10 \
        --decay_fac 0.5 \
        --d_model 768 \
        --n_heads 16 \
        --d_ff 512 \
        --loss_func smape \
        --percent 100 \
        --gpt_layer 6 \
        --itr 1 \
        --model {model} \
        --rank {rank} \
        --dropout 0 \
        --patch_size 1 \
        --stride 1 \
        --print_int 1000 \
        --itr 3 \
        --is_gpt 1"
    )
    

