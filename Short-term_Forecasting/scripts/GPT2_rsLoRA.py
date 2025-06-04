import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "GPT2_rsLoRA"

for rank in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:  # Corrected syntax
    os.system(
        f"python /home/pdey/spinning-storage/pdey/NeurIPS2024/Short-term_Forecasting/run.py "
        f"--task_name short_term_forecast "
        f"--is_training 1 "
        f"--root_path /home/pdey/spinning-storage/pdey/NeurIPS2024/datasets/m4 "
        f"--seasonal_patterns Monthly "
        f"--model_id m4_Monthly "
        f"--model {model_name} "
        f"--data m4 "
        f"--features M "
        f"--enc_in 1 "
        f"--dec_in 1 "
        f"--c_out 1 "
        f"--gpt_layer 6 "
        f"--d_ff 128 "
        f"--d_model 128 "
        f"--patch_size 1 "
        f"--rank {rank} "
        f"--stride 1 "
        f"--batch_size 16 "
        f"--des Exp "
        f"--itr 1 "
        f"--learning_rate 0.002 "
        f"--loss SMAPE"
    )

    for pattern, d_model, d_ff, lr in [
        ("Yearly", 768, 32, 0.001),
        ("Quarterly", 768, 128, 0.001),
        ("Daily", 768, 128, 0.001),
        ("Weekly", 768, 128, 0.001),
        ("Hourly", 768, 128, 0.001),
    ]:
        os.system(
            f"python /home/pdey/spinning-storage/pdey/NeurIPS2024/Short-term_Forecasting/run.py "
            f"--task_name short_term_forecast "
            f"--is_training 1 "
            f"--root_path /home/pdey/spinning-storage/pdey/NeurIPS2024/datasets/m4 "
            f"--seasonal_patterns {pattern} "
            f"--model_id m4_{pattern} "
            f"--model {model_name} "
            f"--data m4 "
            f"--features M "
            f"--enc_in 1 "
            f"--dec_in 1 "
            f"--c_out 1 "
            f"--gpt_layer 6 "
            f"--d_model {d_model} "
            f"--d_ff {d_ff} "
            f"--patch_size 1 "
            f"--rank {rank} "
            f"--stride 1 "
            f"--batch_size 16 "
            f"--des Exp "
            f"--itr 1 "
            f"--learning_rate {lr} "
            f"--loss SMAPE"
        )