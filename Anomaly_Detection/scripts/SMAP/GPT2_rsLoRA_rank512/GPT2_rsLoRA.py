import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rank = 512

os.system(
    f"python /NeurIPS2024/Anomaly_Detection/run.py \
      --task_name anomaly_detection \
      --is_training 1 \
      --root_path /NeurIPS2024/datasets/SMAP \
      --model_id SMAP \
      --model GPT2_rsLoRA \
      --data SMAP \
      --features M \
      --seq_len 100 \
      --pred_len 0 \
      --gpt_layer 6 \
      --d_model 768 \
      --d_ff 768 \
      --patch_size 1 \
      --stride 1 \
      --rank {rank} \
      --enc_in 25 \
      --c_out 25 \
      --anomaly_ratio 1 \
      --batch_size 128 \
      --learning_rate 0.0005 \
      --train_epochs 10"
    )
