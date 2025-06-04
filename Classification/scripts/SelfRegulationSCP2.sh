#export CUDA_VISIBLE_DEVICES=1
for rank in 2 4 8 16 32 64 128 256 512 1024 
do
for lr in 0.0005
do
for patch in 16
do
for stride in 16
do

python /home/pdey/spinning-storage/pdey/NeurIPS2024/Classification_One_for_All/src/main.py \
    --output_dir SCP2 \
    --comment "classification from Scratch" \
    --name SCP2_Rank_$rank \
    --records_file SP2_Rank_$rank.xls \
    --data_dir /home/pdey/spinning-storage/pdey/NeurIPS2024/datasets/SelfRegulationSCP2 \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr $lr \
    --patch_size $patch \
    --stride $stride \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --rank $rank

done
done
done
done