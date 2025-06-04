for rank in 2 4 8 16 32 64 128 256 512 1024 
do
for lr in 0.002
do
for patch in 16
do
for stride in 16
do

python /home/pdey/spinning-storage/pdey/NeurIPS2024/Classification_One_for_All/src/main.py \
    --output_dir SCP1 \
    --comment "classification from Scratch" \
    --name SCP1_Rank_$rank \
    --records_file SCP1_Rank_$rank.xls \
    --data_dir /home/pdey/spinning-storage/pdey/NeurIPS2024/datasets/SelfRegulationSCP1 \
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