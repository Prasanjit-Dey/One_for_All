
for patch in 4
do
for stride in 2
do
for lr in 0.0001
do

python /home/pdey/spinning-storage/pdey/NeurIPS2024/Classification_One_for_All/src/main.py \
    --output_dir FaceDetection \
    --comment "classification from Scratch" \
    --name FaceDetection_Rank1024 \
    --records_file FaceDetection_Rank1024.xls \
    --data_dir /home/pdey/spinning-storage/pdey/NeurIPS2024/datasets/FaceDetection \
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
    --rank 1024

done
done
done