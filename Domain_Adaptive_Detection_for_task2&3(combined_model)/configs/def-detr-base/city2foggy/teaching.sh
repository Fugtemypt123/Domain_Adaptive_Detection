N_GPUS=1
BATCH_SIZE=2
DATA_ROOT=/network_space/storage43/qinyiming/DAOD/Domain_Adaptive_Detection/data
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26502 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 80 \
--epoch_lr_drop 80 \
--mode teaching \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../source_only/model_last.pth \
# --resume /network_space/storage43/qinyiming/DAOD_new/Domain_Adaptive_Detection/outputs/def-detr-base/city2foggy/source_only/model_last.pth \
--epoch_retrain 40 \
--epoch_mae_decay 10 \
--threshold 0.3 \
--max_dt 0.45
