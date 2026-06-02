PREV_MODEL="./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
DATA_PATH="./data/Nav-Finetune/open_uninavid_sampled_500.json"
MODEL_PATH="./model_zoo/univid-7b-full-224-video-fps-1-grid-2-from-uninavid"



deepspeed   uninavid/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $PREV_MODEL \
    --version imgsp_v1 \
    --data_path $DATA_PATH \
    --image_folder ./data/Nav-Finetune \
    --video_folder ./data/Nav-Finetune \
    --vision_tower ./model_zoo/eva_vit_g.pth \
    --image_processor ./uninavid/processor/clip-patch14-224 \
    --tune_vision_encoder False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --video_fps 1 \
    --compress_type "grid:2" \
    --bf16 True \
    --output_dir $MODEL_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 8000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
