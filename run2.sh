python train_unet.py \
  --data_root /Users/xuyifan/Downloads/coperception/train/agent1 \
  --data_root_val /Users/xuyifan/Downloads/coperception/test/agent1 \
  --T 5 \
  --batch 32 \
  --epochs 50 \
  --lr 1e-3 \
  --eval_interval 100 \
  --project test-v2x \
  --name unent-full \
  --use_aug \
  --use_dyn_weight    --lambda_d 1.0 \
  --use_change_head   --alpha_chg 0.5 \
  --use_dice          --alpha_dice 1.0 \
  --use_focal         --alpha_focal 1.0 \
  --use_contrast      --alpha_contrast 0.1 \
  --static_sampling_ratio 0.3 \
  --curriculum_epochs 2


# 11091f31fca78c206c54824275e24bd6bc31662c