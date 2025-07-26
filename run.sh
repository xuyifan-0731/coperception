python train_bev_v2.py \
  --data_root /Users/xuyifan/Downloads/coperception/train/agent1 \
  --data_root_val /Users/xuyifan/Downloads/coperception/test/agent1 \
  --T 5 \
  --batch 32 \
  --epochs 50 \
  --lr 1e-3 \
  --eval_interval 100 \
  --project test-v2x \
  --name dynw_lambda0.5 \
  --use_aug \
  --use_dyn_weight --lambda_d 0.5 \


# 11091f31fca78c206c54824275e24bd6bc31662c