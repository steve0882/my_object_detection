# 训练
python train.py \
  --train_img /path/train2017 \
  --train_ann /path/annotations/instances_train2017.json \
  --val_img   /path/val2017 \
  --val_ann   /path/annotations/instances_val2017.json \
  --num_classes 91 --epochs 50 --batch_size 2 --save checkpoints/detr.pth \
  --resume checkpoints/detr.pth

# 验证
python validate.py --val_img /path/val2017 --val_ann /path/annotations/instances_val2017.json --ckpt checkpoints/detr.pth

# 推理
python predict.py --image test.jpg --ckpt checkpoints/detr.pth --out out.png --score 0.6
