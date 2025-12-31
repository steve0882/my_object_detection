# validate.py
import argparse
import torch
from torch.utils.data import DataLoader

import torchvision
from PIL import Image

from my_detr_1 import DETR
from train import CocoDetrDataset, collate_fn, DetrCriterion  # 复用 train.py 里的实现


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_img", required=True)
    ap.add_argument("--val_ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_classes", type=int, default=91)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    ds = CocoDetrDataset(args.val_img, args.val_ann, size=args.size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = DETR(num_classes=args.num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    criterion = DetrCriterion(num_classes=args.num_classes).to(device)

    total = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(imgs)
            total += criterion(outputs, targets)["loss"].item()

    val_loss = total / max(1, len(loader))
    print(f"val_loss={val_loss:.4f}  (ckpt={args.ckpt})")


if __name__ == "__main__":
    main()
