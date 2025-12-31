# predict.py
import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from my_detr_1 import DETR
from train import box_cxcywh_to_xyxy


def normalize_img(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device)[:, None, None]
    return (t - mean) / std


def load_image(path, size=800, device="cpu"):
    img = Image.open(path).convert("RGB")
    img_r = img.resize((size, size), Image.BILINEAR)
    t = torchvision.transforms.functional.to_tensor(img_r).to(device)
    t = normalize_img(t).unsqueeze(0)  # (1,3,H,W)
    return img, img_r, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="pred.png")
    ap.add_argument("--num_classes", type=int, default=91)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    model = DETR(num_classes=args.num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    img_orig, img_resized, x = load_image(args.image, size=args.size, device=device)

    with torch.no_grad():
        pred_logits, pred_boxes = model(x)  # (Q,B,C+1), (Q,B,4)
        prob = pred_logits.softmax(-1)[:, 0, :-1]  # 去掉 no-object
        scores, labels = prob.max(-1)              # (Q,)
        keep = scores > args.score

        boxes = pred_boxes[:, 0, :][keep]          # cxcywh in [0,1]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)     # [0,1]
        boxes_xyxy = (boxes_xyxy * args.size).clamp(0, args.size)

        scores = scores[keep].cpu().tolist()
        labels = labels[keep].cpu().tolist()
        boxes_xyxy = boxes_xyxy.cpu().tolist()

    # 画在 resized 图上（更简单）
    draw = ImageDraw.Draw(img_resized)
    for (x0, y0, x1, y1), s, c in zip(boxes_xyxy, scores, labels):
        draw.rectangle([x0, y0, x1, y1], width=2)
        draw.text((x0, y0), f"{c}:{s:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(args.out)
    print(f"saved -> {args.out} | dets={len(scores)}")


if __name__ == "__main__":
    main()
