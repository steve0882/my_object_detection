# train.py
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.ops import generalized_box_iou
from PIL import Image

from my_detr_1 import DETR  # 你的模型文件（forward 已修正 inputs / tgt repeat）


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0+x1)/2, (y0+y1)/2, (x1-x0), (y1-y0)], dim=-1)

def _normalize_img(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device)[:, None, None]
    return (t - mean) / std


class CocoDetrDataset(torchvision.datasets.CocoDetection):
    """label: 0..num_classes-1 (0-based), COCO bbox=[x,y,w,h]"""
    def __init__(self, img_dir, ann_file, size=800):
        super().__init__(img_dir, ann_file)
        self.size = size

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = img.convert("RGB")
        w0, h0 = img.size

        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_t = torchvision.transforms.functional.to_tensor(img)
        img_t = _normalize_img(img_t)

        boxes, labels = [], []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            x0 = x / w0 * self.size
            y0 = y / h0 * self.size
            x1 = (x + w) / w0 * self.size
            y1 = (y + h) / h0 * self.size
            boxes.append([x0, y0, x1, y1])
            labels.append(a["category_id"])  # 你的数据是 0-based，无需映射

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        boxes = box_xyxy_to_cxcywh(boxes) / self.size  # normalize to [0,1]
        target = {"boxes": boxes.clamp(0, 1), "labels": labels}
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


def hungarian_match(outputs, targets, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
    pred_logits, pred_boxes = outputs
    Q, B, _ = pred_logits.shape
    device = pred_logits.device
    out_prob = pred_logits.softmax(-1)
    out_bbox = pred_boxes

    indices = []
    for b in range(B):
        tgt_ids = targets[b]["labels"]
        tgt_bbox = targets[b]["boxes"]
        if tgt_bbox.numel() == 0:
            indices.append((torch.empty(0, dtype=torch.int64),
                            torch.empty(0, dtype=torch.int64)))
            continue

        cost_cls = -out_prob[:, b, tgt_ids]
        cost_l1 = torch.cdist(out_bbox[:, b], tgt_bbox, p=1)

        out_xyxy = box_cxcywh_to_xyxy(out_bbox[:, b])
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_g = -generalized_box_iou(out_xyxy, tgt_xyxy)

        C = (cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_g).detach().cpu()

        try:
            from scipy.optimize import linear_sum_assignment
            i, j = linear_sum_assignment(C)
            i = torch.as_tensor(i, dtype=torch.int64, device=device)
            j = torch.as_tensor(j, dtype=torch.int64, device=device)
        except Exception:
            # greedy fallback
            i_list, j_list = [], []
            Cw = C.clone()
            for _ in range(min(Cw.shape[0], Cw.shape[1])):
                k = torch.argmin(Cw)
                pi = (k // Cw.shape[1]).item()
                tj = (k %  Cw.shape[1]).item()
                i_list.append(pi); j_list.append(tj)
                Cw[pi, :] = 1e9
                Cw[:, tj] = 1e9
            i = torch.tensor(i_list, dtype=torch.int64, device=device)
            j = torch.tensor(j_list, dtype=torch.int64, device=device)

        indices.append((i, j))
    return indices


class DetrCriterion(nn.Module):
    """背景类 index = num_classes（最后一类）"""
    def __init__(self, num_classes, w_ce=1.0, w_l1=5.0, w_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.w_ce, self.w_l1, self.w_giou = w_ce, w_l1, w_giou

    def forward(self, outputs, targets):
        pred_logits, pred_boxes = outputs
        Q, B, _ = pred_logits.shape
        device = pred_logits.device

        indices = hungarian_match(outputs, targets)

        tgt_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)
        tgt_boxes = torch.zeros((B, Q, 4), dtype=torch.float32, device=device)
        mask = torch.zeros((B, Q), dtype=torch.bool, device=device)

        for b, (i, j) in enumerate(indices):
            if i.numel() == 0:
                continue
            tgt_classes[b, i] = targets[b]["labels"][j]
            tgt_boxes[b, i] = targets[b]["boxes"][j]
            mask[b, i] = True

        ce = F.cross_entropy(
            pred_logits.permute(1, 0, 2).reshape(-1, pred_logits.shape[-1]),
            tgt_classes.reshape(-1),
            reduction="mean"
        )

        if mask.any():
            pb = pred_boxes.permute(1, 0, 2)[mask]
            tb = tgt_boxes[mask]
            l1 = F.l1_loss(pb, tb, reduction="mean")
            giou = 1.0 - torch.diag(
                generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb))
            ).mean()
        else:
            l1 = torch.tensor(0.0, device=device)
            giou = torch.tensor(0.0, device=device)

        loss = self.w_ce * ce + self.w_l1 * l1 + self.w_giou * giou
        return {"loss": loss, "loss_ce": ce, "loss_l1": l1, "loss_giou": giou}


def save_ckpt(path, model, optim, epoch, best_val):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }, str(path))


def try_resume(resume_path, model, optim, device):
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
        # 把 optimizer state 搬到 device（很关键）
        for s in optim.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    return start_epoch, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img", required=True)
    ap.add_argument("--train_ann", required=True)
    ap.add_argument("--val_img", required=True)
    ap.add_argument("--val_ann", required=True)

    ap.add_argument("--num_classes", type=int, default=91)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=800)

    ap.add_argument("--save", default="checkpoints/detr.pth")
    ap.add_argument("--resume", default="")  # <-- 新增：resume ckpt 路径（可为空）
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    train_ds = CocoDetrDataset(args.train_img, args.train_ann, size=args.size)
    val_ds   = CocoDetrDataset(args.val_img,   args.val_ann,   size=args.size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = DETR(num_classes=args.num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6).to(device)

    criterion = DetrCriterion(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val = float("inf")

    # -------- resume --------
    if args.resume:
        start_epoch, best_val = try_resume(args.resume, model, optimizer, device)
        print(f"Resumed from {args.resume} | start_epoch={start_epoch} | best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(imgs)
            loss = criterion(outputs, targets)["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            v = 0.0
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(imgs)
                v += criterion(outputs, targets)["loss"].item()
            val_loss = v / max(1, len(val_loader))

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(args.save, model, optimizer, epoch, best_val)
            print(f"  -> saved: {args.save}")

if __name__ == "__main__":
    main()
