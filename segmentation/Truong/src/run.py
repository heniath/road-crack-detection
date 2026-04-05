import cv2
import numpy as np
import torch
import time
from pathlib import Path
from config import Config
from inference import load_models, predict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def compute_metrics(pred_mask, gt_mask):
    """Tính IoU, Dice, Precision, Recall từ 2 mask nhị phân."""
    pred = (pred_mask > 127).astype(np.float32)
    gt   = (gt_mask   > 127).astype(np.float32)
    smooth = 1e-6
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    prec = tp / (tp + fp + smooth)
    rec  = tp / (tp + fn + smooth)
    return {"iou": iou, "dice": dice, "prec": prec, "rec": rec}

def process_image(img_path, mask_path,
                  seg_model, mlp_model, device, results):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Cannot read: {img_path.name}")
        return

    t0 = time.time()
    pred_mask, blended, crack_pct, params = predict(
        img, seg_model, mlp_model, device)
    elapsed = time.time() - t0

    # ── Metrics nếu có mask ──────────────────────────────────
    metrics_str = ""
    if mask_path is not None and mask_path.exists():
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            gt = cv2.resize(gt, (pred_mask.shape[1],
                                  pred_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            m = compute_metrics(pred_mask, gt)
            results.append(m)
            metrics_str = (f" | IoU={m['iou']:.3f}"
                           f" Dice={m['dice']:.3f}"
                           f" P={m['prec']:.3f}"
                           f" R={m['rec']:.3f}")

    print(f"  {img_path.name} | crack={crack_pct:.1f}%"
          f" | {elapsed:.2f}s"
          f" | sz={params['sizepre']:.0f}"
          f" th={params['thpre']:.2f}"
          f" KS={params['KS']:.2f}"
          f" tSD={params['thSymDiff']:.1f}"
          f"{metrics_str}")

    # ── Save outputs ─────────────────────────────────────────
    stem = img_path.stem
    out  = Path(Config.OUTPUT_DIR)

    if Config.SAVE_MASK:
        cv2.imwrite(str(out / f"{stem}_mask.png"), pred_mask)

    if Config.SAVE_OVERLAY:
        cv2.putText(blended,
                    f"Crack: {crack_pct:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        if metrics_str:
            cv2.putText(blended,
                        f"IoU={results[-1]['iou']:.3f}  "
                        f"Dice={results[-1]['dice']:.3f}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), blended)

    # ── Show window ──────────────────────────────────────────
    if Config.SHOW_WINDOW:
        panels = [img,
                  cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
                  blended]
        if mask_path is not None and mask_path.exists():
            gt_bgr = cv2.cvtColor(
                cv2.resize(gt, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR)
            panels.insert(2, gt_bgr)   # original | pred | gt | overlay
        combined = np.hstack(panels)
        cv2.imshow("Crack Detection", combined)
        cv2.waitKey(0)

def main():
    print("=" * 55)
    print("Crack Detection — Jetson Nano")
    print("Session 5 | MLP 4-param | ResNet50-UNet+CBAM")
    print("=" * 55)

    device = torch.device(Config.DEVICE
                           if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\nLoading models ...")
    seg_model, mlp_model = load_models(device)

    # ── Scan input/images/ ───────────────────────────────────
    img_dir  = Path(Config.INPUT_DIR) / "images"
    mask_dir = Path(Config.INPUT_DIR) / "masks"

    if not img_dir.exists():
        print(f"  Folder not found: {img_dir}")
        return

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in IMAGE_EXTS])

    if not images:
        print("  No images found in input/images/")
        return

    has_masks = mask_dir.exists()
    print(f"\nFound: {len(images)} images"
          f" | masks folder: {'yes' if has_masks else 'no'}")
    print("-" * 55)

    results = []   # collect metrics

    for img_path in images:
        # Tìm mask cùng tên (bất kể extension)
        mask_path = None
        if has_masks:
            for ext in IMAGE_EXTS:
                candidate = mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

        process_image(img_path, mask_path,
                      seg_model, mlp_model, device, results)

    # ── Summary metrics ──────────────────────────────────────
    if results:
        print("\n" + "=" * 55)
        print(f"Summary — {len(results)} images with masks")
        print("=" * 55)
        for key in ["iou", "dice", "prec", "rec"]:
            vals = [r[key] for r in results]
            print(f"  {key.upper():<6} "
                  f"mean={np.mean(vals):.4f}  "
                  f"min={np.min(vals):.4f}  "
                  f"max={np.max(vals):.4f}")
        print("=" * 55)

        # Lưu CSV
        import csv
        csv_path = Path(Config.OUTPUT_DIR) / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image", "iou", "dice", "prec", "rec"])
            writer.writeheader()
            for img_path, r in zip(
                    [p for p in images
                     if (mask_dir / p.name).exists()
                     or any((mask_dir / (p.stem + e)).exists()
                            for e in IMAGE_EXTS)],
                    results):
                writer.writerow({
                    "image": img_path.name,
                    "iou":   f"{r['iou']:.4f}",
                    "dice":  f"{r['dice']:.4f}",
                    "prec":  f"{r['prec']:.4f}",
                    "rec":   f"{r['rec']:.4f}"})
        print(f"\n  Metrics saved: {csv_path}")

    cv2.destroyAllWindows()
    print(f"\nDone. Results saved to: {Config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()