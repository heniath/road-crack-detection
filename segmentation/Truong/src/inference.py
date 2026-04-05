import torch
import cv2
import numpy as np
from pathlib import Path
from config import Config
from preprocessing import (extract_image_features,
                             full_preprocessing,
                             preprocess_to_tensor)
from model import ResNet50UNetCBAM, ParamMLP

def load_models(device):
    # Segmentation model
    seg_model = ResNet50UNetCBAM().to(device)
    seg_model.load_state_dict(
        torch.load(Config.MODEL_PATH, map_location=device))
    seg_model.eval()
    print(f"  Loaded: {Config.MODEL_PATH}")

    # MLP
    mlp_model = ParamMLP(
        input_dim  = Config.FEATURE_DIM,
        hidden     = Config.MLP_HIDDEN,
        output_dim = 4).to(device)
    mlp_model.load_state_dict(
        torch.load(Config.MLP_PATH, map_location=device))
    mlp_model.eval()
    print(f"  Loaded: {Config.MLP_PATH}")

    return seg_model, mlp_model

@torch.no_grad()
def predict(image_bgr, seg_model, mlp_model, device):
    orig_h, orig_w = image_bgr.shape[:2]

    # Grayscale + resize
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray,
                           (Config.IMG_SIZE, Config.IMG_SIZE))

    # MLP adaptive params
    feat_t = torch.tensor(
        extract_image_features(img_gray)).unsqueeze(0).to(device)
    output = mlp_model(feat_t)
    params = mlp_model.to_params(output, Config.PARAM_RANGES)

    # Preprocessing
    out  = full_preprocessing(img_gray, params)
    pre  = out["preprocessed"]

    # Inference
    t      = preprocess_to_tensor(pre, device)
    logits = seg_model(t)
    pred   = (torch.sigmoid(logits) > Config.THRESHOLD)\
                  .float().squeeze().cpu().numpy()

    # Resize back to original
    pred_full = cv2.resize(
        (pred * 255).astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST)

    # Overlay (red = crack)
    overlay = image_bgr.copy()
    overlay[pred_full > 127] = [0, 0, 255]
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    crack_pct = (pred_full > 127).sum() / (orig_h * orig_w) * 100

    return pred_full, blended, crack_pct, params