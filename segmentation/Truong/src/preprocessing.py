import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from config import Config

normalize_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225])

def extract_image_features(image_gray):
    img_norm    = image_gray.astype(np.float32) / 255.0
    mean_val    = float(np.mean(img_norm))
    std_val     = float(np.std(img_norm))
    median_val  = float(np.median(img_norm))
    img_uint8   = (img_norm * 255).astype(np.uint8)
    img_q       = (img_uint8 // 8).astype(np.uint8)
    glcm        = graycomatrix(img_q, distances=[1],
                                angles=[0, np.pi/4],
                                levels=32, symmetric=True, normed=True)
    contrast    = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    edges       = sobel(img_norm)
    edge_energy = float(np.mean(edges))
    dark_ratio  = float(np.mean(img_norm < 0.30))
    lap_var     = float(cv2.Laplacian(img_uint8, cv2.CV_64F).var())
    lap_norm    = np.tanh(lap_var / 1000.0)
    hist, bins  = np.histogram(img_norm.flatten(), bins=64)
    hist_n      = hist / (hist.sum() + 1e-6)
    bc          = (bins[:-1] + bins[1:]) / 2
    h_mean      = float(np.sum(hist_n * bc))
    h_std       = float(np.sqrt(np.sum(hist_n * (bc - h_mean)**2)))
    h_skew      = float(np.sum(hist_n * ((bc - h_mean) / (h_std + 1e-6))**3))
    h_kurt      = float(np.sum(hist_n * ((bc - h_mean) / (h_std + 1e-6))**4))
    fft_rows    = np.abs(np.fft.fft(img_norm, axis=1))
    fft_cols    = np.abs(np.fft.fft(img_norm, axis=0))
    fft_feat    = (float(np.mean(fft_rows[:, 1:img_norm.shape[1]//2])) +
                   float(np.mean(fft_cols[1:img_norm.shape[0]//2, :])))
    p10         = float(np.percentile(img_norm, 10))
    p90         = float(np.percentile(img_norm, 90))
    return np.array([mean_val, std_val, median_val,
                     contrast, homogeneity, edge_energy,
                     dark_ratio, lap_norm, h_skew, h_kurt,
                     fft_feat, p10, p90, p90 - p10],
                    dtype=np.float32)

def gavilan_preprocess(image_gray, sizepre, thpre):
    h, w   = image_gray.shape
    step   = sizepre // 2
    out_h  = max(1, (h - sizepre) // step + 1)
    out_w  = max(1, (w - sizepre) // step + 1)
    result = np.zeros((out_h, out_w), dtype=np.uint8)
    for i in range(out_h):
        for j in range(out_w):
            y0, y1 = i * step, i * step + sizepre
            x0, x1 = j * step, j * step + sizepre
            roi    = image_gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            hist, _  = np.histogram(roi.flatten(), bins=256, range=(0, 255))
            cumhist  = np.cumsum(hist)
            pout     = np.searchsorted(cumhist, thpre * roi.size)
            result[i, j] = int(np.clip(pout, 0, 255))
    return result

def apply_seed_mask(preprocessed, KS):
    kernel    = np.ones((5, 5), np.float32) / 25
    local_avg = cv2.filter2D(preprocessed.astype(np.float32), -1, kernel)
    return ((preprocessed.astype(np.float32) <= KS * local_avg)
            .astype(np.uint8) * 255)

def full_preprocessing(image_gray, params):
    sizepre   = int(params["sizepre"])
    thpre     = float(params["thpre"])
    KS        = float(params["KS"])
    thSymDiff = float(params["thSymDiff"])

    pre      = gavilan_preprocess(image_gray, sizepre, thpre)
    seed     = apply_seed_mask(pre, KS)
    seed_bin = (seed > 127).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.dilate(seed_bin, kernel, iterations=1)
    eroded   = cv2.erode(seed_bin,  kernel, iterations=1)
    sym_diff = (dilated - eroded).astype(np.float32)
    refined  = np.where(sym_diff * 255 > thSymDiff,
                        seed.astype(np.float32),
                        pre.astype(np.float32))
    refined  = np.clip(refined, 0, 255).astype(np.uint8)
    pre_r    = cv2.resize(pre,
                           (image_gray.shape[1], image_gray.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    refined_r = cv2.resize(refined,
                            (image_gray.shape[1], image_gray.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    return {"preprocessed": pre_r, "refined": refined_r}

def preprocess_to_tensor(preprocessed, device):
    img_3ch = np.stack([preprocessed] * 3,
                        axis=-1).astype(np.float32) / 255.0
    t = torch.from_numpy(img_3ch.transpose(2, 0, 1))
    return normalize_fn(t).unsqueeze(0).to(device)