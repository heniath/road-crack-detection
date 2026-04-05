import os

class Config:
    # Paths
    MODEL_PATH   = "models/session5_best_model.pth"
    MLP_PATH     = "models/mlp_4param.pth"
    INPUT_DIR    = "input"
    OUTPUT_DIR   = "output"

    # Model
    IMG_SIZE     = 256
    THRESHOLD    = 0.5
    DEVICE       = "cuda"   # Jetson Nano có CUDA

    # 4 params
    PARAM_RANGES = {
        "sizepre":   (8,    64),
        "thpre":     (0.05, 0.40),
        "KS":        (0.5,  2.0),
        "thSymDiff": (5.0,  20.0),
    }
    MLP_HIDDEN   = [64, 64, 32]
    FEATURE_DIM  = 14

    # Output
    SAVE_OVERLAY = True
    SAVE_MASK    = True
    SHOW_WINDOW  = True        # cv2.imshow nếu có màn hình