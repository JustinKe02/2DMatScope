"""
enhancement — 图像增强管线: USM 锐化 → Gamma 校正 → CLAHE
"""

import numpy as np
import cv2


def apply_enhancements(frame, sharpen=0.0, gamma=1.0, clahe_clip=0.0):
    """对帧应用图像增强管线

    Args:
        frame: BGR numpy 数组
        sharpen: USM 锐化强度 0~3.0
        gamma: Gamma 校正 0.5~2.5
        clahe_clip: CLAHE clip limit 0~5.0 (0 = 关闭)

    Returns:
        增强后的 BGR numpy 数组
    """
    img = frame

    # 1) USM 锐化 (Unsharp Mask)
    if sharpen > 0.05:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.0 + sharpen, blurred, -sharpen, 0)

    # 2) Gamma 校正
    if abs(gamma - 1.0) > 0.02:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ], dtype=np.uint8)
        img = cv2.LUT(img, table)

    # 3) CLAHE 对比度增强
    if clahe_clip > 0.1:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return img
