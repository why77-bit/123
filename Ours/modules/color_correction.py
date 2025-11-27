# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from utils.color import srgb_to_linear_arr, linear_to_srgb_arr

#色彩校正模块：用 b（线性中值）估算 s 并校正数字图
def color_correct_bg(B_x_path,
                     b,
                     out_path='digital_corrected.png'):
    """
    digital_path: 数字文档原图路径 (sRGB uint8)
    bd_lin: 从拍摄图估计的 b（线性 RGB）
    assume_Bd_white: 若 True 则数字背景 Bd 视为纯白；否则自动从数字图四角估计 Bd
    返回：s (逐通道乘性因子), 保存校正后图像
    """
    eps = 1e-6
    if not os.path.exists(B_x_path):
        raise FileNotFoundError(f'找不到数字图 {B_x_path}')

    B_x_sRGB = cv2.imread(B_x_path).astype(np.float32) / 255.0
    h, w = B_x_sRGB.shape[:2]

    # 估计 Bd（数字背景）— 这里直接为纯白背景
    B_srgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    B_line = srgb_to_linear_arr(B_srgb.reshape(1,1,3).astype(np.float32))[0,0,:]

    # 计算乘性因子 s（线性空间）
    sc = b / (B_line + eps)
    print('估计乘性因子 s (B,G,R) =', sc.tolist())

    # 矫正B_x
    B_x_line = srgb_to_linear_arr(B_x_sRGB)
    B_x_corr_line = B_x_line * sc.reshape(1,1,3)
    B_x_corr_line = np.clip(B_x_corr_line, 0.0, 1.0)
    #保存
    B_x_corr_sRGB = linear_to_srgb_arr(B_x_corr_line)
    cv2.imwrite(out_path, (B_x_corr_sRGB * 255.0).astype(np.uint8))

    return sc, B_x_corr_line

# ---------- 新增：从拍摄的白底摩尔图 I(x) 与 sc 得到无背景摩尔纹 M(x) ----------
def compute_moire_texture(I_x_sRGB, sc):
    """
    captured_moire_bgr: 原始拍摄摩尔图 (BGR uint8)
    sc: 逐通道乘性因子 (B,G,R) 线性空间
    target_shape: 若要调整 M 到指定形状 (h,w)
    返回 M_lin (浮点，线性 RGB，0..1)
    说明：M = I(x) / s_c  （线性空间）
    """
    # 转到 sRGB 0..1 -> 线性
    I_x_sRGB = (I_x_sRGB.astype(np.float32) / 255.0)
    I_x_line = srgb_to_linear_arr(I_x_sRGB)
    # 避免除以0
    sc_safe = np.maximum(np.array(sc).reshape(1,1,3), 1e-6)
    M_x_line = I_x_line / sc_safe  # 按通道除
    M_x_line = np.clip(M_x_line, 0.0, 1.0)
    
    return M_x_line
