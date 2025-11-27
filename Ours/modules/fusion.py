# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# ---------- 局部匹配（局部均值/方差匹配，用于重标定亮度和对比度） ----------
def local_match(src, ref, ksize=31, eps=1e-6):
    """
    在局部窗口上把 src 的均值/方差调整到 ref（避免整体亮度漂移）
    src, ref: 浮点图像 0..1, 相同尺寸
    返回调整后的 src
    """
    if ksize % 2 == 0:
        ksize += 1
    # 计算局部均值与方差（使用 box filter 近似）
    kernel = (ksize, ksize)
    # use cv2.blur for local mean
    mu_src = cv2.blur(src, kernel)
    mu_ref = cv2.blur(ref, kernel)
    mu_src_sq = cv2.blur(src*src, kernel)
    mu_ref_sq = cv2.blur(ref*ref, kernel)
    sigma_src = np.sqrt(np.maximum(mu_src_sq - mu_src*mu_src, 0.0))
    sigma_ref = np.sqrt(np.maximum(mu_ref_sq - mu_ref*mu_ref, 0.0))
    out = (src - mu_src) * (sigma_ref / (sigma_src + eps)) + mu_ref
    out = np.clip(out, 0.0, 1.0)
    return out

def fuse_log_fft1(B_lin, M_lin,
                 eps_log=1e-3,
                 freq_cutoff=0.25,
                 freq_power=4.0,
                 phase_thresh_ratio=1e-3,
                 gamma_max=0.5,
                 clip_log_range=(-10.0, 3.0),
                 local_match_ksize=31):
    """
    最简单的复数融合，无能量增强，
    """
    H, W = B_lin.shape[:2]
    log_B_line = np.log(B_lin + eps_log)
    log_M_line = np.log(M_lin + eps_log)
    min_alpha=0.5
    max_alpha=1.0
    sigma = 0.25
    u = np.arange(W) - W // 2
    v = np.arange(H) - H // 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    D = D / D.max()  # 归一化到 [0, 1]
    # 高斯型函数，中心值接近 min，边缘接近 max
    alpha = min_alpha + (max_alpha - min_alpha) * (1 - np.exp(-D**2 / (2 * sigma**2)))

    img1_b = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,0]))
    img1_g = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,1]))
    img1_r = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,2]))

    img2_b = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,0]))
    img2_g = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,1]))
    img2_r = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,2]))
    
    b_fused = img1_b + img2_b
    g_fused = img1_g + img2_g
    r_fused = img1_r + img2_r

    b_fused = np.real(np.fft.ifft2(np.fft.ifftshift(b_fused)))
    g_fused = np.real(np.fft.ifft2(np.fft.ifftshift(g_fused)))
    r_fused = np.real(np.fft.ifft2(np.fft.ifftshift(r_fused)))

    fused_img = np.stack((b_fused, g_fused, r_fused), axis=-1)

    #裁剪对数域值以防指数爆炸
    min_clip, max_clip = clip_log_range
    LB_prime_real = np.clip(fused_img, min_clip, max_clip)

    #回到线性域
    I_lin_channel = np.exp(LB_prime_real) - eps_log
    # clip to [0,1]
    I_lin_channel = np.clip(I_lin_channel, 0.0, 1.0)
    
    return I_lin_channel

def fuse_log_fft2(B_lin, M_lin,
                 eps_log=1e-3,
                 freq_cutoff=0.25,
                 freq_power=4.0,
                 phase_thresh_ratio=1e-3,
                 gamma_max=0.5,
                 clip_log_range=(-10.0, 3.0),
                 local_match_ksize=31):
    """
    最简单的复数融合，无能量增强，
    """
    H, W = B_lin.shape[:2]
    # log_B_line = np.log(B_lin + eps_log)
    log_B_line = B_lin
    log_M_line = np.log(M_lin + eps_log)
    min_alpha=0.5
    max_alpha=1.0
    sigma = 0.25
    u = np.arange(W) - W // 2
    v = np.arange(H) - H // 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    D = D / D.max()  # 归一化到 [0, 1]
    # 高斯型函数，中心值接近 min，边缘接近 max
    alpha = min_alpha + (max_alpha - min_alpha) * (1 - np.exp(-D**2 / (2 * sigma**2)))

    img1_b = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,0]))
    img1_g = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,1]))
    img1_r = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,2]))

    img2_b = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,0]))
    img2_g = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,1]))
    img2_r = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,2]))
    
    b_fused = img1_b + img2_b
    g_fused = img1_g + img2_g
    r_fused = img1_r + img2_r

    b_fused = np.real(np.fft.ifft2(np.fft.ifftshift(b_fused)))
    g_fused = np.real(np.fft.ifft2(np.fft.ifftshift(g_fused)))
    r_fused = np.real(np.fft.ifft2(np.fft.ifftshift(r_fused)))

    fused_img = np.stack((b_fused, g_fused, r_fused), axis=-1)

    #裁剪对数域值以防指数爆炸
    min_clip, max_clip = clip_log_range
    LB_prime_real = np.clip(fused_img, min_clip, max_clip)

    #回到线性域
    I_lin_channel = np.exp(LB_prime_real) - eps_log
    # clip to [0,1]
    I_lin_channel = np.clip(I_lin_channel, 0.0, 1.0)
    
    return I_lin_channel

def fuse_log_fft_(B_lin, M_lin,
                 eps_log=1e-3,
                 freq_cutoff=0.25,
                 freq_power=4.0,
                 phase_thresh_ratio=1e-3,
                 gamma_max=0.5,
                 clip_log_range=(-10.0, 3.0),
                 local_match_ksize=31):
    '''
    逐通道做幅度融合，权值可选，然后保留原始相位
    '''
    H, W = B_lin.shape[:2]
    fused = np.zeros_like(B_lin)
    for c in range(3):

        LB = np.log(B_lin[...,c] + eps_log)
        LM = np.log(M_lin[...,c] + eps_log)
        SB = np.fft.fft2(LB)
        SM = np.fft.fft2(LM)

        AB = np.abs(SB)
        AM = np.abs(SM)
        PB = np.angle(SB)
        PM = np.angle(SM)

        # min_alpha=0.5
        # max_alpha=1.0
        # sigma = 0.25
        # u = np.arange(W) - W // 2
        # v = np.arange(H) - H // 2
        # U, V = np.meshgrid(u, v)
        # D = np.sqrt(U**2 + V**2)
        # D = D / D.max()  # 归一化到 [0, 1]
        # # 高斯型函数，中心值接近 min，边缘接近 max
        # alpha = min_alpha + (max_alpha - min_alpha) * (1 - np.exp(-D**2 / (2 * sigma**2)))

        # 幅度混合（按频率权重 alpha）
        A_prime = AB + AM

        # 组合复谱
        S_prime = A_prime * np.exp(1j * PB)

        # iFFT -> 对数域图像
        LB_prime = np.fft.ifft2(S_prime)
        # 取实部（理论上是实）
        LB_prime_real = np.real(LB_prime)

        # 裁剪对数域值以防指数爆炸
        min_clip, max_clip = clip_log_range
        LB_prime_real = np.clip(LB_prime_real, min_clip, max_clip)

        # 回到线性域
        I_lin_channel = np.exp(LB_prime_real) - eps_log
        # clip to [0,1]
        I_lin_channel = np.clip(I_lin_channel, 0.0, 1.0)

        fused[...,c] = I_lin_channel

        return fused
    
def fuse_log_fft__(B_lin, M_lin,
                 eps_log=1e-3,
                 freq_cutoff=0.25,
                 freq_power=4.0,
                 phase_thresh_ratio=1e-3,
                 gamma_max=0.5,
                 clip_log_range=(-10.0, 3.0),
                 local_match_ksize=31):
    '''
    复数融合+总能量增强+对比度回归
    '''
    H, W = B_lin.shape[:2]
    log_B_line = np.log(B_lin + eps_log)
    log_M_line = np.log(M_lin + eps_log)
    min_alpha=0.5
    max_alpha=1.0
    sigma = 0.25
    u = np.arange(W) - W // 2
    v = np.arange(H) - H // 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    D = D / D.max()  # 归一化到 [0, 1]
    # 高斯型函数，中心值接近 min，边缘接近 max
    alpha = min_alpha + (max_alpha - min_alpha) * (1 - np.exp(-D**2 / (2 * sigma**2)))

    img1_b = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,0]))
    img1_g = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,1]))
    img1_r = np.fft.fftshift(np.fft.fft2(log_B_line[:,:,2]))

    img2_b = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,0]))
    img2_g = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,1]))
    img2_r = np.fft.fftshift(np.fft.fft2(log_M_line[:,:,2]))

     # 调节总能量
    A_bg_b = np.abs(img1_b)
    A_bg_g = np.abs(img1_g)
    A_bg_r = np.abs(img1_r)
    A_moire_b = np.abs(img2_b)
    A_moire_g = np.abs(img2_g)
    A_moire_r = np.abs(img2_r)
    E_bg_b = np.sum(A_bg_b)
    E_bg_g = np.sum(A_bg_g)
    E_bg_r = np.sum(A_bg_r)
    E_moire_b = np.sum(A_moire_b)
    E_moire_g = np.sum(A_moire_g)
    E_moire_r = np.sum(A_moire_r)
    E_bg = E_bg_b + E_bg_g + E_bg_r
    E_moire = E_moire_b + E_moire_g + E_moire_r
    scale = np.sqrt(E_bg / (E_moire + 1e-8))
    if scale > 1.0:
        img2_b = img2_b * scale 
        img2_g = img2_g * scale
        img2_r = img2_r * scale
    
    b_fused = (1-alpha) * img1_b + alpha * img2_b
    g_fused = (1-alpha) * img1_g + alpha * img2_g
    r_fused = (1-alpha) * img1_r + alpha * img2_r

    b_fused = np.real(np.fft.ifft2(np.fft.ifftshift(b_fused)))
    g_fused = np.real(np.fft.ifft2(np.fft.ifftshift(g_fused)))
    r_fused = np.real(np.fft.ifft2(np.fft.ifftshift(r_fused)))

    fused_img = np.stack((b_fused, g_fused, r_fused), axis=-1)

     # 裁剪对数域值以防指数爆炸
    min_clip, max_clip = clip_log_range
    LB_prime_real = np.clip(fused_img, min_clip, max_clip)

    # 回到线性域
    I_lin_channel = np.exp(LB_prime_real) - eps_log
    # clip to [0,1]
    # I_lin_channel = np.clip(I_lin_channel, 0.0, 1.0)
    I_lin_channel = (I_lin_channel - I_lin_channel.min()) / (I_lin_channel.max() - I_lin_channel.min())

    # 局部亮度/对比度回归：把 fused 的局部均值/方差匹配回 B_lin，避免偏移
    fused_matched = np.zeros_like(fused_img)
    for c in range(3):
        fused_matched[...,c] = local_match(I_lin_channel[...,c], B_lin[...,c], ksize=local_match_ksize)

    fused_matched = np.clip(fused_matched, 0.0, 1.0)

    return fused_matched

def post_process_and_blend(fused_lin, M_x_srgb, out_dir, imagename):
    """
    创建DC分量为零的摩尔纹并与融合结果混合。
    """
    # 加入背景无关的摩尔纹，DC为零创造摩尔纹
    img3_b = np.fft.fftshift(np.fft.fft2(M_x_srgb[:,:,0]))
    img3_g = np.fft.fftshift(np.fft.fft2(M_x_srgb[:,:,1]))
    img3_r = np.fft.fftshift(np.fft.fft2(M_x_srgb[:,:,2]))
    H, W = img3_b.shape
    img3_b[H//2, W//2] = 0
    img3_g[H//2, W//2] = 0
    img3_r[H//2, W//2] = 0

    img3_b_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(img3_b)))
    img3_g_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(img3_g)))
    img3_r_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(img3_r)))
  
    img3_spatial = np.stack((img3_b_spatial, img3_g_spatial, img3_r_spatial), axis=-1)

    img3 = np.clip(img3_spatial, 0, 1)
    img3_u8 = (img3 * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f'{imagename}_dc.png'), img3_u8)

    # 混合
    final_fused_lin = (0.8 * fused_lin) + (0.2 * img3)
    final_fused_lin = np.clip(final_fused_lin, 0, 1)
    
    return final_fused_lin
