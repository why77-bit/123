import cv2
import numpy as np
import os
from modules.background_estimation import estimate_bg_color
from modules.color_correction import color_correct_bg, compute_moire_texture
from modules.fusion import fuse_log_fft1, fuse_log_fft2
from utils.color import linear_to_srgb_arr, srgb_to_linear_arr
from modules.epsilon_enhancer import EpsilonEnhancer

def method1_function(img, bg_path, args):
    """
    这个方法就是最原始的背景估计 + 色彩矫正 + 融合
    """
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 背景估计
    bg_median_lin, bg_mean_lin = estimate_bg_color(
        img,
        k=args.k,
        lowpass_ksize=args.lowpass_ksize,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        save_dir=args.out_dir,
        image_name=imagename
    )
    # 数字底图色彩矫正
    sc, B_x_corr_line = color_correct_bg(bg_path, bg_median_lin,
        out_path=os.path.join(args.out_dir, f'{imagename}_color_corr.png')
    )
    # 计算背景无关摩尔纹
    M_x_line = compute_moire_texture(img, sc)
    M_x_srgb = linear_to_srgb_arr(M_x_line)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_M_x.png'),
                (M_x_srgb * 255).round().astype(np.uint8))
    # 融合
    fused_lin = fuse_log_fft1(
        B_x_corr_line, M_x_line,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )
    fused_srgb = linear_to_srgb_arr(fused_lin)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method2_function(img, bg_path, args):
    """
    该方法在method1的基础上增加了背景增强步骤
    """
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 背景估计
    bg_median_lin, bg_mean_lin = estimate_bg_color(
        img,
        k=args.k,
        lowpass_ksize=args.lowpass_ksize,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        save_dir=args.out_dir,
        image_name=imagename
    )
    # 数字底图色彩矫正
    sc, B_x_corr_line = color_correct_bg(bg_path, bg_median_lin,
        out_path=os.path.join(args.out_dir, f'{imagename}_color_corr.png')
    )
    # 计算背景无关摩尔纹
    M_x_line = compute_moire_texture(img, sc)
    M_x_srgb = linear_to_srgb_arr(M_x_line)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_M_x.png'),
                (M_x_srgb * 255).round().astype(np.uint8))
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        s=0.02,
        eps_min=1e-6,
        eps_max=5e-3,
        guided=True,
        guided_radius=9
    )
    B_x_corr_line_log, eps_map = eps_enhancer.enhance_background(B_x_corr_line)
    # 融合
    fused_lin = fuse_log_fft2(
        B_x_corr_line_log, M_x_line,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )

    fused_srgb = linear_to_srgb_arr(fused_lin)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method3_function(moire_img, bg_path, args):
    '''
    这个方法在method1的基础上去掉了色彩矫正步骤，
    直接使用摩尔层和背景乘法，以及加了一个背景增强步骤
    '''
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 读取背景图
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f'[跳过] 读取背景图失败: {bg_path}')
        return
    # sRGB -> 线性空间
    bg_img_lin = srgb_to_linear_arr(bg_img / 255.0)
    moire_img_lin = srgb_to_linear_arr(moire_img / 255.0)
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        thresh=0.01,
        s=0.02,
        eps_min=1e-6,
        eps_max=1e-2,
        guided=True,
        guided_radius=9
    )
    bg_img_log, eps_map, mask = eps_enhancer.enhance_background(bg_img_lin)
    
    # 可视化《增强矩阵值》
    eps_map_vis = (255 * (eps_map - eps_map.min()) / (eps_map.max() - eps_map.min())).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_eps_map.png'), eps_map_vis)
    
    # 可视化《mask》白色表示背景，黑色表示前景对吧？
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_mask.png'), mask_vis)
    
    # 可视化《增强后的背景图片》
    bg_img_enhanced = np.exp(bg_img_log)
    bg_img_enhanced = np.clip(bg_img_enhanced, 0, 1)
    bg_img_enhanced_srgb = linear_to_srgb_arr(bg_img_enhanced)
    bg_img_enhanced_u8 = (bg_img_enhanced_srgb * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_bg_enhanced.png'), bg_img_enhanced_u8)
    
    # 融合（用原始摩尔图和增强后的背景）
    fused_lin = fuse_log_fft2(
        bg_img_log, moire_img_lin,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )      
    fused_srgb = linear_to_srgb_arr(fused_lin)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method4_function(moire_img, bg_path, args):
    '''
    在合成前用gamma矫正
    实验记录：在融合之前进行增强，能够使结果在直方图上指标有上升，但其他指标均有下降
    '''
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 读取背景图
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f'[跳过] 读取背景图失败: {bg_path}')
        return
    # sRGB -> 线性空间
    bg_img_lin = srgb_to_linear_arr(bg_img / 255.0)
    moire_img_lin = srgb_to_linear_arr(moire_img / 255.0)
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        thresh=0.01,
        s=0.02,
        eps_min=1e-6,
        eps_max=1e-2,
        guided=True,
        guided_radius=9
    )
    bg_img_log, eps_map, mask = eps_enhancer.enhance_background(bg_img_lin)
    
    # 可视化《增强矩阵值》
    eps_map_vis = (255 * (eps_map - eps_map.min()) / (eps_map.max() - eps_map.min())).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_eps_map.png'), eps_map_vis)
    
    # 可视化《mask》白色表示背景，黑色表示前景对吧？
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_mask.png'), mask_vis)
    
    # 可视化《增强后的背景图片》
    bg_img_enhanced = np.exp(bg_img_log)
    bg_img_enhanced = np.clip(bg_img_enhanced, 0, 1)
    bg_img_enhanced_srgb = linear_to_srgb_arr(bg_img_enhanced)
    bg_img_enhanced_u8 = (bg_img_enhanced_srgb * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_bg_enhanced.png'), bg_img_enhanced_u8)

    # ----------- 背景对比度增强方法1：线性拉伸（归一化） -----------
    bg_img_norm = cv2.normalize(bg_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_contrast_linear.png'), bg_img_norm)

    # ----------- 对比度增强方法2：直方图均衡化（Y通道） -----------
    bg_img_yuv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2YUV)
    bg_img_yuv[:,:,0] = cv2.equalizeHist(bg_img_yuv[:,:,0])
    bg_img_hist = cv2.cvtColor(bg_img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_contrast_hist.png'), bg_img_hist)

    # ----------- 对比度增强方法3：CLAHE（LAB空间L通道） -----------
    lab = cv2.cvtColor(bg_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1080,1080))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    bg_img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_contrast_clahe.png'), bg_img_clahe)

    # ----------- 对比度增强方法4：Gamma校正 -----------
    gamma = 1.1
    bg_img_float = bg_img / 255.0
    bg_img_gamma = np.power(bg_img_float, gamma)
    bg_img_gamma_lin = srgb_to_linear_arr(bg_img_gamma)

    bg_img_gamma_u8 = np.clip(bg_img_gamma * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_contrast_gamma.png'), bg_img_gamma_u8)
    
    # 融合（用原始摩尔图和增强后的背景）
    fused_lin = fuse_log_fft1(
        bg_img_gamma_lin, moire_img_lin,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )      
    fused_srgb = linear_to_srgb_arr(fused_lin)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method5_function(moire_img, bg_path, args):
    '''
    融合后做gamama增强
    '''
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 读取背景图
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f'[跳过] 读取背景图失败: {bg_path}')
        return
    # sRGB -> 线性空间
    bg_img_lin = srgb_to_linear_arr(bg_img / 255.0)
    moire_img_lin = srgb_to_linear_arr(moire_img / 255.0)
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        thresh=0.01,
        s=0.02,
        eps_min=1e-6,
        eps_max=1e-2,
        guided=True,
        guided_radius=9
    )
    bg_img_log, eps_map, mask = eps_enhancer.enhance_background(bg_img_lin)
    
    # 可视化《增强矩阵值》
    eps_map_vis = (255 * (eps_map - eps_map.min()) / (eps_map.max() - eps_map.min())).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_eps_map.png'), eps_map_vis)
    
    # 可视化《mask》白色表示背景，黑色表示前景对吧？
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_mask.png'), mask_vis)
    
    # 可视化《增强后的背景图片》
    bg_img_enhanced = np.exp(bg_img_log)
    bg_img_enhanced = np.clip(bg_img_enhanced, 0, 1)
    bg_img_enhanced_srgb = linear_to_srgb_arr(bg_img_enhanced)
    bg_img_enhanced_u8 = (bg_img_enhanced_srgb * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_bg_enhanced.png'), bg_img_enhanced_u8)

    # 融合（用原始摩尔图和增强后的背景）
    fused_lin = fuse_log_fft2(
        bg_img_log, moire_img_lin,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )      

    # 对融合结果做gamma增强
    gamma = 1.1
    fused_lin_gamma = np.power(np.clip(fused_lin, 0, 1), gamma)
    fused_srgb = linear_to_srgb_arr(fused_lin_gamma)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method6_function(moire_img, bg_path, args):
    '''
    在3的基础上，加入前景亮度的增强
    。
    '''
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 读取背景图
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f'[跳过] 读取背景图失败: {bg_path}')
        return
    # sRGB -> 线性空间
    bg_img_lin = srgb_to_linear_arr(bg_img / 255.0)
    moire_img_lin = srgb_to_linear_arr(moire_img / 255.0)
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        thresh=0.01,
        s=0.02,
        eps_min=1e-6,
        eps_max=1e-2,
        guided=True,
        guided_radius=9
    )
    bg_img_log, eps_map, mask = eps_enhancer.enhance_background(bg_img_lin)
    
    # 可视化《增强矩阵值》
    eps_map_vis = (255 * (eps_map - eps_map.min()) / (eps_map.max() - eps_map.min())).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_eps_map.png'), eps_map_vis)
    
    # 可视化《mask》白色表示背景，黑色表示前景
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_mask.png'), mask_vis)
    
    # 可视化《增强后的背景图片》
    bg_img_enhanced = np.exp(bg_img_log)
    bg_img_enhanced = np.clip(bg_img_enhanced, 0, 1)
    bg_img_enhanced_srgb = linear_to_srgb_arr(bg_img_enhanced)
    bg_img_enhanced_u8 = (bg_img_enhanced_srgb * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_bg_enhanced.png'), bg_img_enhanced_u8)

    # 融合（用原始摩尔图和增强后的背景）
    fused_lin = fuse_log_fft2(
        bg_img_log, moire_img_lin,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )      

    # 利用mask对融合结果的前景亮度增强
    # mask==0为前景，mask==1为背景
    fused_lin_enhanced = fused_lin.copy()
    foreground_mask = (mask == 0)
    # 增强前景亮度，系数可调
    brightness_factor = 2.0
    fused_lin_enhanced[foreground_mask] = np.clip(fused_lin_enhanced[foreground_mask] * brightness_factor, 0, 1)

    # 转换为sRGB并保存
    fused_srgb = linear_to_srgb_arr(fused_lin_enhanced)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

def method7_function(moire_img, bg_path, args):
    '''
    在3的基础上，融合后进行前景亮度增强和gamma矫正
    '''
    imagename = os.path.splitext(os.path.basename(bg_path))[0]
    # 读取背景图
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        print(f'[跳过] 读取背景图失败: {bg_path}')
        return
    # sRGB -> 线性空间
    bg_img_lin = srgb_to_linear_arr(bg_img / 255.0)
    moire_img_lin = srgb_to_linear_arr(moire_img / 255.0)
    # 背景增强
    eps_enhancer = EpsilonEnhancer(
        tau=0.05,
        thresh=0.01,
        s=0.02,
        eps_min=1e-6,
        eps_max=1e-2,
        guided=True,
        guided_radius=9
    )
    bg_img_log, eps_map, mask = eps_enhancer.enhance_background(bg_img_lin)
    
    # 可视化《增强矩阵值》
    eps_map_vis = (255 * (eps_map - eps_map.min()) / (eps_map.max() - eps_map.min())).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_eps_map.png'), eps_map_vis)
    
    # 可视化《mask》白色表示背景，黑色表示前景
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_mask.png'), mask_vis)
    
    # 可视化《增强后的背景图片》
    bg_img_enhanced = np.exp(bg_img_log)
    bg_img_enhanced = np.clip(bg_img_enhanced, 0, 1)
    bg_img_enhanced_srgb = linear_to_srgb_arr(bg_img_enhanced)
    bg_img_enhanced_u8 = (bg_img_enhanced_srgb * 255.0).round().astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, f'{imagename}_bg_enhanced.png'), bg_img_enhanced_u8)

    # 融合（用原始摩尔图和增强后的背景）
    fused_lin = fuse_log_fft2(
        bg_img_log, moire_img_lin,
        eps_log=1e-3,
        freq_cutoff=0.25,
        freq_power=4.0,
        phase_thresh_ratio=1e-3,
        gamma_max=0.5,
        clip_log_range=(-10.0, 3.0),
        local_match_ksize=31
    )

    # 先做Gamma矫正（对整幅图）
    gamma = 1.1  # 可调：<1提亮，>1压暗
    fused_lin_gamma = np.power(np.clip(fused_lin, 0, 1), gamma)

    # 再做前景亮度增强（mask==0为前景）
    fused_lin_enhanced = fused_lin_gamma.copy()
    foreground_mask = (mask == 0)
    brightness_factor = 2.0
    fused_lin_enhanced[foreground_mask] = np.clip(
        fused_lin_enhanced[foreground_mask] * brightness_factor, 0, 1
    )

    # 转换为sRGB并保存最终结果
    fused_srgb = linear_to_srgb_arr(fused_lin_enhanced)
    fused_u8 = (fused_srgb * 255.0).round().astype(np.uint8)
    return fused_u8

