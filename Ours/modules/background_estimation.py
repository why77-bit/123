# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from utils.color import srgb_to_linear_arr, linear_to_srgb_arr

#背景估计函数（加入线性中值）
def estimate_bg_color(img_bgr,
                    k=3,
                    lowpass_ksize=7,
                    morph_kernel=5,
                    min_area=200,
                    save_dir='result',
                    image_name='img.jpg'):
    """
    返回：
      final_mask (白=背景, 黑=moiré),
      labels_img,
      cluster_lab_means,
      cluster_sizes,
      bg_median_lin (线性 RGB, float 0..1),
      lp (低通图, BGR uint8)
    注意：bg_median_lin 已经是线性空间的中值（用于后续计算 s）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 低通突出低频彩色条带
    if lowpass_ksize > 1:
        lp = cv2.GaussianBlur(img_bgr, (lowpass_ksize, lowpass_ksize), 0)
    else:
        lp = img_bgr.copy()
    cv2.imwrite(os.path.join(save_dir, f'{image_name}_smooth.png'), lp)

    # 2) 转到 Lab（用于聚类）
    lab = cv2.cvtColor(lp, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    ab = lab[:, :, 1:3].reshape(-1, 2)  # 只用于聚类

    # 3) KMeans 在 ab 上聚类
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels_flat = km.fit_predict(ab)
    labels_img = labels_flat.reshape(h, w)

    # 4) 计算每簇的平均 Lab（三通道），用于与白色比较
    cluster_lab_means = np.zeros((k, 3), dtype=np.float32)
    cluster_sizes = np.zeros((k,), dtype=int)
    lab_flat = lab.reshape(-1, 3)
    for i in range(k):
        mask_i = (labels_flat == i)
        cluster_sizes[i] = int(mask_i.sum())
        if cluster_sizes[i] > 0:
            cluster_lab_means[i] = lab_flat[mask_i].mean(axis=0)
        else:
            cluster_lab_means[i] = np.array([0., 128., 128.], dtype=np.float32)

    # 5) 白色在 Lab 中位置
    white_lab = cv2.cvtColor(np.uint8([[[255,255,255]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0,0,:]

    # 6) 找到最接近白的簇作为背景
    dists_to_white = np.linalg.norm(cluster_lab_means - white_lab[None, :], axis=1)
    bg_label = int(np.argmin(dists_to_white))

    # 7) 构造摩尔 mask：非背景簇视为摩尔（255）
    mask = (labels_img != bg_label).astype(np.uint8) * 255

    # 8) 形态学清理 & 去小连通域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, comps, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[comps == i] = 255
    mask_moire = clean
    final_mask = np.where(mask_moire==255, 0, 255).astype(np.uint8)  # 0 表示 moiré（黑色），255 表示背景（白色）
    cv2.imwrite(os.path.join(save_dir, f'{image_name}_mask.png'), final_mask) #保存mask

    #这部分是可视化摩尔纹图扣掉摩尔纹后剩余的图片
    # bg_only = lp.copy()                # 用低通图或用 img.copy() 看不同效果
    # bg_only[final_mask == 255] = (0,0,0)   # 把 moiré 部分设为白（BGR）
    # cv2.imwrite(os.path.join(save_dir, f'{image_name}_bg_only_white.png'), bg_only)

    # ----- 背景像素 (final_mask==255 为背景) -----#这里有争议
    bg_pixels_bgr = lp[final_mask == 0]  # BGR uint8 array (N,3)
    if bg_pixels_bgr.size == 0:
        raise RuntimeError("没有检测到背景像素，请检查 mask 或参数。")

    # 转到 sRGB 0..1，再到线性
    bg_pixels_srgb = (bg_pixels_bgr.astype(np.float32) / 255.0)
    bg_pixels_lin = srgb_to_linear_arr(bg_pixels_srgb)  # 线性空间

    # 用中值法估计 b（线性空间）
    bg_median_lin = np.median(bg_pixels_lin, axis=0)   # float, B,G,R in linear 0..1
    bg_mean_lin = bg_pixels_lin.mean(axis=0)

    # 同时保存 mean/madian 图像（以便可视化，但注意文件是 sRGB）
    bg_median_srgb = linear_to_srgb_arr(bg_median_lin)
    bg_median_u8 = (bg_median_srgb*255.0).round().astype(np.uint8)
    bg_mean_srgb = linear_to_srgb_arr(bg_mean_lin)
    bg_mean_u8 = (bg_mean_srgb*255.0).round().astype(np.uint8)
    #保存
    bg_mean_img = np.tile(bg_mean_u8.reshape(1,1,3), (h, w, 1))
    bg_median_img = np.tile(bg_median_u8.reshape(1,1,3), (h, w, 1))
    # cv2.imwrite(os.path.join(save_dir, f'{image_name}_mean.png'), bg_mean_img)
    cv2.imwrite(os.path.join(save_dir, f'{image_name}_median.png'), bg_median_img) #保存中值背景颜色图
    cv2.imwrite(os.path.join(save_dir, f'{image_name}_mean.png'), bg_mean_img) #保存均值背景颜色图
    # 打印信息（线性空间）
    print('---- 背景估计（线性空间） ----')
    print('bg_median_lin (B,G,R) =', bg_median_lin.tolist()) 

    return bg_median_lin, bg_mean_lin
