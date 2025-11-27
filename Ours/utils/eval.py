import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
def log_psd_l2_gray(gt_img, result_img):
    """
    灰度图log-PSD L2距离
    """
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    res_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    gt_gray = gt_gray.astype(np.float32) / 255.0
    res_gray = res_gray.astype(np.float32) / 255.0
    hann_y = np.hanning(gt_gray.shape[0])
    hann_x = np.hanning(gt_gray.shape[1])
    window = np.outer(hann_y, hann_x)
    gt_gray *= window
    res_gray *= window
    def log_psd(img):
        F = np.fft.fftshift(np.fft.fft2(img))
        psd = np.abs(F) ** 2
        log_psd = np.log1p(psd)
        log_psd /= np.linalg.norm(log_psd) + 1e-12
        return log_psd
    gt_psd = log_psd(gt_gray)
    res_psd = log_psd(res_gray)

    l2 = np.linalg.norm(gt_psd - res_psd)
    l2_norm = l2 / (np.linalg.norm(gt_psd) + 1e-12)
    
    return l2_norm

def log_psd_l2_color(gt_img, result_img):
    """
    计算两张图的 log-PSD L2 距离
    对小平移不敏感。图像尺寸需相同。
    """
    # --- 逐通道计算二维log-PSD L2距离 ---
    l2_list = []
    for ch in range(3):
        gt_ch = gt_img[:,:,ch].astype(np.float32) / 255.0
        res_ch = result_img[:,:,ch].astype(np.float32) / 255.0
        # Hann窗减少边缘泄漏
        hann_y = np.hanning(gt_ch.shape[0])
        hann_x = np.hanning(gt_ch.shape[1])
        window = np.outer(hann_y, hann_x)
        gt_ch *= window
        res_ch *= window
        # 计算FFT并取对数功率谱
        def log_psd(img):
            F = np.fft.fftshift(np.fft.fft2(img))
            psd = np.abs(F) ** 2
            # log_psd = np.log1p(psd)
            log_psd = psd
            log_psd /= np.linalg.norm(log_psd) + 1e-12
            return log_psd
        gt_psd = log_psd(gt_ch)
        res_psd = log_psd(res_ch)
        # 直接对二维log-PSD做L2距离
        l2 = np.linalg.norm(gt_psd - res_psd)
        l2_norm = l2 / (np.linalg.norm(gt_psd) + 1e-12)
        l2_list.append(l2_norm)
    # 求三通道平均
    return np.mean(l2_list)

def evaluate_results(result_dir, gt_dir):
    import lpips
    lpips_model = lpips.LPIPS(net='alex')
    print("LPIPS loaded")
    # 支持的图片扩展名
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    # 获取结果目录下所有图片文件
    result_files = [f for f in sorted(os.listdir(result_dir)) if f.lower().endswith(exts)]
    gt_files = set(os.listdir(gt_dir))
    # 自动将摩尔图名映射为GT名
    gt_map = {f: f.replace('_mo', '_cap') for f in result_files}
    # 初始化各项指标列表
    psnr_list, ssim_list, hist_sim_list, lpips_list, psd_l2_list = [], [], [], [], []
    eval_fnames = []
    failed_fnames = []
    count = 0
    for fname in result_files:
        gt_fname = gt_map.get(fname, fname)
        if gt_fname not in gt_files:
            details.append({"fname": fname, "error": "未找到GT"})
            continue
        # 读取结果图片和GT图片
        result_img = cv2.imread(os.path.join(result_dir, fname))
        gt_img = cv2.imread(os.path.join(gt_dir, gt_fname))
        # 跳过无法读取或尺寸不一致的图片
        if result_img is None or gt_img is None or result_img.shape != gt_img.shape:
            failed_fnames.append(fname)
            continue
        # 计算PSNR和SSIM
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        psnr_val = psnr(gt_rgb, result_rgb, data_range=255)
        ssim_val = ssim(gt_rgb, result_rgb, data_range=255, channel_axis=-1)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        # 计算每个通道的直方图交集相似度
        sim = 0
        for ch in range(3):
            hist1 = cv2.calcHist([result_img], [ch], None, [256], [0,256])
            hist2 = cv2.calcHist([gt_img], [ch], None, [256], [0,256])
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            sim += cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        hist_sim_list.append(sim / 3)
        # 计算LPIPS感知距离
        
        def img_to_tensor(img):
            # BGR转RGB，归一化到[0,1]，再归一化到[-1,1]，转换为torch tensor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
            img = img * 2 - 1
            return img
        t1 = img_to_tensor(result_img)
        t2 = img_to_tensor(gt_img)
        lpips_val = lpips_model(t1, t2).item()
        lpips_list.append(lpips_val)

        psd_val = log_psd_l2_gray(gt_img, result_img)
        psd_l2_list.append(psd_val)

        eval_fnames.append(fname)
        count += 1
    if count == 0:
        print("未找到可评价的图片对")
        return None

    # 计算均值
    psnr_mean = np.mean(psnr_list)
    ssim_mean = np.mean(ssim_list)
    hist_sim_mean = np.mean(hist_sim_list)
    lpips_mean = np.mean(lpips_list)
    psd_l2_mean = np.mean(psd_l2_list)

    # 构造详细结果列表
    details = []
    idx = 0
    for fname in result_files:
        if fname not in gt_files:
            details.append({"fname": fname, "error": "未找到GT"})
            continue
        if fname in eval_fnames:
            details.append({
                "fname": fname,
                "psnr": psnr_list[idx],
                "ssim": ssim_list[idx],
                "hist_sim": hist_sim_list[idx],
                "lpips": lpips_list[idx],
                "psd_l2": psd_l2_list[idx]
            })
            idx += 1
        elif fname in failed_fnames:
            details.append({"fname": fname, "error": "评价失败（尺寸不一致或读取失败）"})
        else:
            details.append({"fname": fname, "error": "未知原因失败"})

    # 返回所有均值和详细结果
    return {
        "count": count,
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
        "hist_sim_mean": hist_sim_mean,
        "lpips_mean": lpips_mean,
        "psd_l2_mean": psd_l2_mean,
        "details": details,  # 这里就是每张图片的详细评价结果
        "failed": failed_fnames
    }


