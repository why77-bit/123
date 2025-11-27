# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from utils.io import find_by_stem
from pipelines.method import method1_function, method2_function, method3_function, \
method4_function, method5_function, method6_function, method7_function
from utils.eval import evaluate_results  # 新增评价函数

# ------------------ 图像处理主流程 ------------------
def process_images(args):
    # 1. 收集待处理图片
    
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    moire_files = [f for f in sorted(os.listdir(args.moire_dir)) if f.lower().endswith(exts)]
    if not moire_files:
        print(f'在 {args.moire_dir} 未找到图片文件，已退出。')
        sys.exit(0)

    # 统计配对和跳过信息
    total = len(moire_files)
    not_found_bg = []
    failed_read = []
    # 仅做轻量检查：背景文件是否存在，摩尔图文件是否非空（避免重复用 cv2.imread）
    for fname in moire_files:
        stem, _ = os.path.splitext(fname)
        bg_stem = stem.replace('_mo', '_src')
        bg_fname = bg_stem + '.png'
        bg_path = os.path.join(args.bg_dir, bg_fname)
        if not os.path.exists(bg_path):
            not_found_bg.append(fname)
            continue
        moire_path = os.path.join(args.moire_dir, fname)
        try:
            # 快速判断文件是否可读（非空）——比 cv2.imread 快且不需解析图像
            if os.path.getsize(moire_path) == 0:
                failed_read.append(fname)
        except Exception:
            failed_read.append(fname)
    valid_pairs = total - len(not_found_bg) - len(failed_read)
    # 可选：打印验证结果，帮助定位“卡住”原因
    print(f"候选数: {total}, 可处理对数(估计): {valid_pairs}, 缺少底图: {len(not_found_bg)}, 无法读取文件: {len(failed_read)}")

    # 日志内容准备
    from datetime import datetime
    log_lines = []
    log_lines.append(f"处理批次时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"输出目录: {os.path.join(args.out_dir, args.method)}")
    log_lines.append(f"共发现 {total} 个候选摩尔图。")
    log_lines.append(f"预计处理 {valid_pairs} 对图片。")
    if not_found_bg:
        log_lines.append(f"有 {len(not_found_bg)} 张摩尔图未找到对应底图，跳过：")
        log_lines.append(', '.join(not_found_bg))
    if failed_read:
        log_lines.append(f"有 {len(failed_read)} 张摩尔图读取失败，跳过：")
        log_lines.append(', '.join(failed_read))
    log_lines.append('正式开始处理...')
    log_lines.append('-' * 60)

    # 写入处理前日志
    from datetime import datetime
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    method_folder = os.path.join(args.out_dir, f"{now_str}_{args.method}")
    os.makedirs(method_folder, exist_ok=True)  # 确保目录存在
    args.out_dir = method_folder  # 更新输出目录参数
    eval_log_path = os.path.join(method_folder, "eval.log")
    with open(eval_log_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    # 2. 创建输出文件夹
    os.makedirs(method_folder, exist_ok=True)

    # 3. 获取处理方法
    method_func_map = {
        "method1_function": method1_function,
        "method2_function": method2_function,
        "method3_function": method3_function,
        "method4_function": method4_function,
        "method5_function": method5_function,
        "method6_function": method6_function,
        "method7_function": method7_function,
    }
    method_func = method_func_map.get(args.method, method7_function)

    # 4. 逐张处理并保存结果
    print("Start processing images")
    for fname in tqdm(moire_files, desc="合成图片进度"):
        stem, _ = os.path.splitext(fname)
        bg_stem = stem.replace('_mo', '_src')
        bg_fname = bg_stem + '.png'
        bg_path = os.path.join(args.bg_dir, bg_fname)
        if not os.path.exists(bg_path):
            continue
        moire_path = os.path.join(args.moire_dir, fname)
        moire_img = cv2.imread(moire_path)
        if moire_img is None:
            continue
        args._save_dir = method_folder
        result_img = method_func(moire_img, bg_path, args)
        # 保存为与GT一致的文件名
        result_stem = stem.replace('_mo', '_cap')

        result_folder = os.path.join(args.out_dir, "results")
        os.makedirs(result_folder, exist_ok=True)  # 确保目录存在
        out_path = os.path.join(result_folder, f"{result_stem}.png")
        cv2.imwrite(out_path, result_img)

    print("Image processing finished")

    # 5. 评价结果并写日志
    if hasattr(args, 'gt_dir') and args.gt_dir:
        # 统计评价配对信息
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        result_files = [f for f in sorted(os.listdir(method_folder)) if f.lower().endswith(exts)]
        gt_files = set(os.listdir(args.gt_dir))
        eval_candidates = [f for f in result_files if f in gt_files]
        missing_gt = [f for f in result_files if f not in gt_files]

        eval_result = evaluate_results(method_folder, args.gt_dir)
        eval_log_lines = []
        from datetime import datetime
        eval_log_lines.append(f"评测批次时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        eval_log_lines.append(f"输出目录: {method_folder}")
        eval_log_lines.append(f"预计评价图片数: {len(result_files)}")
        eval_log_lines.append(f"可配对评价图片数: {len(eval_candidates)}")
        if missing_gt:
            eval_log_lines.append(f"有 {len(missing_gt)} 张结果图片未找到对应GT，跳过：")
            eval_log_lines.append(', '.join(missing_gt))
        if eval_result is not None:
            eval_log_lines.append(f"实际成功评价图片数: {eval_result['count']}")
            if eval_result['failed']:
                eval_log_lines.append(f"有 {len(eval_result['failed'])} 张图片评价失败（尺寸不一致或读取失败）：")
                eval_log_lines.append(', '.join(eval_result['failed']))
            eval_log_lines.append(
                f"PSNR: {eval_result['psnr_mean']:.2f} | SSIM: {eval_result['ssim_mean']:.4f} | HistSim: {eval_result['hist_sim_mean']:.4f} | LPIPS: {eval_result['lpips_mean']:.4f} | log-PSD L2: {eval_result['psd_l2_mean']:.4f}"
            )
            for item in eval_result['details']:
                if "error" in item:
                    eval_log_lines.append(f"{item['fname']}\t{item['error']}")
                else:
                    eval_log_lines.append(
                        f"{item['fname']}\tPSNR:{item['psnr']:.2f}\tSSIM:{item['ssim']:.4f}\tHistSim:{item['hist_sim']:.4f}\tLPIPS:{item['lpips']:.4f}\tlogPSD_L2:{item['psd_l2']:.4f}"
                    )
            eval_log_lines.append('-' * 60)
            with open(eval_log_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(eval_log_lines) + '\n')
            print(f"已追加评价结果到日志: {eval_log_path}")

# ------------------ 程序入口 ------------------
if __name__ == '__main__':
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理：按相同文件名配对摩尔图与数字底图')
    parser.add_argument('--moire_dir', default='/root/autodl-tmp/DDI-test/mo', help='摩尔图目录（拍摄图）')
    parser.add_argument('--bg_dir', default='/root/autodl-tmp/DDI-test/src', help='数字底图目录（背景图）')
    parser.add_argument('--out_dir', default='/root/autodl-tmp/code/Ours/result', help='输出目录')
    parser.add_argument('--gt_dir', default='/root/autodl-tmp/DDI-test/cap', help='GT图片目录（用于评价）')
    parser.add_argument('--method', default='method7_function', help='处理方法名')
    parser.add_argument('--k', type=int, default=3, help='KMeans 聚类的簇数')
    parser.add_argument('--lowpass_ksize', type=int, default=7, help='低通滤波核大小')
    parser.add_argument('--morph_kernel', type=int, default=5, help='形态学操作核大小')
    parser.add_argument('--min_area', type=int, default=200, help='最小连通域面积')
    parser.add_argument('--eval_only', default=False, help='只运行评价，不处理图片')
    args = parser.parse_args()

    # 2. 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. 只评价模式或完整处理流程
    if args.eval_only:
        from datetime import datetime
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        method_folder = os.path.join(args.out_dir, f"{now_str}_{args.method}")
        os.makedirs(method_folder, exist_ok=True)  # 确保目录存在
        eval_log_path = os.path.join(method_folder, "eval.log")
        # 统计评价配对信息
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        result_files = [f for f in sorted(os.listdir(method_folder)) if f.lower().endswith(exts)]
        gt_files = set(os.listdir(args.gt_dir))
        eval_candidates = [f for f in result_files if f in gt_files]
        missing_gt = [f for f in result_files if f not in gt_files]

        eval_result = evaluate_results(method_folder, args.gt_dir)
        eval_log_lines = []
        from datetime import datetime
        eval_log_lines.append(f"评测批次时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        eval_log_lines.append(f"输出目录: {method_folder}")
        eval_log_lines.append(f"预计评价图片数: {len(result_files)}")
        eval_log_lines.append(f"可配对评价图片数: {len(eval_candidates)}")
        if missing_gt:
            eval_log_lines.append(f"有 {len(missing_gt)} 张结果图片未找到对应GT，跳过：")
            eval_log_lines.append(', '.join(missing_gt))
        if eval_result is not None:
            eval_log_lines.append(f"实际成功评价图片数: {eval_result['count']}")
            if eval_result['failed']:
                eval_log_lines.append(f"有 {len(eval_result['failed'])} 张图片评价失败（尺寸不一致或读取失败）：")
                eval_log_lines.append(', '.join(eval_result['failed']))
            eval_log_lines.append(
                f"PSNR: {eval_result['psnr_mean']:.2f} | SSIM: {eval_result['ssim_mean']:.4f} | HistSim: {eval_result['hist_sim_mean']:.4f} | LPIPS: {eval_result['lpips_mean']:.4f} | log-PSD L2: {eval_result['psd_l2_mean']:.4f}"
            )
            for item in eval_result['details']:
                if "error" in item:
                    eval_log_lines.append(f"{item['fname']}\t{item['error']}")
                else:
                    eval_log_lines.append(
                        f"{item['fname']}\tPSNR:{item['psnr']:.2f}\tSSIM:{item['ssim']:.4f}\tHistSim:{item['hist_sim']:.4f}\tLPIPS:{item['lpips']:.4f}\tlogPSD_L2:{item['psd_l2']:.4f}"
                    )
            eval_log_lines.append('-' * 60)
            with open(eval_log_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(eval_log_lines) + '\n')
            print(f"已追加评价结果到日志: {eval_log_path}")
    else:
        process_images(args)