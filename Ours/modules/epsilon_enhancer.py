# -*- coding: utf-8 -*-
import numpy as np
import cv2
from utils.color import srgb_to_linear_arr, linear_to_srgb_arr

class EpsilonEnhancer:
    """
    用于根据颜色能量自适应增强背景的 ε mask 生成器。
    公式：
        M(x) = (R^2 + G^2 + B^2)/3
        F(M) = 1 - sigmoid((M - tau)/s)
        eps(x) = eps_min + (eps_max - eps_min)*F(M)
    """

    def __init__(self, tau=0.05, thresh=0.01, s=0.02, eps_min=1e-6, eps_max=5e-3,
                 guided=True, guided_radius=9, guided_eps=1e-6):
        self.tau = tau
        self.thresh = thresh
        self.s = s
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.guided = guided
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def _guided_filter(self, I, p, r, eps):
        """简易 guided filter（单通道）"""
        I = I.astype(np.float32)
        p = p.astype(np.float32)
        mean_I = cv2.boxFilter(I, -1, (r, r))
        mean_p = cv2.boxFilter(p, -1, (r, r))
        mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))

        q = mean_a * I + mean_b
        return q

    def compute_epsilon_mask(self, bg_linear):
        """
        bg_linear: 背景图（线性RGB，0..1）
        返回: eps_map (HxW), mask (HxW, bool)
        mask: 能量分割，M < tau 为背景（True），否则为前景（False）
        """
        # Step 1. 计算颜色能量 M(x)
        M = np.sum(bg_linear ** 2, axis=2) / 3.0  # [0,1]

        # Step 2. Sigmoid 映射
        F = 1.0 - 1.0 / (1.0 + np.exp(-(M - self.tau) / self.s))

        # Step 3. 计算 ε mask
        eps_map = self.eps_min + (self.eps_max - self.eps_min) * F

        # Step 4. Guided filter 平滑（避免频谱伪影）
        if self.guided:
            guide_gray = cv2.cvtColor(
            (linear_to_srgb_arr(bg_linear) * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            eps_map = self._guided_filter(
                guide_gray, eps_map, r=self.guided_radius, eps=self.guided_eps
            )

        # Step 5. 能量分割生成mask
        mask = (M < self.thresh)  # 背景为True，前景为False
        
        return eps_map, mask

    def enhance_background(self, bg_linear):
        """
        在进入频域合成之前，对背景加上 eps_map。
        返回: log(B+eps_map) 用于后续融合
        """
        eps_map, mask = self.compute_epsilon_mask(bg_linear)
        enhanced = np.log(bg_linear + eps_map[..., None])
        return enhanced, eps_map, mask
