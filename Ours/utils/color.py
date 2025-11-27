# -*- coding: utf-8 -*-
import numpy as np

# sRGB -> line RGB
def srgb_to_linear_arr(srgb):
    """
    srgb: float array in 0..1, shape (...,3)
    返回 linear rgb 同样 shape
    """
    a = 0.055
    srgb = np.clip(srgb, 0.0, 1.0)
    mask = (srgb <= 0.04045).astype(np.float32)
    linear = srgb / 12.92 * mask + (((srgb + a) / (1.0 + a)) ** 2.4) * (1.0 - mask)
    return np.clip(linear, 0.0, 1.0)

# line RGB -> sRGB
def linear_to_srgb_arr(linear):
    """
    linear: float array in 0..1
    返回 sRGB 0..1
    """
    a = 0.055
    linear = np.clip(linear, 0.0, 1.0)
    mask = (linear <= 0.0031308).astype(np.float32)
    srgb = linear * 12.92 * mask + ((1.0 + a) * (linear ** (1.0 / 2.4)) - a) * (1.0 - mask)
    return np.clip(srgb, 0.0, 1.0)
