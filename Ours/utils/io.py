# -*- coding: utf-8 -*-
import os

def find_by_stem(directory, stem, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """
    在指定目录中查找与给定 stem（文件名不含扩展名）匹配的文件。
    支持多种扩展名，返回第一个匹配的文件路径。
    """
    for e in exts:
        p = os.path.join(directory, stem + e)
        if os.path.isfile(p):
            return p
    return None
