import cv2
import pandas as pd
import numpy as np
import os


def get_patch_vector(image: np.ndarray) -> np.ndarray:
    """
    传入图片或者sliding window在检测图片中滑到的位置
    返回代表对应图片位置的RGB颜色特征和位置特征的向量
    """
    image_x = image.shape[0]  # 图片垂直方向长度
    image_y = image.shape[1]  # 图片水平方向长度
    # 将图片分解为3×3的patches以减少计算复杂度
    # 对不能被3整除的边进行padding填充, 填充方式为复制边界的像素
    if not image_x % 3 == 0 and image_y % 3 == 0:
        x_padding = image_x % 3
        y_padding = image_x % 3
        image_padded = cv2.copyMakeBorder(image, 0, x_padding, 0, y_padding, cv2.BORDER_REPLICATE)

    pass


def get_distance_matrix(appearance_vector, location_vector, parameter_lambda):
    """计算模板与sliding window的距离矩阵"""
    pass
