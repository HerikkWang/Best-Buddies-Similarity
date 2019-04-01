import cv2
import numpy as np


def get_patches_vector(image: np.ndarray, patch_size: int=3) -> np.ndarray:
    """
    :param image: sliding window或template对应的图像
    :param patch_size: 用于图片patch分割的尺寸
    :return: 返回代表对应图片位置的RGB颜色特征和位置特征的向量
    """
    image_x = image.shape[0]  # 图片垂直方向长度
    image_y = image.shape[1]  # 图片水平方向长度
    img = image.copy()
    # 将图片分解为3×3的patches以减少计算复杂度
    # 对不能被3整除的边进行padding填充, 填充方式为复制边界的像素
    x_padding = patch_size - image_x % patch_size
    y_padding = patch_size - image_y % patch_size

    if x_padding == patch_size:
        x_padding = 0
    if y_padding == patch_size:
        y_padding = 0

    if not (x_padding == 0 and y_padding == 0):
        # 对图片的bottom, right方向进行填充
        img = cv2.copyMakeBorder(img, 0, x_padding, 0, y_padding, cv2.BORDER_REPLICATE)

    patches_vector = []

    for x in range(0, img.shape[0], patch_size):
        for y in range(0, img.shape[1], patch_size):
            # 提取patch中心像素点的坐标(x, y)
            location_x = x + int(patch_size / 2)
            location_y = y + int(patch_size / 2)
            # 提取patch所有像素点的颜色特征(RGB)3×patch_size×patch_size
            appearance_vector = img[x:x + patch_size, y:y + patch_size].reshape(3 * patch_size * patch_size)
            patch_vector = np.append(np.array([location_x, location_y]), appearance_vector)
            patches_vector.append(patch_vector)

    patches_vector = np.array(patches_vector)

    return patches_vector


def get_distance_matrix(vector1: np.ndarray, vector2: np.ndarray, parameter_lambda: int=2) -> np.ndarray:
    """
    计算模板与sliding window的距离矩阵
    :param vector1: 模板的特征向量表示
    :param vector2: sliding window的特征向量表示
    :param parameter_lambda: 用于控制位置特征与颜色特征之间的权重关系
    :return: 返回模板与sliding window的距离矩阵
    """
    distance_matrix = np.zeros((len(vector1), len(vector2)))
    for x in range(0, distance_matrix.shape[0]):
        for y in range(0, distance_matrix.shape[1]):
            distance_matrix[x, y] = parameter_lambda * np.sum((vector1[x, :2] - vector2[y, :2]) ** 2) +\
                np.sum((vector1[x, 2:] - vector2[y, 2:]) ** 2)

    return distance_matrix


def bbs_calculator(distance_matrix: np.ndarray) -> float:
    """
    计算模板与sliding window之间的best-buddies similarity
    :param distance_matrix: distance matrix between template and sliding window
    :return: best-buddies similarity
    """
    # 计算distance_matrix两个维度上, 即每行每列中最大值的位置
    x_nearest_neighbors = np.argmin(distance_matrix, axis=0)
    y_nearest_neighbors = np.argmin(distance_matrix, axis=1)

    # 将两个维度上的最大值所在的矩阵坐标添加到两个集合中, 对两个集合取交集即可计算出模板与sliding window之间的BBS相似度
    x_nearest_coordinates = set([x_nearest_neighbors[i], i] for i in range(len(x_nearest_neighbors)))
    y_nearest_coordinates = set([i, y_nearest_neighbors[i]] for i in range(len(y_nearest_neighbors)))

    normalized_coefficient = np.min(distance_matrix.shape)
    bbs = normalized_coefficient * len(x_nearest_coordinates & y_nearest_coordinates)

    return bbs
