"""
試作関数置き場。
テストとかしてないし、そのうち消すかもしれない。
作者がいいように使う部分。
"""

import numpy as np
from scipy import ndimage
from .kutinawa_filter import image_stack
import random

def nearly_image(tgt_img,rh,rw):
    """
    着目画素を中心として、指定半径内で最も値が近い画素とスワップ下画像を出力する
    ぱっと見は、入力画像とほとんど一緒の画像が出力されるが、孤立点とかは消える。
    :param tgt_img:
    :param rh:
    :param rw:
    :return:
    """
    stack_img = image_stack(tgt_img,rh,rw)
    stack_diff_img = image_stack(tgt_img,rh,rw)-np.tile(tgt_img[:,:,np.newaxis],(1,1,(rh*2+1)*(rw*2+1)))
    stack_diff_img[:,:,(rh*2+1)*(rw*2+1)//2] = 9999
    min_index = np.argmin(np.abs(stack_diff_img),axis=2)

    nearly_img = np.zeros_like(tgt_img)
    for i in np.arange((rh*2+1)*(rw*2+1)):
        nearly_img = nearly_img + stack_img[:,:,i]*(min_index==i)
    return nearly_img


def order_img(tgt_img,rh,rw):
    """
    着目画素が、指定半径内で何番目に小さい画素値か出力する
    :param tgt_img:
    :param rh:
    :param rw:
    :return:
    """
    argsort_img = np.argsort(image_stack(tgt_img, rh, rw), axis=2)
    temp_img = argsort_img == (rh * 2 + 1) * (rw * 2 + 1) // 2
    order_img = np.zeros_like(tgt_img)
    for i in np.arange((rh*2+1)*(rw*2+1)):
        order_img = order_img + temp_img[:,:,i]*i
    return order_img


def generate_alone_defect_pix_map__low_precision(img_size, non_overlap_area, defect_pix_per=0.01, random_seed=123):
    random.seed(random_seed)
    target_array = np.arange(img_size[0]*img_size[1]).tolist()
    random.shuffle(target_array)
    target_2darray = np.reshape(np.array(target_array),img_size)
    defect_map_pre = target_2darray<np.floor(img_size[0]*img_size[1]*defect_pix_per)
    defect_map_del = defect_map_pre*(ndimage.convolve(defect_map_pre.astype(int),non_overlap_area)<=1)
    return defect_map_del



