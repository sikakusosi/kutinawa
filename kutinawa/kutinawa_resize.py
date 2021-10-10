"""
画像の拡大縮小、並べ替えの関数群.
"""
import numpy as np
from scipy import ndimage

def binning_2x2(target_img):
    """
    説明------------------------------------
    2x2の範囲でビニング
    引数------------------------------------
    target_img　:　ビニングしたい画像(ndarray)
    戻り値----------------------------------
    ビニングされた画像(ndarray)
    """
    return (target_img[0::2,0::2]+target_img[0::2,1::2]+target_img[1::2,0::2]+target_img[1::2,1::2])/4.0

def mag_x2_bilinear_CP_binning_2x2(target_img):
    """
    説明------------------------------------
    binning_2x2 の対になる、拡大
    binning前と重心位置が合うように拡大する
    引数------------------------------------
    target_img　:　拡大したい画像(ndarray)
    戻り値----------------------------------
    バイリニア拡大画像(ndarray)
    """
    out_img = np.zeros((np.shape(target_img)[0]*2,np.shape(target_img)[1]*2))
    fil_bil = np.array([[ 0,         0,         0],
                        [ 0, 0.75*0.75, 0.75*0.25],
                        [ 0, 0.25*0.75, 0.25*0.25]])
    out_img[0::2,0::2] = ndimage.convolve(target_img,fil_bil,mode='mirror')
    out_img[0::2,1::2] = ndimage.convolve(target_img,np.rot90(fil_bil,3),mode='mirror')
    out_img[1::2,0::2] = ndimage.convolve(target_img,np.rot90(fil_bil,1),mode='mirror')
    out_img[1::2,1::2] = ndimage.convolve(target_img,np.rot90(fil_bil,2),mode='mirror')

    return out_img

def shrink_NN_grid(target_img, grid_size):
    target_img_h, target_img_w = np.shape(target_img)
    s_h = np.arange(0, target_img_h + 1, grid_size)
    s_w = np.arange(0, target_img_w + 1, grid_size)

    temp_img = np.array([np.sum(target_img[hh:hh + grid_size, :], axis=0) for hh in s_h[0:-1]])
    return (np.array([np.sum(temp_img[:, ww:ww + grid_size], axis=1) for ww in s_w[0:-1]]).T) / (grid_size * grid_size)


def mag_NN(target_img,mag_rate):
    """
    ニアレストネイバー拡大

    :param target_img:  拡大したい画像(ndarray)
    :param mag_rate:    拡大率(整数のみ)
    :return:            拡大された画像(ndarray)
    """
    return np.repeat(np.repeat(target_img,mag_rate,axis=0),mag_rate,axis = 1)


def raster_2d_to_1d(target_img):
    '''
    2次元配列で表現されている画像をラスタスキャン順に1次元にする
    :param target_img: 2次元ndarray配列(高、幅)
    :return: 1次元ndarray配列
    '''
    return np.squeeze(np.reshape(target_img, [np.shape(target_img)[1] * np.shape(target_img)[0], 1]))

