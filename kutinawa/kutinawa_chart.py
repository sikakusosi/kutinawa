"""
チャートCGを作成する関数群.
"""
import numpy as np


def rgb_to_bayer(target_rgb):
    out_bayer = np.zeros((np.shape(target_rgb)[0], np.shape(target_rgb)[1]))
    out_bayer[0::2, 0::2] = target_rgb[0::2, 0::2, 0]
    out_bayer[0::2, 1::2] = target_rgb[0::2, 1::2, 1]
    out_bayer[1::2, 0::2] = target_rgb[1::2, 0::2, 1]
    out_bayer[1::2, 1::2] = target_rgb[1::2, 1::2, 2]

    return out_bayer


def generate_random_array(size_taple,seed=123):
    np.random.seed(123)
    return np.random.random_sample(size_taple)

def generate_gaussian_random_array(size_taple,seed=123):
    np.random.seed(seed)
    return np.random.normal(size=size_taple)

def generate_CZP(img_size):
    '''
    CZPチャートを生成する
    :param img_size: 出力するCZPチャートのサイズ、タプル形式で(高、幅)、2の倍数にすること
    :return: ndarray形式のCZP画像(0-1)
    '''
    x_img = np.tile(np.arange(0, img_size[1]) - img_size[1] // 2, (img_size[0], 1))
    y_img = np.rot90(np.tile(np.arange(0, img_size[0]) - img_size[0] // 2, (img_size[1], 1)), 3)
    czp_img = (np.cos(np.pi * x_img * x_img / np.min(img_size) + np.pi * y_img * y_img / np.min(img_size))+1)/2.0

    return czp_img

def generate_gradation_chart(img_size):
    '''
    無彩色グラデ、有彩色グラデのチャートを作成する
    :param img_size: 高さ(2の倍数)、幅(24の倍数)
    :return: 0-1のグラデーション画像
    '''
    out_img = np.zeros((img_size[0],img_size[1],3))
    rgb_01 = np.array([[1, 1, 0, 0, 0, 1, 1],
                       [0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0]])

    out_img[:, :, 0] = np.tile(np.concatenate([np.linspace(rgb_01[0, 0], rgb_01[0, 1], img_size[1] // 6),
                                               np.linspace(rgb_01[0, 1], rgb_01[0, 2], img_size[1] // 6),
                                               np.linspace(rgb_01[0, 2], rgb_01[0, 3], img_size[1] // 6),
                                               np.linspace(rgb_01[0, 3], rgb_01[0, 4], img_size[1] // 6),
                                               np.linspace(rgb_01[0, 4], rgb_01[0, 5], img_size[1] // 6),
                                               np.linspace(rgb_01[0, 5], rgb_01[0, 6], img_size[1] // 6), ], axis=0),
                               (img_size[0], 1))
    # wa.imagesc(out_img[:,:,0] )
    out_img[:, :, 1] = np.tile(np.concatenate([np.linspace(rgb_01[1, 0], rgb_01[1, 1], img_size[1] // 6),
                                               np.linspace(rgb_01[1, 1], rgb_01[1, 2], img_size[1] // 6),
                                               np.linspace(rgb_01[1, 2], rgb_01[1, 3], img_size[1] // 6),
                                               np.linspace(rgb_01[1, 3], rgb_01[1, 4], img_size[1] // 6),
                                               np.linspace(rgb_01[1, 4], rgb_01[1, 5], img_size[1] // 6),
                                               np.linspace(rgb_01[1, 5], rgb_01[1, 6], img_size[1] // 6), ], axis=0),
                               (img_size[0], 1))
    # wa.imagesc(out_img[:,:,1] )
    out_img[:, :, 2] = np.tile(np.concatenate([np.linspace(rgb_01[2, 0], rgb_01[2, 1], img_size[1] // 6),
                                               np.linspace(rgb_01[2, 1], rgb_01[2, 2], img_size[1] // 6),
                                               np.linspace(rgb_01[2, 2], rgb_01[2, 3], img_size[1] // 6),
                                               np.linspace(rgb_01[2, 3], rgb_01[2, 4], img_size[1] // 6),
                                               np.linspace(rgb_01[2, 4], rgb_01[2, 5], img_size[1] // 6),
                                               np.linspace(rgb_01[2, 5], rgb_01[2, 6], img_size[1] // 6), ], axis=0),
                               (img_size[0], 1))

    # 上方向に減衰ゲイン
    out_img[:, :, 0] = out_img[:, :, 0] * np.rot90(
        np.tile(np.concatenate([np.linspace(1, 0, img_size[0] // 2), np.linspace(0, 0, img_size[0] // 2)], axis=0),
                (img_size[1], 1)))
    out_img[:, :, 1] = out_img[:, :, 1] * np.rot90(
        np.tile(np.concatenate([np.linspace(1, 0, img_size[0] // 2), np.linspace(0, 0, img_size[0] // 2)], axis=0),
                (img_size[1], 1)))
    out_img[:, :, 2] = out_img[:, :, 2] * np.rot90(
        np.tile(np.concatenate([np.linspace(1, 0, img_size[0] // 2), np.linspace(0, 0, img_size[0] // 2)], axis=0),
                (img_size[1], 1)))
    # wa.imagesc(out_img)

    # 無彩色グラデ追加
    out_img[0:img_size[0] // 2, :, 0] = out_img[0:img_size[0] // 2, :, 0] + np.rot90(
        np.tile(np.linspace(0, 1, img_size[0] // 2), (img_size[1], 1)))
    out_img[0:img_size[0] // 2, :, 1] = out_img[0:img_size[0] // 2, :, 1] + np.rot90(
        np.tile(np.linspace(0, 1, img_size[0] // 2), (img_size[1], 1)))
    out_img[0:img_size[0] // 2, :, 2] = out_img[0:img_size[0] // 2, :, 2] + np.rot90(
        np.tile(np.linspace(0, 1, img_size[0] // 2), (img_size[1], 1)))

    return out_img