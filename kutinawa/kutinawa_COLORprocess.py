"""
画像の色変換に関する関数群.
RGB <-> YUV や WBを集約する.
"""
import numpy as np

######################################################################################################################## RGB <-> YC
def matrix3x3_x_img3ch(target_img, input_mat):
    """
    下記のような行列計算を行う
    /out0\   /imput_mat0 imput_mat1 imput_mat2\   /img0\
    |out1| = |imput_mat3 imput_mat4 imput_mat5| . |img1|
    \out2/   \imput_mat6 imput_mat7 imput_mat8/   \img2/
    :param target_img:
    :param input_mat:
    :return:
    """
    out_img = np.zeros_like(target_img)
    out_img[:,:,0] = target_img[:,:,0] * input_mat[0] + target_img[:, :, 1] * input_mat[1] + target_img[:, :, 2] * input_mat[2]
    out_img[:,:,1] = target_img[:,:,0] * input_mat[3] + target_img[:, :, 1] * input_mat[4] + target_img[:, :, 2] * input_mat[5]
    out_img[:,:,2] = target_img[:,:,0] * input_mat[6] + target_img[:, :, 1] * input_mat[7] + target_img[:, :, 2] * input_mat[8]
    return out_img

def rgb_to_yuv(target_img):
    return matrix3x3_x_img3ch(target_img,[0.299   ,0.587   ,0.114,
                                          -0.14713,-0.28886,0.436,
                                          0.615   ,-0.51499,-0.10001])
def rgb_to_BT601(target_img):
    return matrix3x3_x_img3ch(target_img,[0.299    ,0.587    ,0.114,
                                          -0.168736,-0.331264,0.5,
                                          0.5      ,-0.418688,-0.081312])
def rgb_to_BT709(target_img):
    return matrix3x3_x_img3ch(target_img,[0.2126   ,0.7152   ,0.0722,
                                          -0.114572,-0.385428,0.5,
                                          0.5      ,-0.454153,-0.045847])

def yuv_to_rgb(target_img):
    return matrix3x3_x_img3ch(target_img,[1 ,0       ,1.13983,
                                          1 ,-0.39465,-0.58060,
                                          1 ,2.03211 ,0       ])
def BT601_to_rgb(target_img):
    return matrix3x3_x_img3ch(target_img,[1,0        ,1.402,
                                          1,-0.344136,-0.714136,
                                          1,1.772    ,0         ])
def BT709_to_rgb(target_img):
    return matrix3x3_x_img3ch(target_img,[1,0        ,1.5748,
                                          1,-0.187324,-0.468124,
                                          1,1.8556   ,0         ])


def rgb_to_hsv(tgt_img):
    """
    rgb画像をhsvに変換する。
    colorsys.rgb_to_hsvと計算誤差程度のずれはあるが一致をする。
    :param tgt_img: rgb画像
    :return:
    """
    max_color = np.nanmax(tgt_img, axis=2)
    min_color = np.nanmin(tgt_img, axis=2)
    color_range = max_color - min_color
    h = 60 / 360 * ((tgt_img[:, :, 1] - tgt_img[:, :, 2]) / color_range)
    h[max_color == tgt_img[:, :, 1]] = (60 / 360 * ((tgt_img[:, :, 2] - tgt_img[:, :, 0]) / color_range) + 120 / 360)[max_color == tgt_img[:, :, 1]]
    h[max_color == tgt_img[:, :, 2]] = (60 / 360 * ((tgt_img[:, :, 0] - tgt_img[:, :, 1]) / color_range) + 240 / 360)[max_color == tgt_img[:, :, 2]]
    h[h < 0] = h[h < 0] + 1
    h[(tgt_img[:, :, 0] == tgt_img[:, :, 1]) * (tgt_img[:, :, 0] == tgt_img[:, :, 2])] = 0
    s = color_range / max_color
    s[np.isnan(s)] = 0
    # v = max_color
    return np.concatenate([h[:, :, np.newaxis], s[:, :, np.newaxis], max_color[:, :, np.newaxis]], axis=2)


def hsv_to_rgb(tgt_img):
    """
    hsv画像をrgbに変換する。
    colorsys.hsv_to_rgbと計算誤差程度のずれはあるが一致をする。
    :param tgt_img: hsv画像
    :return:
    """
    i = tgt_img[:, :, 0] * 6
    f = i - np.floor(i)
    p = (tgt_img[:, :, 2] * (1 - tgt_img[:, :, 1]))[:, :, np.newaxis]
    q = (tgt_img[:, :, 2] * (1 - tgt_img[:, :, 1] * f))[:, :, np.newaxis]
    t = (tgt_img[:, :, 2] * (1 - tgt_img[:, :, 1] * (1 - f)))[:, :, np.newaxis]
    v = (tgt_img[:, :, 2])[:, :, np.newaxis]
    i = np.tile(np.mod(np.floor(i), 6)[:, :, np.newaxis], (1, 1, 3))
    out_img = np.concatenate([v, t, p], axis=2)
    out_img[i == 1] = np.concatenate([q, v, p], axis=2)[i == 1]
    out_img[i == 2] = np.concatenate([p, v, t], axis=2)[i == 2]
    out_img[i == 3] = np.concatenate([p, q, v], axis=2)[i == 3]
    out_img[i == 4] = np.concatenate([t, p, v], axis=2)[i == 4]
    out_img[i == 5] = np.concatenate([v, p, q], axis=2)[i == 5]
    return out_img

######################################################################################################################## WB
def wb_for_bayer(target_img,gain,raw_mode):
    out_img = np.zeros_like(target_img)
    if raw_mode==0:#RGGB
        out_img[0::2,0::2] = target_img[0::2,0::2]*gain[0]
        out_img[0::2,1::2] = target_img[0::2,1::2]*gain[1]
        out_img[1::2,0::2] = target_img[1::2,0::2]*gain[1]
        out_img[1::2,1::2] = target_img[1::2,1::2]*gain[2]
    elif raw_mode==1:#GRBG
        out_img[0::2,0::2] = target_img[0::2,0::2]*gain[1]
        out_img[0::2,1::2] = target_img[0::2,1::2]*gain[0]
        out_img[1::2,0::2] = target_img[1::2,0::2]*gain[2]
        out_img[1::2,1::2] = target_img[1::2,1::2]*gain[1]
    elif raw_mode==2:#GBRG
        out_img[0::2,0::2] = target_img[0::2,0::2]*gain[1]
        out_img[0::2,1::2] = target_img[0::2,1::2]*gain[2]
        out_img[1::2,0::2] = target_img[1::2,0::2]*gain[0]
        out_img[1::2,1::2] = target_img[1::2,1::2]*gain[1]
    elif raw_mode==3:#BGGR
        out_img[0::2,0::2] = target_img[0::2,0::2]*gain[2]
        out_img[0::2,1::2] = target_img[0::2,1::2]*gain[1]
        out_img[1::2,0::2] = target_img[1::2,0::2]*gain[1]
        out_img[1::2,1::2] = target_img[1::2,1::2]*gain[0]
    else:
        out_img[0::2,0::2] = target_img[0::2,0::2]*gain[0]
        out_img[0::2,1::2] = target_img[0::2,1::2]*gain[1]
        out_img[1::2,0::2] = target_img[1::2,0::2]*gain[1]
        out_img[1::2,1::2] = target_img[1::2,1::2]*gain[2]
    return out_img

def wbgain_estimate_gray_world(target_img,raw_mode):
    if raw_mode==0:#RGGB
        g_mean = np.mean(np.array([target_img[0::2,1::2],target_img[1::2,0::2]]))
        r_gain = g_mean/np.mean(target_img[0::2,0::2])
        b_gain = g_mean/np.mean(target_img[1::2,1::2])
    elif raw_mode==1:#GRBG
        g_mean = np.mean(np.array([target_img[0::2,0::2],target_img[1::2,1::2]]))
        r_gain = g_mean/np.mean(target_img[0::2,1::2])
        b_gain = g_mean/np.mean(target_img[1::2,0::2])
    elif raw_mode==2:#GBRG
        g_mean = np.mean(np.array([target_img[0::2,0::2],target_img[1::2,1::2]]))
        r_gain = g_mean/np.mean(target_img[1::2,0::2])
        b_gain = g_mean/np.mean(target_img[0::2,1::2])
    elif raw_mode==3:#BGGR
        g_mean = np.mean(np.array([target_img[0::2,1::2],target_img[1::2,0::2]]))
        r_gain = g_mean/np.mean(target_img[1::2,1::2])
        b_gain = g_mean/np.mean(target_img[0::2,0::2])
    else:
        g_mean = np.mean(np.array([target_img[0::2,1::2],target_img[1::2,0::2]]))
        r_gain = g_mean/np.mean(target_img[0::2,0::2])
        b_gain = g_mean/np.mean(target_img[1::2,1::2])
    return [r_gain,1,b_gain]
