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

def rgb_to_hsv(target_img):
    ch_wise_max = np.nanmax(target_img, axis=2)
    ch_wise_min = np.nanmin(target_img, axis=2)
    h = 60 * ((target_img[:, :, 1] - target_img[:, :, 2]) / (ch_wise_max - ch_wise_min))
    h[ch_wise_max == target_img[:, :, 1]] = (60 * ((target_img[:, :, 2] - target_img[:, :, 0]) / (ch_wise_max - ch_wise_min)) + 120)[ch_wise_max == target_img[:, :, 1]]
    h[ch_wise_max == target_img[:, :, 2]] = (60 * ((target_img[:, :, 0] - target_img[:, :, 1]) / (ch_wise_max - ch_wise_min)) + 240)[ch_wise_max == target_img[:, :, 2]]
    h[h < 0] = h[h < 0] + 360
    h[(target_img[:, :, 0] == target_img[:, :, 1]) * (target_img[:, :, 0] == target_img[:, :, 2])] = 0
    s = (ch_wise_max - ch_wise_min) / ch_wise_max
    v = ch_wise_max
    return np.concatenate([h[:, :, np.newaxis], s[:, :, np.newaxis], v[:, :, np.newaxis]], axis=2) / 360

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
