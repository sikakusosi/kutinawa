"""
高画質化処理の関数群.
画像サイズを変えず、画質を向上させる様な処理をここに集約する。
NR,demosaic,sharpness,deconvolution,等が該当する
"""
import numpy as np
from scipy import ndimage
from.kutinawa_filter import fast_boxfilter, generate_gaussian_filter

######################################################################################################################## noise reduction

def generate_noise_map(target_raw,A,K,S,OB):
    # var(I) = A(I-OB)*(I-OB) + K(I-OB) + S*S
    # A ：画素ごとのゲインばらつき
    # K ：ショットノイズの輝度依存係数、基本的に光電変換係数(のファクター倍)
    # S ：暗電流ノイズの標準偏差
    # I ：画素値
    # OB：画像の黒レベル

    target_raw2 = target_raw - OB
    noise_gain_map = np.zeros_like(target_raw)
    noise_gain_map[0::2, 0::2] = np.sqrt(A[0]*target_raw2[0::2, 0::2]*target_raw2[0::2, 0::2] + target_raw2[0::2, 0::2]*K[0] + S[0]*S[0])
    noise_gain_map[0::2, 1::2] = np.sqrt(A[1]*target_raw2[0::2, 1::2]*target_raw2[0::2, 1::2] + target_raw2[0::2, 1::2]*K[1] + S[1]*S[1])
    noise_gain_map[1::2, 0::2] = np.sqrt(A[2]*target_raw2[1::2, 0::2]*target_raw2[1::2, 0::2] + target_raw2[1::2, 0::2]*K[2] + S[2]*S[2])
    noise_gain_map[1::2, 1::2] = np.sqrt(A[3]*target_raw2[1::2, 1::2]*target_raw2[1::2, 1::2] + target_raw2[1::2, 1::2]*K[3] + S[3]*S[3])

    return noise_gain_map

def bilateral_filter(target_img,fil_size,spacial_sigma,pixval_sigma,mode='reflect'):
    pad_image = np.pad(target_img, ((int(fil_size[0] / 2),), (int(fil_size[1] / 2),)), mode)
    inner_pixels_stride = np.lib.stride_tricks.as_strided(pad_image, target_img.shape + fil_size, pad_image.strides * 2)
    center_pixel_stride = np.tile(target_img.reshape(target_img.shape + (1, 1)), (1, 1) + fil_size)
    weights = generate_gaussian_filter(fil_size,spacial_sigma,sum1=False) * np.exp(-((center_pixel_stride - inner_pixels_stride) ** 2) / (2.0 * pixval_sigma * pixval_sigma * 2.0))
    weight_sum = np.sum(weights, axis=(2, 3))
    bil_img = np.einsum('ijkl,ijkl->ij', weights, inner_pixels_stride) / weight_sum
    return bil_img

def non_local_means(target_img,block_size,search_size,stride,th):
    """
                  search_size
    <------------------------------------->
       block_size
    <------------->
    +--------------+-----------------------+
    |              |                       |
    |    +----+    |                       |
    |    |    |    |  -----------------    |
    |    +----+    |             ---       |
    |              |         ---           |
    +--------------+     ---               |
    |                ---                   |
    |            ---        +--------------+
    |        ---            |              |
    |    ---                |    +----+    |
    |  -------------------> |    |    |    |
    |                       |    +----+    |
    |                       |              |
    +-----------------------+--------------+

    計算速度が探索範囲のサイズのみに依存するnlm
    (厳密には画像サイズによるメモリの差等もあるため画像サイズ＋探索範囲)
    :param target_img:対象画像(1ch)。ndarray
    :param block_size:パッチサイズ。奇数のみ。listもしくはタプル
    :param search_size:探索範囲のサイズ。上図のように探索パッチは探索範囲をはみ出さないように動く。listもしくはタプル
    :param stride:探索パッチを何画素ごとに動かすかを示す。listもしくはタプル
    :param th:ブロックマッチングのしきい値。スカラもしくは入力画像と同じサイズの画像
    :return:nlm後の画像。ndarray
    """

    # 探索パッチが探索範囲の端まで行かない場合に警告
    if ((search_size[0]-block_size[0])%stride[0]!=0) or ((search_size[1]-block_size[1])%stride[1]!=0):
        print("kutinawa-warning! : It is not possible to search the entire search area with that parameter.")

    # paddingして探索範囲をカバーできるように
    target_img2 = np.pad(target_img, ((search_size[0] // 2, search_size[0] // 2), (search_size[1] // 2, search_size[1] // 2)), 'reflect')

    # 出力画像を定義
    out_img = np.zeros_like(target_img)
    out_count = np.zeros_like(target_img)

    # nlm
    hw = np.shape(target_img)
    for ul_h in np.arange(0,search_size[0],stride[0]):
        # print(ul_h)
        for ul_w in np.arange(0,search_size[1],stride[1]):
            temp_img = target_img2[ul_h:ul_h+hw[0],ul_w:ul_w+hw[1]]
            diff = target_img-temp_img
            sad = fast_boxfilter(diff*diff,block_size[0],block_size[1])
            out_img[sad<=th] = out_img[sad<=th]+temp_img[sad<=th]
            out_count[sad<=th] = out_count[sad<=th] + 1
    return out_img/out_count

def non_local_means_with_guide(target_img,guide_img,block_size,search_size,stride,th):
    """
                  search_size
    <------------------------------------->
       block_size
    <------------->
    +--------------+-----------------------+
    |              |                       |
    |    +----+    |                       |
    |    |    |    |  -----------------    |
    |    +----+    |             ---       |
    |              |         ---           |
    +--------------+     ---               |
    |                ---                   |
    |            ---        +--------------+
    |        ---            |              |
    |    ---                |    +----+    |
    |  -------------------> |    |    |    |
    |                       |    +----+    |
    |                       |              |
    +-----------------------+--------------+

    計算速度が探索範囲のサイズのみに依存するnlm
    (厳密には画像サイズによるメモリの差等もあるため画像サイズ＋探索範囲)
    :param target_img:NR対象画像(1ch)。ndarray
    :param guide_img:ブロックマッチング対象画像、target_imgと同サイズであること。guide_img＝target_imgにすることで通常のNLMとなる。通常target_imgよりS/Nの良い画像を使用する。ndarray
    :param block_size:パッチサイズ。listもしくはタプル
    :param search_size:探索範囲のサイズ。上図のように探索パッチは探索範囲をはみ出さないように動く。listもしくはタプル
    :param stride:探索パッチを何画素ごとに動かすかを示す。listもしくはタプル
    :param th:ブロックマッチングのしきい値。スカラもしくは入力画像と同じサイズの画像
    :return:nlm後の画像。ndarray
    """

    # 探索パッチが探索範囲の端まで行かない場合に警告
    if ((search_size[0]-block_size[0])%stride[0]!=0) or ((search_size[1]-block_size[1])%stride[1]!=0):
        print("kutinawa : warning! - It is not possible to search the entire search area with that parameter.")

    # paddingして探索範囲をカバーできるように
    guide_img2 = np.pad(guide_img, ((search_size[0] // 2, search_size[0] // 2), (search_size[1] // 2, search_size[1] // 2)), 'reflect')
    target_img2 = np.pad(target_img, ((search_size[0] // 2, search_size[0] // 2), (search_size[1] // 2, search_size[1] // 2)), 'reflect')

    # 出力画像を定義
    out_img = np.zeros_like(target_img)
    out_count = np.zeros_like(target_img)

    # nlm
    hw = np.shape(target_img)
    for ul_h in np.arange(0,search_size[0],stride[0]):
        # print(ul_h)
        for ul_w in np.arange(0,search_size[1],stride[1]):
            temp_target_img = target_img2[ul_h:ul_h+hw[0],ul_w:ul_w+hw[1]]
            temp_guide_img  = guide_img2[ul_h:ul_h+hw[0],ul_w:ul_w+hw[1]]
            diff = guide_img-temp_guide_img
            sad = fast_boxfilter(diff*diff,block_size[0],block_size[1])
            out_img[sad<th] = out_img[sad<th]+temp_target_img[sad<th]
            out_count[sad<th] = out_count[sad<th] + 1
    return out_img/out_count

def guided_filter(target_img,guide_img,box_h_size,box_v_size,eps=0.01):
    """
    guided filter
    [1]K.He, J.Sun, and X.Tang, "Guided Image Filtering", European Conference on Computer Vision (ECCV), 2010
    http://kaiminghe.com/publications/eccv10guidedfilter.pdf
    [2]K.He, J.Sun, and X.Tang, "Guided Image Filtering", IEEE Transactions on Pattern Analysis and Machine Intelligence(PAMI), 2013
    http://kaiminghe.com/publications/pami12guidedfilter.pdf
    [3]K.He, J.Sun, "Fast Guided Filter", arXiv, 2015
    https://arxiv.org/abs/1505.00996

    :param target_img:処理対象の画像。ndarray
    :param guide_img:ガイド画像。処理対象の画像と同じsizeであること。ndarray
    :param box_h_size:guided filter内で使うboxfilterの幅
    :param box_v_size:guided filter内で使うboxfilterの高さ
    :param eps:0割りを防止するための微小値
    :return:guided filter結果
    """
    box_count = box_h_size*box_v_size
    mean_I = fast_boxfilter(guide_img,box_h_size,box_v_size)/box_count
    mean_p = fast_boxfilter(target_img,box_h_size,box_v_size)/box_count
    corr_I = fast_boxfilter(guide_img*guide_img,box_h_size,box_v_size)/box_count
    corr_Ip= fast_boxfilter(guide_img*target_img,box_h_size,box_v_size)/box_count
    var_I  = corr_I-mean_I*mean_I
    cov_Ip = corr_Ip-mean_I*mean_p
    a      = cov_Ip/(var_I+eps)
    b      = mean_p - a*mean_I
    mean_a = fast_boxfilter(a,box_h_size,box_v_size)/box_count
    mean_b = fast_boxfilter(b,box_h_size,box_v_size)/box_count
    q      = mean_a*guide_img + mean_b
    return q

######################################################################################################################## demosaic
def demosaic_121(target_img,raw_mode):
    """
    Low quality demosaic.
    R and B are just interpolated with 121 filter, G with 141 filter.
    低クオリティのデモザイク。
    R,Bは121フィルタ、Gは141フィルタで補間してるだけ。
    :param target_img:  bayer
    :return:            RGB
    """
    fil_RB = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 4
    fil_G = np.array([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]]) / 4

    target_rawR = np.zeros_like(target_img)
    target_rawB = np.zeros_like(target_img)
    target_rawG = np.zeros_like(target_img)
    out_img = np.zeros((np.shape(target_img)[0], np.shape(target_img)[1], 3))

    if raw_mode==0:#RGGB
        target_rawR[0::2, 0::2] = target_img[0::2,0::2].copy()
        target_rawG[0::2, 1::2] = target_img[0::2,1::2].copy()
        target_rawG[1::2, 0::2] = target_img[1::2,0::2].copy()
        target_rawB[1::2, 1::2] = target_img[1::2,1::2].copy()
    elif raw_mode==1:#GRBG
        target_rawG[0::2,0::2] = target_img[0::2,0::2].copy()
        target_rawR[0::2,1::2] = target_img[0::2,1::2].copy()
        target_rawB[1::2,0::2] = target_img[1::2,0::2].copy()
        target_rawG[1::2,1::2] = target_img[1::2,1::2].copy()
    elif raw_mode==2:#GBRG
        target_rawG[0::2,0::2] = target_img[0::2,0::2].copy()
        target_rawB[0::2,1::2] = target_img[0::2,1::2].copy()
        target_rawR[1::2,0::2] = target_img[1::2,0::2].copy()
        target_rawG[1::2,1::2] = target_img[1::2,1::2].copy()
    elif raw_mode==3:#BGGR
        target_rawB[0::2,0::2] = target_img[0::2,0::2].copy()
        target_rawG[0::2,1::2] = target_img[0::2,1::2].copy()
        target_rawG[1::2,0::2] = target_img[1::2,0::2].copy()
        target_rawR[1::2,1::2] = target_img[1::2,1::2].copy()
    else:
        target_rawR[0::2,0::2] = target_img[0::2,0::2].copy()
        target_rawG[0::2,1::2] = target_img[0::2,1::2].copy()
        target_rawG[1::2,0::2] = target_img[1::2,0::2].copy()
        target_rawB[1::2,1::2] = target_img[1::2,1::2].copy()

    out_img[:, :, 0] = ndimage.convolve(target_rawR, fil_RB, mode='mirror')
    out_img[:, :, 2] = ndimage.convolve(target_rawB, fil_RB, mode='mirror')
    out_img[:, :, 1] = ndimage.convolve(target_rawG, fil_G, mode='mirror')

    return out_img


def demosaic_GBTF(target_raw):
    """
    I. Pekkucuksen and Y. Altunbasak, "Gradient based threshold free color filter array interpolation,"
    2010 IEEE International Conference on Image Processing, Hong Kong, 2010, pp. 137-140.
    を実装したもの
    メモリ効率の都合上、変数の使いまわし等あるので、「論文と比較しての可読性」を求める場合は
    別に作成したgbft_ref.pyを参照のこと

    :param target_raw:  0-1に正規化されたbayer画像（bayer順序はrggb）
    :return:            0-1のRGB画像
    """
    ############################################################ 2.1. GreenChannelInterpolation
    # R,G,Bの水平、垂直補間
    fil_1 = np.array([[-1 / 4, 1 / 2, 1 / 2, 1 / 2, -1 / 4]])
    tilde_DELTA_H = ndimage.convolve(target_raw, fil_1, mode='mirror')  # 水平補間
    tilde_DELTA_V = ndimage.convolve(target_raw, np.rot90(fil_1), mode='mirror')  # 垂直補間

    # 垂直水平の色差(G-R、G-B)map
    # tilde_DELTA_V = np.zeros_like(target_raw)
    tilde_DELTA_V[0::2, 0::2] = tilde_DELTA_V[0::2, 0::2] - target_raw[0::2, 0::2]  # ~Gv-R 補間済みGと補間なしRの差分
    tilde_DELTA_V[1::2, 0::2] = target_raw[1::2, 0::2] - tilde_DELTA_V[1::2, 0::2]  # G-~Rv 補間なしGと補間済みRの差分
    tilde_DELTA_V[1::2, 1::2] = tilde_DELTA_V[1::2, 1::2] - target_raw[1::2, 1::2]  # ~Gv-B 補間済みGと補間なしBの差分
    tilde_DELTA_V[0::2, 1::2] = target_raw[0::2, 1::2] - tilde_DELTA_V[0::2, 1::2]  # G-~Bv 補間なしGと補間済みBの差分

    # tilde_DELTA_H = np.zeros_like(target_raw)
    tilde_DELTA_H[0::2, 0::2] = tilde_DELTA_H[0::2, 0::2] - target_raw[0::2, 0::2]  # ~Gh-R 補間済みGと補間なしRの差分
    tilde_DELTA_H[0::2, 1::2] = target_raw[0::2, 1::2] - tilde_DELTA_H[0::2, 1::2]  # G-~Rh 補間なしGと補間済みRの差分
    tilde_DELTA_H[1::2, 1::2] = tilde_DELTA_H[1::2, 1::2] - target_raw[1::2, 1::2]  # ~Gh-B 補間済みGと補間なしBの差分
    tilde_DELTA_H[1::2, 0::2] = target_raw[1::2, 0::2] - tilde_DELTA_H[1::2, 0::2]  # G-~Bh 補間なしGと補間済みBの差分

    # ~Δg,rを計算するため、ωN～Eを計算するための、Dv,Dhを計算する
    fil_dh = np.array([[0, 0, 0],
                       [1, 0, -1],
                       [0, 0, 0], ])

    Dh = np.abs(ndimage.convolve(tilde_DELTA_H, fil_dh, mode='mirror'))
    Dv = np.abs(ndimage.convolve(tilde_DELTA_V, np.rot90(fil_dh), mode='mirror'))

    # ~Δg,rを計算するため、ωN～Eを計算する
    fil_wn = np.array([[0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0], ])

    fil_ws = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0], ])

    fil_ww = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0], ])

    fil_we = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0], ])

    temp = ndimage.convolve(Dv, fil_wn, mode='mirror')
    wn = 1.0 / np.clip(temp * temp, 0.00001, None)
    temp = ndimage.convolve(Dv, fil_ws, mode='mirror')
    ws = 1.0 / np.clip(temp * temp, 0.00001, None)
    temp = ndimage.convolve(Dh, fil_ww, mode='mirror')
    ww = 1.0 / np.clip(temp * temp, 0.00001, None)
    temp = ndimage.convolve(Dh, fil_we, mode='mirror')
    we = 1.0 / np.clip(temp * temp, 0.00001, None)

    wt = wn + ws + ww + we

    del Dv, Dh, temp

    # ~Δg,rを計算する
    fil_n = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0], ]) / 5.0

    fil_s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0], ]) / 5.0

    fil_w = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0], ]) / 5.0

    fil_e = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0], ]) / 5.0

    temp_n = ndimage.convolve(tilde_DELTA_V, fil_n, mode='mirror')
    temp_s = ndimage.convolve(tilde_DELTA_V, fil_s, mode='mirror')
    temp_w = ndimage.convolve(tilde_DELTA_H, fil_w, mode='mirror')
    temp_e = ndimage.convolve(tilde_DELTA_H, fil_e, mode='mirror')
    tilde_DELTA = (wn * temp_n + ws * temp_s + ww * temp_w + we * temp_e) / wt

    del tilde_DELTA_V, tilde_DELTA_H, temp_n, temp_s, temp_w, temp_e, wn, ws, ww, we, wt

    # Gの補間
    result_G = np.zeros_like(target_raw)
    result_G[0::2, 1::2] = target_raw[0::2, 1::2]
    result_G[1::2, 0::2] = target_raw[1::2, 0::2]
    result_G[0::2, 0::2] = target_raw[0::2, 0::2] + tilde_DELTA[0::2, 0::2]
    result_G[1::2, 1::2] = target_raw[1::2, 1::2] + tilde_DELTA[1::2, 1::2]

    ############################################################ 2.2. Red and Blue Channel Interpolation
    # B位置のR、R位置のBの補間
    prb = np.array([[0, 0, -1, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 10, 0, 10, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 10, 0, 10, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, -1, 0, 0], ]) / 32.0

    tilde_DELTA_prb = ndimage.convolve(tilde_DELTA, prb, mode='mirror')
    result_R = np.zeros_like(target_raw)
    result_R[0::2, 0::2] = target_raw[0::2, 0::2]
    result_R[1::2, 1::2] = result_G[1::2, 1::2] - tilde_DELTA_prb[1::2, 1::2]
    result_B = np.zeros_like(target_raw)
    result_B[1::2, 1::2] = target_raw[1::2, 1::2]
    result_B[0::2, 0::2] = result_G[0::2, 0::2] - tilde_DELTA_prb[0::2, 0::2]

    # G位置のR/Bの補間
    fil_rb = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0], ]) / 4.0
    r_bilinear = ndimage.convolve(result_R, fil_rb, mode='mirror')
    g_bilinear = ndimage.convolve(result_G, fil_rb, mode='mirror')
    b_bilinear = ndimage.convolve(result_B, fil_rb, mode='mirror')
    result_R[0::2, 1::2] = result_G[0::2, 1::2] - g_bilinear[0::2, 1::2] + r_bilinear[0::2, 1::2]
    result_R[1::2, 0::2] = result_G[1::2, 0::2] - g_bilinear[1::2, 0::2] + r_bilinear[1::2, 0::2]
    result_B[0::2, 1::2] = result_G[0::2, 1::2] - g_bilinear[0::2, 1::2] + b_bilinear[0::2, 1::2]
    result_B[1::2, 0::2] = result_G[1::2, 0::2] - g_bilinear[1::2, 0::2] + b_bilinear[1::2, 0::2]

    return np.clip(np.stack([result_R, result_G, result_B], 2), 0, 1)