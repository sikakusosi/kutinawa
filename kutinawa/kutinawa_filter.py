"""
filterを使った畳み込み処理に関連する関数群.
"""
import numpy as np
from scipy import ndimage
from scipy import stats
from numpy.lib.stride_tricks import as_strided

def multi_filter(target_img,filter_mat,mode='constant'):
    """
    一つの画像に対して、複数のフィルターを掛けた結果を返す

    :param target_img: フィルタリングしたい画像(ndarray)
    :param filter_mat: フィルター(第0次元方向にフィルタ種類)
    :param mode: パディングのモード、ndimage.convolveのmodeに準拠
    :return: フィルタリングされた画像(ndarray)
    """
    filtered_img = np.array([ndimage.convolve(target_img, np.squeeze(fil), mode=mode) for fil in filter_mat])
    return filtered_img


def fast_boxfilter(target_img,fil_h,fil_w):
    """
    integral imageを用いたO(1)の高速boxfilter
    結果は「ndimage.convolve(target_img, np.ones((fil_h,fil_w)), mode='mirror')」と、計算誤差程度の一致をする
    ただintegral imageの特性上、バイナリ一致とかはしないし、画像の右下ほど誤差が乗る

    雑アルゴル：パディング → integral image作成 → A-B-C+D
    参考：www.sanko-shoko.net/note.php?id=kzqj

    :param target_img:  フィルタリングしたい画像
    :param fil_h:       boxフィルタの高さ（奇数）
    :param fil_w:       boxフィルタの幅（奇数）
    :return:box         フィルタリング後の画像
    """

    # target_img2 = np.pad(target_img,(((fil_h//2)+1, (fil_h//2)), ((fil_w//2)+1, (fil_w//2))),'reflect')
    integ_img = np.cumsum(np.cumsum(np.pad(target_img,(((fil_h//2)+1, (fil_h//2)), ((fil_w//2)+1, (fil_w//2))),'reflect'), axis=0), axis=1)
    return integ_img[fil_h::, fil_w::] - integ_img[0:-fil_h, fil_w::] - integ_img[fil_h::, 0:-fil_w] + integ_img[0:-fil_h, 0:-fil_w]

def fast_boxfilter_even(target_img,fil_h,fil_w):
    """
    integral imageを用いたO(1)の高速boxfilter
    結果は「ndimage.convolve(target_img, np.ones((fil_h,fil_w)), mode='mirror')」と、計算誤差程度の一致をする
    ただintegral imageの特性上、バイナリ一致とかはしないし、画像の右下ほど誤差が乗る
    雑アルゴル：パディング → integral image作成 → A-B-C+D
    参考：www.sanko-shoko.net/note.php?id=kzqj

    :param target_img:  フィルタリングしたい画像
    :param fil_h:       boxフィルタの高さ（偶数）
    :param fil_w:       boxフィルタの幅（偶数）
    :return:            boxフィルタリング後の画像
    """
    target_img2 = np.pad(target_img,(((fil_h//2)+1, (fil_h//2)), ((fil_w//2)+1, (fil_w//2))),'reflect')
    integ_img = np.cumsum(np.cumsum(target_img2, axis=0), axis=1)
    temp = integ_img[fil_h::, fil_w::] - integ_img[0:-fil_h, fil_w::] - integ_img[fil_h::, 0:-fil_w] + integ_img[0:-fil_h, 0:-fil_w]
    return temp[1::,1::]

def fast_bayer_boxfilter(target_bayer,fil_h,hil_w):
    """
    bayer画像のR/G1/G2/Bにそれぞれ高速boxfilter適用

    :param target_bayer:    対象のbayer
    :param fil_h:           フィルタの高さ（奇数）
    :param hil_w:           フィルタの幅（奇数）
                            （fil_h=3,fil_w=5なら、R/G1/G2/Bそれぞれに 3x5 のフィルタがかかる）
    :return:                フィルタリング後の画像
    """
    out_bayer = np.zeros_like(target_bayer)
    out_bayer[0::2, 0::2] = fast_boxfilter(target_bayer[0::2, 0::2], fil_h, hil_w)
    out_bayer[0::2, 1::2] = fast_boxfilter(target_bayer[0::2, 1::2], fil_h, hil_w)
    out_bayer[1::2, 0::2] = fast_boxfilter(target_bayer[1::2, 0::2], fil_h, hil_w)
    out_bayer[1::2, 1::2] = fast_boxfilter(target_bayer[1::2, 1::2], fil_h, hil_w)
    return out_bayer

def fast_freepos_boxfilter(target_img,fil):
    """
    要素はboxフィルタなんだけど、位置がずれてるフィルタを高速にフィルタリングする
    フィルタ自体のshapeも、有効要素の形状も「奇数x奇数」を前提
    例：fil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],           A
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],  　       |
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],  　A      |
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],  　|      |
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],  　|奇数  | 奇数
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],  　|      |
                        [1, 1, 1, 1, 1, 0, 0, 0, 0],  　V      |
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],  　       |
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],])　       V
                        <----奇数---->
                        <----------奇数---------->

    :param target_img:  フィルタリングしたい画像
    :param fil:         フィルタ(2次元array)　注！フィルタ自体のshapeも、有効要素の形状も「奇数x奇数」を前提
    :return:
    """
    # 入力フィルタ自体の中心算出
    fil_center = np.floor(np.array(np.shape(fil)) / 2)

    # 有効な(≠0)要素で形成される矩形領域のサイズと中心算出
    box_index = np.where(fil != 0)
    box_fil_h = box_index[0][-1] - box_index[0][0] + 1
    box_fil_w = box_index[1][-1] - box_index[1][0] + 1
    box_fil_center = np.array([np.floor(np.array(box_fil_h) / 2) + box_index[0][0], np.floor(np.array(box_fil_w) / 2) + box_index[1][0]])

    # 結果のシフト量算出
    shift_list = (fil_center - box_fil_center).astype(np.int)

    temp = fast_boxfilter(target_img, fil_h=box_fil_h, fil_w=box_fil_w)
    target_img_shape = np.array(np.shape(target_img)).astype(np.int)
    result_img = np.zeros_like(target_img)

    result_img[np.clip(-shift_list[0], 0, None):target_img_shape[0] - shift_list[0],
    np.clip(-shift_list[1], 0, None):target_img_shape[1] - shift_list[1]] = temp[np.clip(shift_list[0], 0, None):np.clip(target_img_shape[0] + shift_list[0],None, target_img_shape[0]),
                                                                            np.clip(shift_list[1], 0, None):np.clip(target_img_shape[1] + shift_list[1],None, target_img_shape[1]),]
    return result_img

def variance_filter(target_img,fil):
    """
    局所分散を計算する
    :param target_img:  対象画像
    :param fil:         フィルタ、総和が1であること
    :return:            局所分散画像（計算誤差で0を割ることがある）
    """
    temp = ndimage.convolve(target_img, fil, mode='mirror')
    return ndimage.convolve(target_img*target_img, fil, mode='mirror') - (temp*temp)

def covariance_filter(target_img1,target_img2,fil):
    """
    局所分散を計算する
    :param target_img1 ,target_img2: 対象画像
    :param fil                     : フィルタ、総和が1であること
    :return                        : 局所共分散画像（計算誤差で0を割ることがある）
    """
    return ndimage.convolve(target_img1*target_img2, fil, mode='mirror')-ndimage.convolve(target_img1, fil, mode='mirror')*ndimage.convolve(target_img2, fil, mode='mirror')

def fast_box_variance_filter(target_img,fil_h,fil_w):
    """
    局所分散を計算する
    :param target_img:  対象画像
    :param fil:         フィルタ、総和が1であること
    :return:            局所分散画像（計算誤差で0を割ることがある）
    """
    temp = fast_boxfilter(target_img, fil_h=fil_h, fil_w=fil_w)/(fil_h*fil_w)
    return fast_boxfilter(target_img*target_img, fil_h=fil_h, fil_w=fil_w)/(fil_h*fil_w) - (temp*temp)

def generate_bayer_boxfilter(fil_size,mode='sum1'):
    if mode=='sum1':
        fil = np.zeros((fil_size,fil_size))
        fil[0::2,0::2] = 1
        fil = fil/np.sum(fil)
    else:
        fil = np.zeros((fil_size, fil_size))
        fil[0::2, 0::2] = 1

    return fil

def generate_gaussian_filter(shape,sigma):
    """
    フィルタの総和が1になる、任意矩形形状のガウシアンフィルタを返す
    generate_gaussian_filter((5,5),0.5) = fspecial('gaussian',5,0.5)<Matlab func>
    例外処理として、sigma=0の場合はshapeはそのまま、値は中心に1のフィルタを返す

    :param shape: タプル、フィルタの(y_size, x_size)。奇数であること
    :param sigma: ガウス関数のシグマ
    :return: フィルタの総和が1になる、任意矩形形状のガウシアンフィルタ
    """
    shape = (int(shape[0]),int(shape[1]))
    if sigma==0:
        temp = np.zeros(shape)
        temp[shape[0]//2,shape[1]//2] = 1
        return temp

    m,n = [(ss-1.0)/2.0 for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    gf = np.exp( -(x*x + y*y) / (2.0*sigma*sigma) )
    gf[ gf < np.finfo(gf.dtype).eps*gf.max() ] = 0
    sum_gf = gf.sum()
    if sum_gf != 0:
        gf = gf/sum_gf
    return gf


def image_stack(target_img,rh,rw,padmode='reflect'):
    """
    位置をずらした画像を2次元目にスタックする。
    バイラテラルフィルタ等のpython実装に有用。
    :param target_img:
    :param rh: ずらし量の高さ方向半径
    :param rw: ずらし量の幅方向半径
    :return: スタックした画像
    """
    dh = rh+rh+1
    dw = rw+rw+1
    target_img_size = np.shape(target_img)
    stack_img = np.tile(np.pad(target_img,((rh,rh), (rw,rw)),padmode)[:,:,np.newaxis],(1,1,dh*dw))

    for h in np.arange(0,dh):
        stack_img[rh:-rh, :, h*(dw):h*dw+dw] = stack_img[h:h+target_img_size[0],:,h*dw:h*dw+dw]
    w_index = np.arange(0,dh*dw,dw)
    for w in np.arange(0,dw):
        stack_img[:, rw:-rw, w_index+w] = stack_img[:, w:w+target_img_size[1], w_index+w]
    return stack_img[rh:-rh,rw:-rw,:]

def mode_filter(target_img,rh,rw):
    stack_target_img = image_stack(target_img,rh,rw)
    mode_img,mode_count_img = stats.mode(stack_target_img,axis=2)
    return np.squeeze(mode_img),np.squeeze(mode_count_img)


def convolve2d_nan(target_img,weights,mode='mirror'):
    """
    2次元畳み込みをする関数。
    numpyのみで動作可能。
    またnanが混入していても、無視して動作可能。
    :param target_img:  畳み込み対象の画像。2d-ndarray
    :param weights:     畳み込みの重み。必ず「奇数 x 奇数」のarrayであること。2d-ndarray
    :param mode:        パディング方式を指定する。下記いずれかの文字列。scipy.ndimage.convolveに準拠。
                        ‘reflect’     (c b a | a b c d | d c b)
                        ‘constant’    (k k k | a b c d | k k k)
                        ‘nearest’     (a a a | a b c d | d d d)
                        ‘mirror’      (d c b | a b c d | c b a)
                        ‘wrap’        (b c d | a b c d | a b c)
                        参考：https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

    :return:            畳み込みされた画像。2d-ndarray
                        scipy.ndimage.convolve(target_img, weights, mode=mode)と計算誤差程度(最大誤差は10-e13くらい)の一致をする
    """

    if mode=='mirror':
        stk_img = image_stack(target_img,np.shape(weights)[0]//2,np.shape(weights)[1]//2,padmode='reflect')
    elif mode=='reflect':
        stk_img = image_stack(target_img,np.shape(weights)[0]//2,np.shape(weights)[1]//2,padmode='symmetric')
    elif mode=='constant':
        stk_img = image_stack(target_img,np.shape(weights)[0]//2,np.shape(weights)[1]//2,padmode='constant')
    elif mode=='nearest':
        stk_img = image_stack(target_img,np.shape(weights)[0]//2,np.shape(weights)[1]//2,padmode='edge')
    elif mode=='wrap':
        stk_img = image_stack(target_img,np.shape(weights)[0]//2,np.shape(weights)[1]//2,padmode='wrap')
    else:
        raise Exception("""kutinawa-Error! : [mode] was not specified correctly.Specify one of "reflect', 'constant', 'nearest', 'mirror', or 'wrap'.""")


    stk_wgt = weights.flatten()[::-1][np.newaxis,np.newaxis,:]
    out_img = np.nansum(stk_img*stk_wgt,axis=2)
    return out_img



def conv2d_only_numpy(target_img,weights,mode):
    """
    2次元畳み込みをする関数。
    numpyのみで動作可能。
    :param target_img:  畳み込み対象の画像。2d-ndarray
    :param weights:     畳み込みの重み。必ず「奇数 x 奇数」のarrayであること。2d-ndarray
    :param mode:        パディング方式を指定する。下記いずれかの文字列。scipy.ndimage.convolveに準拠。
                        ‘reflect’     (c b a | a b c d | d c b)
                        ‘constant’    (k k k | a b c d | k k k)
                        ‘nearest’     (a a a | a b c d | d d d)
                        ‘mirror’      (d c b | a b c d | c b a)
                        ‘wrap’        (b c d | a b c d | a b c)
                        参考：https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

    :return:            畳み込みされた画像。2d-ndarray
                        scipy.ndimage.convolve(target_img, weights, mode=mode)と計算誤差程度(最大誤差は10-e13くらい)の一致をする
    """

    if len(target_img.shape) != 2:
        print('img should be 2d-array')
        return
    if weights.shape[0]%2 == 0 or weights.shape[1]%2 == 0:
        print('Kernel size shold be odd')
        return

    rh,rw = np.shape(weights)[0]//2,np.shape(weights)[1]//2
    if weights.shape[0]>weights.shape[1]:
        weights = np.pad(weights,((0,0), (rh-rw,rh-rw)),'constant')
    elif weights.shape[0]<weights.shape[1]:
        weights = np.pad(weights,((rw-rh,rw-rh), (0,0)),'constant')

    rh,rw = np.shape(weights)[0]//2,np.shape(weights)[1]//2
    if mode=='mirror':
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'reflect')
    elif mode=='reflect':
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'symmetric')
    elif mode=='constant':
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'constant')
    elif mode=='nearest':
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'edge')
    elif mode=='wrap':
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'wrap')
    else:
        target_img = np.pad(target_img,((rh,rh), (rw,rw)),'constant')

    sub_shape = tuple(np.subtract(target_img.shape, weights.shape) + 1)
    conv_shape = sub_shape + weights.shape
    strides = target_img.strides + target_img.strides
    submatrices = as_strided(target_img, conv_shape, strides)
    convolved_mat = np.einsum('ij, klij->kl', np.rot90(weights,2), submatrices)
    return convolved_mat