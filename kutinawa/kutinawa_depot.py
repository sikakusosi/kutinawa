"""
まとまりの無い関数の置き場。
そのうちどこかに移設統合する予定。
技術的まとまりができた時点で、既存のファイルを改変統合ないしは新たなファイルを作成し移設する。
(テスト不足とかではない。)
"""
import numpy as np
from .kutinawa_filter import fast_boxfilter,fast_box_variance_filter

def generate_window2d(hw, type_func):
    """
    最大値が1の、指定の幅高の2次元ウィンドウを作成する。

    :param hw:        listもしくはtaple形式の(高,幅)
    :param type_func: 下記いずれかの関数。
                      numpy.bartlett(M)
                      numpy.blackman(M)
                      numpy.hamming(M)
                      numpy.hanning(M)
                      (参考：https://numpy.org/doc/stable/reference/routines.window.html)
                      (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.htmlの関数もいくつか使用可)
    :return:          最大値が1の、指定の幅高の2次元ウィンドウ(ndarray)
    """
    window1d_1h = type_func(hw[0])
    window1d_1w = type_func(hw[1])
    return np.clip(np.sqrt(np.outer(window1d_1h,window1d_1w)),0,1)

def phase_only_correlation(ref_img, target_img):
    """
    平行移動のみの位置合わせ。想定精度は1ピクセル。

    :param ref_img:    基準となる画像
    :param target_img: 位置合わせしたい画像　(画像サイズはref_img≧target_imgとなるようにすること)
    :return:           ref_imgと同サイズ,位置合わせ済みの画像
    """
    # target_img padding
    ref_img_size    = np.array(np.shape(ref_img))
    target_img_size = np.array(np.shape(target_img))
    size_diff       = ref_img_size-target_img_size
    target_img_pad = np.pad(target_img, [(size_diff[0] // 2, size_diff[0] - size_diff[0] // 2), (size_diff[1] // 2, size_diff[1] - size_diff[1] // 2)], "constant")

    # 画像端の影響を低減させるためにvisnettingウィンドウを乗算
    visnetting_window = generate_window2d(ref_img_size, np.hanning)

    # POC、ズレ量算出
    ref_fft           = np.fft.fftshift(np.fft.fft2(ref_img * visnetting_window))
    target_fft        = np.fft.fftshift(np.fft.fft2(target_img_pad * visnetting_window))
    cps               = (ref_fft * target_fft.conj()) / np.abs(ref_fft * target_fft.conj())
    r                 = np.abs(np.fft.fftshift(np.fft.ifft2(cps)))
    r_pos             = np.array(divmod(np.argmax(r),ref_img_size[1]))
    shift_pos         = np.array([ref_img_size[0]//2,ref_img_size[1]//2])-r_pos

    # 出力画像
    shift_img = np.zeros_like(ref_img)
    shift_img[np.clip(-shift_pos[0],0,None):ref_img_size[0]-np.clip(shift_pos[0],0,None),
              np.clip(-shift_pos[1],0,None):ref_img_size[1]-np.clip(shift_pos[1],0,None)] \
              = target_img_pad[np.clip(shift_pos[0],0,None):ref_img_size[0]+np.clip(shift_pos[0],None,0),
                               np.clip(shift_pos[1],0,None):ref_img_size[1]+np.clip(shift_pos[1],None,0)]
    return shift_img


def snr_psnr(ref_img,eval_img,maximum_signal_range):
    diff_img = ref_img-eval_img
    mse = np.nanmean(diff_img*diff_img)

    snr,psnr=np.Inf,np.Inf
    if mse != 0:
        # SNR
        signal_minmax_range = np.nanmax(ref_img)-np.nanmin(ref_img)
        snr = 10*np.log10(signal_minmax_range*signal_minmax_range/mse)
        # PSNR
        psnr = 10*np.log10(maximum_signal_range*maximum_signal_range/mse)

    return snr,psnr


def ssim(ref_img, tgt_img, fil_h = 11,fil_w = 11,ref_img_range = np.NaN,C1_gain = 0.01,C2_gain = 0.03):
    if np.isnan(ref_img_range):
        ref_img_range = np.nanmax(ref_img)-np.nanmin(ref_img)
    C1 = (ref_img_range*C1_gain)*(ref_img_range*C1_gain)
    C2 = (ref_img_range*C2_gain)*(ref_img_range*C2_gain)

    ref_local_mean = fast_boxfilter(ref_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w
    ref_local_var  = fast_box_variance_filter(ref_img,fil_h=fil_h,fil_w=fil_w)
    tgt_local_mean = fast_boxfilter(tgt_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w
    tgt_local_var  = fast_box_variance_filter(tgt_img,fil_h=fil_h,fil_w=fil_w)
    ref_eva_cov    = fast_boxfilter(ref_img*tgt_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w - (ref_local_mean*tgt_local_mean)

    ssim = ( ( (2*ref_local_mean*tgt_local_mean+C1)*(2*ref_eva_cov+C2) )
             / ( (ref_local_mean*ref_local_mean+tgt_local_mean*tgt_local_mean+C1)*(ref_local_var+tgt_local_var+C2) ) )

    mssim = np.nanmean(ssim)

    return ssim,mssim

def bayer_RB_G_merge(target_RB,target_G,bayer_mode='RGGB'):
    out_img = np.zeros_like(target_RB)
    if bayer_mode=='RGGB':
        out_img[0::2, 0::2] = target_RB[0::2, 0::2]
        out_img[0::2, 1::2] = target_G[0::2, 1::2]
        out_img[1::2, 0::2] = target_G[1::2, 0::2]
        out_img[1::2, 1::2] = target_RB[1::2, 1::2]
    elif bayer_mode=='GRBG':
        out_img[0::2, 0::2] = target_G[0::2, 0::2]
        out_img[0::2, 1::2] = target_RB[0::2, 1::2]
        out_img[1::2, 0::2] = target_RB[1::2, 0::2]
        out_img[1::2, 1::2] = target_G[1::2, 1::2]
    elif bayer_mode=='GBRG':
        out_img[0::2, 0::2] = target_G[0::2, 0::2]
        out_img[0::2, 1::2] = target_RB[0::2, 1::2]
        out_img[1::2, 0::2] = target_RB[1::2, 0::2]
        out_img[1::2, 1::2] = target_G[1::2, 1::2]
    elif bayer_mode=='BGGR':
        out_img[0::2, 0::2] = target_RB[0::2, 0::2]
        out_img[0::2, 1::2] = target_G[0::2, 1::2]
        out_img[1::2, 0::2] = target_G[1::2, 0::2]
        out_img[1::2, 1::2] = target_RB[1::2, 1::2]

    return out_img


def rgb_to_bayer(target_rgb,bayer_mode='RGGB'):
    out_bayer = np.zeros((np.shape(target_rgb)[0], np.shape(target_rgb)[1]))
    if bayer_mode=='RGGB':
        out_bayer[0::2, 0::2] = target_rgb[0::2, 0::2, 0]
        out_bayer[0::2, 1::2] = target_rgb[0::2, 1::2, 1]
        out_bayer[1::2, 0::2] = target_rgb[1::2, 0::2, 1]
        out_bayer[1::2, 1::2] = target_rgb[1::2, 1::2, 2]
    elif bayer_mode=='GRBG':
        out_bayer[0::2, 0::2] = target_rgb[0::2, 0::2, 1]
        out_bayer[0::2, 1::2] = target_rgb[0::2, 1::2, 0]
        out_bayer[1::2, 0::2] = target_rgb[1::2, 0::2, 2]
        out_bayer[1::2, 1::2] = target_rgb[1::2, 1::2, 1]
    elif bayer_mode=='GBRG':
        out_bayer[0::2, 0::2] = target_rgb[0::2, 0::2, 1]
        out_bayer[0::2, 1::2] = target_rgb[0::2, 1::2, 2]
        out_bayer[1::2, 0::2] = target_rgb[1::2, 0::2, 0]
        out_bayer[1::2, 1::2] = target_rgb[1::2, 1::2, 1]
    elif bayer_mode=='BGGR':
        out_bayer[0::2, 0::2] = target_rgb[0::2, 0::2, 2]
        out_bayer[0::2, 1::2] = target_rgb[0::2, 1::2, 1]
        out_bayer[1::2, 0::2] = target_rgb[1::2, 0::2, 1]
        out_bayer[1::2, 1::2] = target_rgb[1::2, 1::2,0]

    return out_bayer


def raw_split(target_raw, raw_mode='RG;GB;', fill_num=None):
    ############################################### raw_modeの整形
    if raw_mode in ['bayer','Bayer','BAYER']:
        raw_mode = 'RG;GB;'
    elif raw_mode in ['quad_bayer','Quad_Bayer']:
        raw_mode = 'RRGG;RRGG;GGBB;GGBB;'

    raw_mode = raw_mode.replace(' ','')
    if raw_mode[-1]==';':
        raw_mode = raw_mode[0:-1]
    raw_mode_list = raw_mode.split(';')
    split_y = len(raw_mode_list)
    split_x = len(raw_mode_list[0])

    ############################################## 出力の調整
    if fill_num is None:
        out_img_list = [np.zeros((np.shape(target_raw)[0]//split_y, np.shape(target_raw)[1]//split_x)) for i in np.arange(split_y*split_x)]
        for yyy in np.arange(split_y):
            for xxx in np.arange(split_x):
                out_img_list[yyy*split_x+xxx] = target_raw[yyy::split_y,xxx::split_x]
    else:
        raster_raw_mode = []
        for raw_mode_y in raw_mode_list:
            for raw_mode_x in raw_mode_y:
                raster_raw_mode.append(raw_mode_x)
        unique_symbol = list(dict.fromkeys(raster_raw_mode))
        unique_symbol_dict = dict(zip(unique_symbol, np.arange(len(unique_symbol))))

        out_img_list = [np.ones((np.shape(target_raw)[0], np.shape(target_raw)[1]))*fill_num for i in np.arange(len(unique_symbol))]
        for yyy in np.arange(split_y):
            for xxx in np.arange(split_x):
                out_img_list[unique_symbol_dict[raster_raw_mode[yyy*split_x+xxx]]][yyy::split_y,xxx::split_x] = target_raw[yyy::split_y,xxx::split_x]

    return out_img_list


def index_map_merge(merge_target_img_list,index_map):
    merged_img = merge_target_img_list[0]
    for i in np.arange(1,len(merge_target_img_list)):
        merged_img[index_map==i] = merge_target_img_list[i][index_map==i]
    return merged_img
