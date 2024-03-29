"""
まとまりの無い関数の置き場。
そのうちどこかに移設統合する予定。
技術的まとまりができた時点で、既存のファイルを改変統合ないしは新たなファイルを作成し移設する。
(テスト不足とかではない。)
"""
import numpy as np
from scipy import signal
from scipy import ndimage
from .kutinawa_filter import fast_boxfilter,fast_box_variance_filter,generate_gaussian_filter,multi_filter
from .kutinawa_num2num import linear_LUT

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
    merged_img = merge_target_img_list[0].copy()
    for i in np.arange(1,len(merge_target_img_list)):
        merged_img[index_map==i] = merge_target_img_list[i][index_map==i]
    return merged_img


def image_power_spectrum2d(target_img):
    fft_data = np.fft.fft2(target_img)
    shifted_fft_data = np.fft.fftshift(fft_data)
    power_spectrum2d = np.abs(shifted_fft_data)*np.abs(shifted_fft_data)
    return power_spectrum2d

def image_power_spectrum1d(target_img):
    power_spectrum2d = image_power_spectrum2d(target_img)
    h = power_spectrum2d.shape[0]
    w = power_spectrum2d.shape[1]
    wc = w // 2
    hc = h // 2
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc).astype(int)
    power_spectrum1d = ndimage.sum(power_spectrum2d, r, index=np.arange(0, wc))
    return power_spectrum1d

def linear_envelope(target_data):
    target_data2 = np.squeeze(target_data)
    target_data2_mean = np.mean(target_data2)
    target_data2 = target_data2 - target_data2_mean
    peaks_p, _ = signal.find_peaks(target_data2, height=0)
    peaks_m, _ = signal.find_peaks(-target_data2, height=0)
    liner_env_p = linear_LUT(np.arange(0, len(target_data2)), peaks_p, target_data2[peaks_p], mode='clip')+target_data2_mean
    liner_env_m = linear_LUT(np.arange(0, len(target_data2)), peaks_m, target_data2[peaks_m], mode='clip')+target_data2_mean
    return liner_env_p,liner_env_m

def instant_phase_frequency(target_data):
    target_data2 = np.squeeze(target_data)
    target_data2 = target_data2-np.mean(target_data2)
    z = signal.hilbert(target_data2)
    inst_phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(inst_phase) / (2 * np.pi)
    return inst_phase,inst_freq


seven_fivedct = {'0': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 1, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '1': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '2': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '3': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '4': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '5': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '6': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '7': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '8': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '9': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '.': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'a': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'b': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'c': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'd': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'e': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'f': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'g': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'h': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'i': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'j': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'k': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 1, 0, 1, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'l': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'm': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'n': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'o': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'p': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'q': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'r': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 1, 0, 0,],
                                [ 0, 1, 1, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 's': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 't': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'u': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'v': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'w': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'x': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'y': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'z': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'A': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'B': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'C': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'D': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'E': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'F': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'G': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'H': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'I': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'J': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'K': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 1, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'L': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'M': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 0, 1, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'N': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 1, 1, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'O': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'P': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'Q': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'R': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'S': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'T': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'U': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'V': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 1, 0, 1, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'W': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 1, 0, 1, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'X': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'Y': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 'Z': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '_': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '+': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '-': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '*': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '/': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '=': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ':': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ';': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ')': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '(': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '[': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ']': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 1, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '<': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '>': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '&': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 1, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '@': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 1, 0, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '%': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 1, 0,],
                                [ 0, 1, 1, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 1, 0, 1, 1, 0,],
                                [ 0, 1, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '#': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 1, 1, 1, 1, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '"': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '^': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 1, 0, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '~': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 0, 1, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '!': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '?': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 1, 0, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 1, 0, 0, 0, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '|': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '`': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ',': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '{': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 1, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 1, 0, 0, 0, 0,],
                                [ 0, 0, 0, 1, 1, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 '}': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 1, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 0, 0, 1, 0, 0,],
                                [ 0, 0, 1, 1, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 ' ': np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,],
                                [ 0, 0, 0, 0, 0, 0, 0,], ]),
                 }
def generate_char_img(in_str):
    in_str_list = [c for c in str(in_str)]
    out_img = np.zeros((9,len(in_str_list)*7))
    for i, lll in enumerate(in_str_list):
        out_img[:,i*7:(i+1)*7] = seven_fivedct.get(lll, np.array([[ 0, 0, 0, 0, 0, 0, 0,],
                                                                  [ 0, 1, 0, 1, 0, 1, 0,],
                                                                  [ 0, 0, 1, 0, 1, 0, 0,],
                                                                  [ 0, 1, 0, 1, 0, 1, 0,],
                                                                  [ 0, 0, 1, 0, 1, 0, 0,],
                                                                  [ 0, 1, 0, 1, 0, 1, 0,],
                                                                  [ 0, 0, 1, 0, 1, 0, 0,],
                                                                  [ 0, 1, 0, 1, 0, 1, 0,],
                                                                  [ 0, 0, 0, 0, 0, 0, 0,], ]))
    return out_img

def direct_draw_roi(target_img,roi_pos,roi_num):
    target_img_roi = target_img.copy()
    target_img_roi = target_img_roi.astype(float)
    target_img_roi[roi_pos[0][0]:roi_pos[0][1], roi_pos[1][0],:] = 255
    target_img_roi[roi_pos[0][0]:roi_pos[0][1], roi_pos[1][1],:] = 255
    target_img_roi[roi_pos[0][0], roi_pos[1][0]:roi_pos[1][1],:] = 255
    target_img_roi[roi_pos[0][1], roi_pos[1][0]:roi_pos[1][1],:] = 255

    roi_num_img = (generate_char_img(roi_num)*255)
    target_img_roi[roi_pos[0][0]+2:roi_pos[0][0]+2+np.shape(roi_num_img)[0],roi_pos[1][0]+2:roi_pos[1][0]+2+np.shape(roi_num_img)[1],0] += roi_num_img
    target_img_roi[roi_pos[0][0]+2:roi_pos[0][0]+2+np.shape(roi_num_img)[0],roi_pos[1][0]+2:roi_pos[1][0]+2+np.shape(roi_num_img)[1],1] += roi_num_img
    target_img_roi[roi_pos[0][0]+2:roi_pos[0][0]+2+np.shape(roi_num_img)[0],roi_pos[1][0]+2:roi_pos[1][0]+2+np.shape(roi_num_img)[1],2] += roi_num_img
    return (np.clip(target_img_roi,0,255)).astype(np.uint8)


def multi_patch_picker(target_img,patch_LU_list,patch_size_list):
    """
    複数箇所の矩形領域（パッチ）を画像から取得し、list化して返す
    :param target_img: パッチを取得する画像
    :param patch_LU_list: 取得したいパッチの左上座標、[(y0,x0),(y1,x1),...]の形式で指定
    :param patch_size_list: 取得したいパッチのサイズ、[(y_size0,x_size0),(y_size1,x_size1),...]の形式 or (y_size,x_size)の形式で指定
                            (y_size,x_size)の形式の場合、すべての座標に対してそのサイズでの切り取りが行われる
    :return: list化されたパッチ、０次元目にそれぞれのパッチ、1,2次元目に縦横が格納される
    """
    if not isinstance(patch_LU_list,list):
        patch_LU_list = [patch_LU_list]

    if not isinstance(patch_size_list,list):
        patch_size_list = [patch_size_list for i in np.arange(len(patch_LU_list))]

    patch_list = [target_img[patch_LU[0]:patch_LU[0]+patch_size[0],patch_LU[1]:patch_LU[1]+patch_size[1]] for patch_LU,patch_size in zip(patch_LU_list,patch_size_list)]
    return patch_list


def multi_patch_stats(target_img,patch_LU_list,patch_size_list):
    """
    複数箇所の矩形領域（パッチ）を画像から取得し、統計量を返す
    :param target_img: パッチを取得する画像
    :param patch_LU_list: 取得したいパッチの左上座標、[(y0,x0),(y1,x1),...]の形式で指定
    :param patch_size_list: 取得したいパッチのサイズ、[(y_size0,x_size0),(y_size1,x_size1),...]の形式 or (y_size,x_size)の形式で指定
                            (y_size,x_size)の形式の場合、すべての座標に対してそのサイズでの切り取りが行われる
    :return: 各パッチのmean,std,min,max,median
    """
    patch_list = multi_patch_picker(target_img,patch_LU_list,patch_size_list)
    return np.mean(patch_list,axis=(1,2)),np.std(patch_list,axis=(1,2)),np.min(patch_list,axis=(1,2)),np.max(patch_list,axis=(1,2)),np.median(patch_list,axis=(1,2))


def weighted_least_squares(in_x, in_y, weight):
    """
    重み付き最小二乗法で、傾きと切片を求める
    :param in_x       : 入力x、一次元のlistもしくはarray形式
    :param in_y       : 入力y、一次元のlistもしくはarray形式
    :param weight     : 重み、一次元のlistもしくはarray形式
    　　　　             （注意：in_x, in_y, weightは同じ長さであること）
    :return grad      : 求められた傾き
    :return intercept : 求められた切片
    """
    X1 = np.sum(weight * in_x)
    X2 = np.sum(weight * in_x * in_x)
    W0 = np.sum(weight)
    Y1 = np.sum(in_y * weight)
    Z2 = np.sum(in_x * in_y * weight)
    grad = (X1 * Y1 - W0 * Z2) / (X1 * X1 - W0 * X2)
    intercept = (X1 * Z2 - X2 * Y1) / (X1 * X1 - W0 * X2)
    return grad, intercept


def least_squares(in_x, in_y):
    """
    最小二乗法で、傾きと切片を求める
    :param in_x       : 入力x、一次元のlistもしくはarray形式
    :param in_y       : 入力y、一次元のlistもしくはarray形式
    　　　　             （注意：in_x, in_yは同じ長さであること）
    :return grad      : 求められた傾き
    :return intercept : 求められた切片
    """
    in_w = np.ones_like(in_x)
    return weighted_least_squares(in_x, in_y, in_w)


def bs_noise_characteristic(target_img, seg_num=256, seg_min_max=(0, 0.9)):
    seg_one = (seg_min_max[1] - seg_min_max[0]) / seg_num

    mean_img = fast_boxfilter(target_img, 5, 5) / 5 / 5
    var_img = fast_box_variance_filter(target_img, 5, 5)

    labeled_img = np.clip((mean_img - seg_min_max[0]) // seg_one, 0, seg_num)
    label_min_var = ndimage.minimum(var_img, labels=labeled_img, index=np.arange(seg_num))
    label_pix_val = ndimage.mean(mean_img, labels=labeled_img, index=np.arange(seg_num))
    var_mean_ratio = (label_min_var - label_min_var[0]) / label_pix_val
    label_pix_val = label_pix_val[var_mean_ratio < np.sort(var_mean_ratio)[int(seg_num * 0.1)]]
    label_min_var = label_min_var[var_mean_ratio < np.sort(var_mean_ratio)[int(seg_num * 0.1)]]

    grad, intercept = weighted_least_squares(in_x=label_pix_val, in_y=label_min_var, weight=1 / (label_pix_val))

    return np.array([grad, intercept])


def harris_corner_detector(target_img,window_size,k,th):
    fil_sobel = [np.array([[ 1, 0,-1],
                          [ 2, 0,-2],
                          [ 1, 0,-1], ]),
                np.array([[-1,-2,-1],
                          [ 0, 0, 0],
                          [ 1, 2, 1],]),
                ]

    blur_img = ndimage.convolve(target_img, generate_gaussian_filter(window_size,np.min(window_size)/3))
    sobel_img = multi_filter(blur_img,fil_sobel)

    dx2 = sobel_img[0]*sobel_img[0]
    dy2 = sobel_img[1]*sobel_img[1]
    dxy = sobel_img[0]*sobel_img[1]

    Sx2 = ndimage.convolve(dx2,np.ones(window_size))
    Sy2 = ndimage.convolve(dy2,np.ones(window_size))
    Sxy = ndimage.convolve(dxy,np.ones(window_size))

    S_det = Sx2*Sy2-Sxy*Sxy
    S_tr  = Sx2+Sy2
    R_img = S_det - k * (S_tr * S_tr)

    corner_map = R_img>th
    return corner_map


def otsu_binarization_threshold(target_img):
    """
    大津の2値化アルゴリズムによって求められた閾値を返す
    target_img>otsu_binarization_threshold(target_img)とすることで2値化
    :param target_img: 入力画像、intであること
    :return: 大津の2値化アルゴリズムによって求められた閾値
    """
    target_hist,bin = np.histogram(target_img,np.arange(np.min(target_img),np.max(target_img)+2))
    num_left = np.cumsum(target_hist)
    num_right = num_left[-1] - num_left
    mean_left = np.cumsum(target_hist * bin[:-1])
    mean_right = (mean_left[-1] - mean_left)
    mean_left[num_left!=0] = mean_left[num_left!=0] / num_left[num_left!=0]
    mean_left[num_left==0] = 0
    mean_right[num_right!=0] = mean_right[num_right!=0] / num_right[num_right!=0]
    mean_right[num_right==0] = 0
    class_var = num_left*num_right*(mean_left-mean_right)*(mean_left-mean_right)
    class_var[np.isnan(class_var)]=0
    th = bin[np.argmax(class_var)+1]
    return th

macbeth_color = ['#735244','#c29682','#627a9d','#576c43','#8580b1','#67bdaa',
                 '#d67e2c','#505ba6','#c15a63','#5e3c6c','#9dbc40','#e0a32e',
                 '#383d96','#469449','#af363c','#e7c71f','#bb5695','#0885a1',
                 '#f3f3f2','#c8c8c8','#a0a0a0','#7a7a79','#555555','#343434',]


def in_polygon(polygon, tgt_point):
    """
    2次元平面上で、ある点が指定した多角形の中にあるかどうかを判定する
    crossing number algorithm の実装

    ○以下のコードと等価
        line_cross_cnt = np.zeros(np.shape(tgt_point)[0])
        for i in np.arange(np.shape(polygon)[0] - 1):
            y_flag = (tgt_point[:, 1] < polygon[i, 1]) != (tgt_point[:, 1] < polygon[i + 1, 1])
            x_flag = (tgt_point[:, 0] < (polygon[i + 1, 0] - polygon[i, 0]) * (tgt_point[:, 1] - polygon[i, 1]) / (polygon[i + 1, 1] - polygon[i, 1]) + polygon[i, 0])
            line_cross_cnt += (y_flag * x_flag).astype(int)
        in_flag = np.mod(line_cross_cnt,2)!=0

    ○例
    import numpy as np
    import kutinawa as wa

    polygon = np.array([[0.210, 0.320],
                        [0.220,0.315],
                        [0.395,0.135],
                        [0.255,0.250], ])# (x,y)
    tgt_point = wa.generate_random_array((10000,10000))

    in_flag = wa.in_polygon(polygon,tgt_point)
    wa.plotq([polygon[:, 0], tgt_point[:, 0]*in_flag, tgt_point[:, 0]*np.abs(1-in_flag)], [polygon[:, 1], tgt_point[:, 1]*in_flag, tgt_point[:, 1]*np.abs(1-in_flag)],
             marker=['','o','x'], linewidth=[3,0,0])

    :param polygon: 2次元array形式、[[x1,y1],[x2,y2],...]のように多角形の座標を時計回りで記述
    :param tgt_point: 2次元array形式、[[x1,y1],[x2,y2],...]のように判定したい座標を記述
    :return: tgt_pointがpolygon内にあるかどうかのbool
    """

    def loop_array(target_array):
        return np.concatenate([target_array, target_array[0][np.newaxis, :]], axis=0)

    polygon = loop_array(polygon)

    y_flag = (np.tile(tgt_point[:, 1], (np.shape(polygon)[0], 1)) < (polygon[:, 1])[:, np.newaxis])
    y_flag = y_flag[:-1] != y_flag[1:]
    x_flag = tgt_point[:, 0] < ((polygon[1:, 0] - polygon[:-1, 0])[:, np.newaxis] * (np.tile(tgt_point[:, 1], (np.shape(polygon)[0] - 1, 1)) - (polygon[:-1, 1])[:, np.newaxis]) / (polygon[1:, 1] - polygon[:-1, 1])[:, np.newaxis]) + (polygon[:-1, 0])[:, np.newaxis]
    xy_flag = (y_flag * x_flag).astype(int)
    line_cross_cnt = np.sum(xy_flag, axis=0)
    in_flag = np.mod(line_cross_cnt, 2) != 0

    return in_flag


def isleft(l,r,tgt_points):
    """
    点tgt_pointsが直線lrの"左側"にあるかどうか判定する

    例：
    import kutinawa as wa
    tgt_points = wa.generate_random_array((10000, 10000))
    l = [0.25, 0.25]
    r = [0.75, 0.75]
    left_flag = isleft(l, r, tgt_points)

    wa.plotq([np.array([l[0],r[0]]),tgt_points[:, 0] * left_flag, tgt_points[:, 0] * np.abs(1 - left_flag)],
             [np.array([l[1],r[1]]),tgt_points[:, 1] * left_flag, tgt_points[:, 1] * np.abs(1 - left_flag)],
             marker=['', 'o', 'x'], linewidth=[3, 0, 0])
    :param l:
    :param r:
    :param tgt_points:
    :return:
    """
    z = ((r[0] - l[0]) * (tgt_points[:,1] - l[1])) - ((tgt_points[:,0] - l[0]) * (r[1] - l[1]))
    return z > 0


def upper_hull(l, r, tgt_points):
    if len(tgt_points) == 0:
        return []

    result_points = []

    # 直線lr：ax+by+c=0 と 点tgt_pointsの距離算出
    a = (r[1] - l[1]) / (r[0] - l[0])
    b = -1
    c = l[1] - a * l[0]
    dist = np.abs(tgt_points[:,0]*a+tgt_points[:,1]*b+c) #/np.sqrt(a**2+b**2)

    # 直線lrの"左側"にいるかどうか判定
    left_flag = isleft(l,r,tgt_points)

    # 直線lrの"左側"にいる&直線lrから最も遠い点、なければNone
    farthest_point = tgt_points[np.argmax(dist*left_flag)] if np.sum(left_flag)>0 else []
    result_points.append(farthest_point)

    # 再帰で呼びだし
    result_points = result_points + upper_hull(l, farthest_point, tgt_points[left_flag])
    result_points = result_points + upper_hull(farthest_point, r, tgt_points[left_flag])

    return result_points


def quick_hull(target_points):
    """
    与えられた複数のx,y座標から、QuickHullアルゴリズムにより凸包を求める。
    凸包を構成する座標を時計回りにソートした状態で返す。
    :param target_points: (x,y)座標のリスト。[[x0,y0],[x1,y1],[x2,y2],...]
    :return: 
    """
    target_points = np.array(target_points)
    l_point,r_point = target_points[np.argmin(target_points,axis=0)[0]],target_points[np.argmax(target_points,axis=0)[0]]
    result_points = [l_point,r_point]

    temp_result   = upper_hull(l_point, r_point, target_points)
    temp_result   = [t for t in temp_result if len(t)>0]
    result_points = result_points+temp_result

    temp_result   = upper_hull(r_point, l_point, target_points)
    temp_result   = [t for t in temp_result if len(t)>0]
    result_points = result_points+temp_result

    weight_point  = np.mean(result_points, axis=0)
    result_points = result_points[np.argsort(-np.arctan2(np.array(result_points)[:, 1] - weight_point[1], np.array(result_points)[:, 0] - weight_point[0]))]

    return result_points
