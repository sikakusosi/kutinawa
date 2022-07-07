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

