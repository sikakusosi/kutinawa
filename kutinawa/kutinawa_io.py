"""
画像入出力の関数群.
"""
import os
import datetime
import csv

import numpy as np
from PIL import Image

from .kutinawa_resize import raster_2d_to_1d

##########################################################　IO系
def imread(target_img_path):
    return np.array(Image.open(target_img_path)).astype(float)

def imwrite(target_img,save_path):
    Image.fromarray(target_img).save(save_path)
    pass

def imread_raw(target_img_path, height, width, mode=np.uint16):
    """
    modeで定義された型でraw(バイナリ)画像を読み込む
    :param target_img_path  : 読み込む画像のパス(ファイル名、拡張子含む)
    :param height           : 画像の高さ
    :param width            : 画像の幅
    :param mode             : 読み込みに使用する型、numpyで規定されている型であること
    :return                 : 読み込まれた画像、2次元ndarray
    """
    img = np.fromfile(target_img_path, mode).reshape([height, width])
    return img

def imwrite_raw(target_img, save_path, mode=np.uint16):
    """
    modeで定義された型でキャストしてraw画像を書き出す
    :param target_img       : 書き出す画像、2次元ndarray
    :param save_path        : 保存したいパス(ファイル名、拡張子含む)
    :param mode             : 書き出しに使用する型、numpyで規定されている型であること
    :return                 : なし
    """
    if type(target_img) == list:
        target_img = np.array(target_img)

    target_img = target_img.astype(mode)
    target_img.tofile(save_path)
    pass


def imread_raw16bin(target_img_path):
    """
    raw16形式のバイナリ
    16bitのバイナリで先頭2画素分に幅、高の数値が入っている
    """
    img = np.fromfile(target_img_path, np.uint16)
    img = img[2:].reshape([img[1], img[0]])
    return img

def imread_csv(target_img_path, delimiter=',', skip_header=0):
    """
    csvファイルを２次元ndarrayとして読み込む
    :param target_img_path  : 読み込むcsvのパス
    :param delimiter        : csvの区切り文字
    :param skip_header      : csvのheaderの読み飛ばしたい行数
    :return                 : 読み込まれた画像、2次元ndarray
    """
    return np.genfromtxt(target_img_path, delimiter=delimiter, skip_header=skip_header)

def imwrite_csv(target_img, save_path, delimiter=','):
    """
    ２次元ndarrayをcsvファイルとして書き出す
    :param target_img       : 書き出す画像、2次元ndarray
    :param save_path        : 保存したいパス(ファイル名、拡張子含む)
    :param delimiter        : csvの区切り文字
    :return                 : なし
    """
    np.savetxt(save_path, target_img, delimiter=delimiter)
    pass


def raw_to_tiff(target_img_path, height, width, save_path):
    (Image.fromarray(imread_raw(target_img_path, height, width))).save(save_path)
    pass


def easy_dump(target_str, spe_idx='', time_stamp=False, lk=locals()):
    """
    簡易に、array形式の配列を画像出力するための関数。（テスト不十分、使用は自己責任で）
    :param target_str:　出力したい変数名を文字列で書くと変数名をファイル名として出力する。
                        出力ファイル名さえ変数名にしておけば、拡張子やパスを指定して出力することも可能。
                        例 : target_str = r'./folder1/folder2/変数名.jpg'
    :param spe_idx:     出力したい変数の一部のみ出力したい場合使用。
                        配列のindex指定と同じ記述で書く。
                        例 : spe_idx = [0::2,1:100,1]
    :param time_stamp:  ファイル名にタイムスタンプを書くか否か。
    :param lk:          関数を叩くコードの変数等をすべてリストアップしたdict。
                        基本はデフォルトから変えないこと。
    :return:            なし。
    """
    save_path = os.path.dirname(target_str)
    save_file = os.path.splitext(os.path.basename(target_str))[0]
    save_ext  = os.path.splitext(os.path.basename(target_str))[1]
    if save_ext=='':
        save_ext='.tiff'

    dt_now    = datetime.datetime.now()
    time_smp  = '_TS{0:%y}_{0:%m}_{0:%d}_{0:%H}_{0:%M}_{0:%S}'.format(dt_now)
    print(save_path,save_file,save_ext,time_smp)

    save_spe_idx = spe_idx.translate(str.maketrans({':': 'i'}))
    if time_stamp:
        save_pfe = os.path.join(save_path,save_file+save_spe_idx+time_smp+save_ext)
    else:
        save_pfe = os.path.join(save_path,save_file+save_spe_idx+save_ext)


    if save_ext=='raw':
        print(save_pfe)
        # wa.imwrite_raw(lk[save_file],save_pfe)
        imwrite_raw(eval("lk[save_file]"+spe_idx),save_pfe)
    else:
        print(save_pfe)
        # wa.imwrite(lk[save_file],save_pfe)
        imwrite(eval("lk[save_file]"+spe_idx),save_pfe)
    pass



def ppm_read(target_img_path):
    """
    P2もしくはP3のPPMファイルを読む関数
    空き行とかあるとだめ、当座の間に合わせの関数
    :param target_img_path:
    :return:
    """
    with open(target_img_path) as f:
        list_str = f.readlines()
        width, height = list_str[1].split()
        width = int(width)
        height = int(height)
        max_val, offset_str = list_str[2].split()
        offset_val = int(offset_str.split('=')[1])

        if list_str[0] == 'P2\n':
            list_int = [int(s) for s in list_str[3:]]
            img = np.reshape(list_int,[height,width]) - offset_val
        elif list_str[0] == 'P3\n':
            list_int = np.array([(int(s.split()[0]),
                                  int(s.split()[1]),
                                  int(s.split()[2])) for s in list_str[3:]]) - offset_val
            img = np.zeros((height,width,3))
            img[:, :, 0] = np.reshape(list_int[:, 0], [height, width])
            img[:, :, 1] = np.reshape(list_int[:, 1], [height, width])
            img[:, :, 2] = np.reshape(list_int[:, 2], [height, width])

    return img



def ppm_write(target_img, save_path, min_val, max_val):
    if np.shape(np.shape(target_img))[0] == 2 or np.shape(target_img)[2] == 1:
        print("P2")
        if min_val<0:
            offset_val = -min_val
        else:
            offset_val = 0

        ppm_txt = "P2\n"+str(np.shape(target_img)[1])+" "+str(np.shape(target_img)[0])+"\n"+str(max_val+offset_val)+" #offset="+str(offset_val)+"\n"
        ppt_val = '\n'.join(map(str, raster_2d_to_1d((target_img + offset_val).astype(int)).tolist()))

    elif np.shape(target_img)[2] == 3:
        #後で書く
        print("P3")
        if min_val<0:
            offset_val = -min_val
        else:
            offset_val = 0

        ppm_txt = "P3\n"+str(np.shape(target_img)[1])+" "+str(np.shape(target_img)[0])+"\n"+str(max_val+offset_val)+" #offset="+str(offset_val)+"\n"
        target_img = (target_img + offset_val).astype(int)
        temp_val = np.hstack([raster_2d_to_1d(target_img[:, :, 0])[:, np.newaxis],
                              raster_2d_to_1d(target_img[:, :, 1])[:, np.newaxis],
                              raster_2d_to_1d(target_img[:, :, 2])[:, np.newaxis]])
        ppt_val = '\n'.join([' '.join(map(str, temp_val[i, :].tolist())) for i in np.arange(np.shape(temp_val)[0])])
    else:
        print("error!")

    with open(save_path, mode='w') as f:
        f.write(ppm_txt+ppt_val)

    pass


