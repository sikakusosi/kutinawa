"""
数値変換に関連する関数群.
LUT,数値を別の表現へ変換,固定小数対応キャスト処理等を集約する。
"""
import numpy as np
import math

######################################################################################################################## LUT
def linear_LUT(input, x, y, mode='nonclip'):
    input2 = np.float64(input)
    output = input2.copy()
    if mode == 'clip':
        output[input2 > x[-1]] = y[-1]
        output[input2 < x[0]] = y[0]

    for i in np.arange(len(x) - 1):
        grad = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        output[(input2 >= x[i]) & (input2 <= x[i + 1])] = (input2[(input2 >= x[i]) & (input2 <= x[i + 1])] - x[i]) * grad + y[i]

    return output

######################################################################################################################## 数値を別の表現へ変換
def dB_to_ratio(in_dB):
    """
    dBを比に変換して出力する。
    工率の量の比を出力する（ in_dB=10log10(return) ）ことに注意。
    センサーゲインの場合はこれの出力を√、PSNRの場合そのまま。
    :param in_dB:
    :return:比
    """
    return np.power(10, in_dB / 10)

def ratio_to_dB(in_ratio):
    """
    比をdBに変換して出力する。
    工率の量の比を出力する（ return=10log10(in_ratio) ）ことに注意。
    センサーゲインの場合は入力を2乗しておく。
    :param in_ratio:
    :return:dB表記の量
    """
    return 10*np.log10(in_ratio)

# convert RGB tuple to hexadecimal code
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# convert hexadecimal to RGB tuple
def hex_to_dec(hex):
    red = ''.join(hex.strip('#')[0:2])
    green = ''.join(hex.strip('#')[2:4])
    blue = ''.join(hex.strip('#')[4:6])
    return (int(red, 16), int(green, 16), int(blue,16))

########################################################################################################################
def round_decimal_point(target_array, decimal_point):
    p = np.power(10.0, decimal_point)
    s = np.copysign(1, target_array)
    return (s * target_array * p * 2 + 1) // 2 / p * s

def round_effective_digit(target_array,effective_digit):
    return round_decimal_point(target_array, effective_digit - ((np.log10(np.abs(target_array))).astype(int) + 1))

def ceil_effective_digit(target_array,effective_digit):
    decimal_point = effective_digit - ((np.log10(np.abs(target_array))).astype(int) + 1)
    p = np.power(10.0, decimal_point)
    return np.ceil(target_array * p)/p

######################################################################################################################## 固定小数を使うための丸め関数
def cast_round(input,bit_comp):
    """
    ハードで行われる四捨五入(0.5は1に、-0.5は0にする)を再現するための関数
    指定bit長にデータを整形し、floatで出力
    bit長オーバー分はクリップ
    :param input: ndarray
    :param bit_comp: ['符号有無(u or s)', 整数部bit長, 小数部bit長]
    :return: 整形された信号(float)
    """
    out_val = input + (1 / np.power(2, bit_comp[2] + 1))
    out_val = (out_val * np.power(2, bit_comp[2] + 1)) // 2 / np.power(2, bit_comp[2])

    integer_max = np.power(2, bit_comp[1])
    decimal_max = 1-1/np.power(2, bit_comp[2])

    minus_val_min = 0 if bit_comp[0] == 'u' else -integer_max-decimal_max
    plus_val_max = integer_max+decimal_max-1
    out_val = np.clip(out_val,minus_val_min,plus_val_max)

    return out_val

def cast_floor(input,bit_comp):
    """
    指定bit長にデータを整形し、doubleで出力
    bit長オーバー分はクリップ
    :param input: ndarray
    :param bit_comp: ['符号有無', 整数部数値, 小数部数値]
    :return: 整形された信号(double)
    """
    integer_max = np.power(2, bit_comp[1])
    decimal_max = 1-1/np.power(2, bit_comp[2])

    minus_val_min = 0 if bit_comp[0] == 'u' else -integer_max-decimal_max
    plus_val_max = integer_max+decimal_max-1
    shift_gain = np.power(2, bit_comp[2])

    return np.floor( np.clip(input, minus_val_min, plus_val_max) * shift_gain)/shift_gain


def cast_evenround(input,bit_comp):
    """
    ハードで行われる四捨五入(0.5は1に、-0.5は0にする)を再現するための関数
    指定bit長にデータを整形し、floatで出力
    bit長オーバー分はクリップ
    :param input: ndarray
    :param bit_comp: ['符号有無(u or s)', 整数部bit長, 小数部bit長]
    :return: 整形された信号(float)
    """
    out_val = np.round(input*(2**bit_comp[2]))/(2**bit_comp[2])

    integer_max = np.power(2, bit_comp[1])
    decimal_max = 1-1/np.power(2, bit_comp[2])

    minus_val_min = 0 if bit_comp[0] == 'u' else -integer_max-decimal_max
    plus_val_max = integer_max+decimal_max-1
    out_val = np.clip(out_val,minus_val_min,plus_val_max)

    return out_val

######################################################################################################################## exif解釈用
def exif_ShutterSpeedValue_to_denominator(exif_ShutterSpeedValue):
    return math.ceil(np.power(2,exif_ShutterSpeedValue))