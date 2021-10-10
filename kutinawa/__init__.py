"""
作成者：sikakusosi

使い方
1. 実行したい.pyファイルと同じ階層に「kutinawa」フォルダを置く
2. 実行したい.pyファイルで「import kutinawa as wa」とインポート
3. あとは通常のモジュールと同様に使用できる
"""
from .kutinawa_test import *
from .kutinawa_io import *
from .kutinawa_visualise import *
from .kutinawa_resize import *
from .kutinawa_filter import *
from .kutinawa_HQprocess import *
from .kutinawa_COLORprocess import *
from .kutinawa_num2num import *
from .kutinawa_chart import *
from .kutinawa_fileOP import *
from .kutinawa_depot import *

__all__ = ['kutinawa_test',
           'kutinawa_io',
           'kutinawa_visualise',
           'kutinawa_resize',
           'kutinawa_filter',
           'kutinawa_HQprocess',
           'kutinawa_COLORprocess',
           'kutinawa_num2num',
           'kutinawa_chart',
           'kutinawa_fileOP',
           'kutinawa_depot']




