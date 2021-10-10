"""
ファイルやフォルダの操作を行う関数群.
"""
import os

def makedirs_plus(dir_path, permission=0o2777):
    '''
    dirの存在を確認して、なかったら作成
    :param dir_path:作成したいdirパス
    :param permission:与えたいパーミッション、デフォは全開放
    :return:なし
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.chmod(dir_path, permission)
    pass




