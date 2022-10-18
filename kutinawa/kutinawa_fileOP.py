"""
ファイルやフォルダの操作を行う関数群.
"""
import os
import glob
import pathlib
import platform
import datetime

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


def creation_date(filepath):
    """
    現在実行中のOSに関わらず、ファイルの作成日時を返す
    （Mac以外のUNIX系は作成日の取得ができない(os.stat(path_to_file).st_birthtimeが通らない)ので、最終更新日を返す）
    :param filepath: 作成日時を取得したいファイルのパス
    :return: ファイル作成時間
    """
    if platform.system() == 'Windows':
        return os.path.getctime(filepath)
    else:
        stat = os.stat(filepath)
        try:
            return stat.st_birthtime
        except AttributeError:
            # Mac以外のUNIX系は作成日の取得ができない(os.stat(path_to_file).st_birthtimeが通らない)ので、最終更新日を返す
            return stat.st_mtime


def listdir_sort(target_dir,target_exp='',subfolder=False,sort_by='name'):
    """
    指定したディレクトリ内の、指定した拡張子のファイルをソートした文字列リストの状態で返す。
    :param target_dir: 対象のディレクトリ
    :param target_exp: 対象の拡張子
    :param subfolder: サブフォルダ以下の検索を行うか否か
    :param sort_by: ソートする基準
    :return:ソート済みのファイルパス文字列のリスト
    """
    if subfolder:
        target_path_list = list(pathlib.Path(target_dir).glob('**/*' + target_exp))
    else:
        target_path_list = list(pathlib.Path(target_dir).glob('*' + target_exp))

    target_path_list = [str(target_path) for target_path in target_path_list]
    if sort_by=='name':
        target_path_list.sort()
    elif sort_by=='create_date':
        create_date_path_list = [[datetime.datetime.fromtimestamp(creation_date(target_path)), target_path] for target_path in target_path_list]
        create_date_path_list.sort(reverse=True)
        target_path_list = [create_date_path[1] for create_date_path in create_date_path_list]

    return target_path_list
















