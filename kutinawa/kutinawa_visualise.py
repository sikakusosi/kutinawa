"""
表示系の関数群.
"""
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .kutinawa_num2num import rgb_to_hex
from .kutinawa_io import imread


def scalar_to_color(value,mode):
    """
    0-1の値から、RGB値を返す
    疑似カラー生成用関数
    :param value:
    :return:
    """
    value = 6.0 * value
    if value <= 0:
        color = [ 1.0, 0.0, 0.0 ]
    elif value <= 1.0:
        color = [ 1.0, value, 0.0 ]
    elif value <= 2.0:
        value = ( value - 1.0 )
        color = [ 1.0 - value, 1.0, 0.0 ]
    elif value <= 3.0:
        value = ( value - 2.0 )
        color = [ 0.0, 1.0, value ]
    elif value <= 4.0:
        value = ( value - 3.0 )
        color = [ 0.0, 1.0 - value, 1.0 ]
    elif value <= 5.0:
        value = ( value - 4.0 )
        color = [ value, 0.0, 1.0 ]
    elif value <= 6.0:
        value = ( value - 5.0 )
        color = [ 1.0, 0.0, 1.0-value ]
    else:
        color = [ 1.0, 0.0, 0.0 ]

    if mode=='hex':
        color = rgb_to_hex((np.array(color)*255).astype(np.int))
    elif mode=='RGBa':
        color = np.array(color.append(0.8))
    else:
        color = np.array(color)

    return color


def imagesc(img, colormap='viridis', colorbar='on', coloraxis=[0,0], mode='view', create_figure='on', subplot_mode='off', val_view='off',fig_title=''):
    """
    :param img:
    :param colormap:
    :param colorbar:
    :param mode:
    :param create_figure:
    :param subplot:
    :return:
    """

    if mode=='view':
        img = np.squeeze(img)  # imgの余計な次元を削除

        # img_min = np.nanmin(img[:])
        # img_max = np.nanmax(img[:])
        img_min = np.nanmin(img)
        img_max = np.nanmax(img)

        if coloraxis[0]>=coloraxis[1]:
            caxis = [img_min, img_max]
        else:
            caxis = coloraxis

        # 表示系
        if create_figure=='on':# 新規figure作成
            fig = plt.figure()

        if subplot_mode=='off':# subplotなし
            ax = plt.axes()
            plt.imshow(img, interpolation='nearest', cmap=colormap, vmin=img_min,vmax=img_max)  # 補完方法をnearestにして勝手に平滑化されることを防止(もしかしたらいらないかも)
            plt.clim(caxis[0], caxis[1])
            plt.title(fig_title)
            if colorbar == 'on':# colorbar表示
                plt.colorbar()
            if val_view=='on':# 画素値を重畳テキスト表示(小数2桁まで)
                ys, xs = np.meshgrid(range(img.shape[0]), range(img.shape[1]), indexing='ij')
                for (x, y, val) in zip(xs.flatten(), ys.flatten(), img.flatten()):
                    plt.text(x, y, '{0:.2f}'.format(val), horizontalalignment='center', verticalalignment='center', )
                plt.show()

        else:# subplot表示
            img_num = len(img)# 0次元目の要素数
            sub_x = np.ceil(np.sqrt(img_num)).astype(int)
            sub_y = np.ceil(img_num/sub_x).astype(int)

            for sub_num in np.arange(0,img_num):
                if sub_num==0:
                    first_ax = plt.subplot(sub_y, sub_x, sub_num+1)# subplot間の軸合わせ(linkaxis)のために最初に描画ずる画像の軸取得
                else:
                    plt.subplot(sub_y, sub_x, sub_num + 1,sharex=first_ax,sharey=first_ax)# subplot間の軸合わせ(linkaxis)

                plt.imshow(img[sub_num], interpolation='nearest', cmap=colormap, vmin=img_min,vmax=img_max)  # 補完方法をnearestにして勝手に平滑化されることを防止(もしかしたらいらないかも)
                plt.clim(caxis[0], caxis[1])
                plt.title(fig_title)
                if colorbar == 'on':# colorbar表示
                    plt.colorbar()
                if val_view == 'on':# 画素値を重畳テキスト表示(小数2桁まで)
                    ys, xs = np.meshgrid(range(img.shape[0]), range(img.shape[1]), indexing='ij')
                    for (x, y, val) in zip(xs.flatten(), ys.flatten(), img.flatten()):
                        plt.text(x, y, '{0:.2f}'.format(val), horizontalalignment='center', verticalalignment='center', )
                fig.subplots_adjust(wspace=0)# subplotを最密に表示

            plt.show()

    pass


def caxis(cmin,cmax):
    plt.clim(cmin,cmax)
    plt.gca().figure.canvas.draw()
    pass

def xlim(xmin,xmax):
    plt.gca().set_xlim(xmin,xmax)
    plt.gca().figure.canvas.draw()
    pass

def ylim(ymin,ymax):
    plt.gca().set_ylim(ymin,ymax)
    plt.gca().figure.canvas.draw()
    pass




def imageq(target_img_list, coloraxis=(0,0), colormap='viridis', colorbar=True, val_view=False, view_mode='tile', cross_cursor=False,
           ctrl_func_dict1=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict2=r"ndimage.convolve(target_img,wa.generate_gaussian_filter((5,5),0.5),mode='mirror')",
           ctrl_func_dict3=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict4=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict5=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict6=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict7=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict8=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           ctrl_func_dict9=r"wa.fast_boxfilter(target_img=target_img,fil_h=5,fil_w=5)/25",
           singlecoloraxis=True,
           ):
    """
    ショートカットキーで色々できる、画像ビューワー
    :param target_img_list: 表示したい画像。
                            2次元list内に、1つ以上のndarray形式の画像を格納して渡す。
                            2次元listの0次元目が縦、1次元目が横に対応して表示される。
                            例：target_img_list=[[img_a, img_b],
                            　　                 [img_c, img_d]]
                                表示は下記のようになる。
                                ┌─────────────────────┐
                                │ ┌───────┐ ┌───────┐ │
                                │ │ img_a │ │ img_b │ │
                                │ └───────┘ └───────┘ │
                                │ ┌───────┐ ┌───────┐ │
                                │ │ img_c │ │ img_d │ │
                                │ └───────┘ └───────┘ │
                                └─────────────────────┘
                            略記法として、1次元listに1つ以上のndarray形式画像を格納して渡す / 1つのndarray形式の画像をそのまま渡す 事が可能。
                            例えば、入力を [img_a, img_b] とした場合は、[[img_a, img_b]]と等価となる。
    :param singlecoloraxis:
    :param coloraxis:       表示に使う疑似カラーの範囲。
                            (最小値,最大値)という構成のtuple　もしくは　2次元リストに格納された同様のtupleを受け付ける。

    :param colormap:
    :param colorbar:
    :param val_view:
    :param view_mode:
    :param cross_cursor:
    :return:
    """
    print("""
    ======= kutinawa imageq =======
    ------------ ビューイング系操作 ------------ 
    Drag                    : 画像の移動
    shift+Drag              : 画像の部分拡大
    Double click            : 画像全体表示
    Click on the image      : クリックした画像を”着目画像”に指定
    ------------ 画像比較 ------------ 
    D                       : ”着目画像”と他の画像の差分を表示
    ------------ clim 調整 ------------ 
    A                       : ”着目画像”の現在描画されている領域でclimを自動スケーリング
    W                       : 全画像の最大-最小を用いて全画像のclimを設定
    left(←),right(→)(+alt)  : ”着目画像”のclim上限を1%小さく(<),下限を1%大きく(>)、(+alt)時はclim上限を1%大きく(<),下限を1%小さく(>)
    up(↑),down(↓)           : ”着目画像”のclim範囲を1% 正(up),負(down)側にずらす
    S                       : ”着目画像”のclimを他の画像にも同期
    ------------ line・ROIを用いた解析 ------------ 
    i, -                    : キー押下時のマウス位置における、縦(i),横(-)方向のラインプロファイルをを別ウィンドウで表示
    r (+alt)                : キー押下時のマウス位置を左上としたROIを設定 (ROIサイズをデフォルトサイズ(11x11)に戻し、画像外に移動)
    >, < (+alt)             : ROIサイズの水平(>),垂直(<)拡大(縮小(+alt))を行う
    I, =                    : ROIの水平範囲を平均した縦(I),垂直範囲を平均した横(=)方向のラインプロファイルをを別ウィンドウで表示
    m                       : ROI内画素の画素値を別ウィンドウで表示
    h, H                    : ROI内(h)、表示範囲内(H)の画素値ヒストグラム表示 
    ------------ 画像書き出し、読み込み ------------ 
    P                       : 現在のfigureをPNGで保存
    ctrl+v                  : コピーした画像を着目画像領域に貼り付け
    """)

    ctrl_func_dict = {'ctrl+1':ctrl_func_dict1,
                      'ctrl+2':ctrl_func_dict2,
                      'ctrl+3':ctrl_func_dict3,
                      'ctrl+4':ctrl_func_dict4,
                      'ctrl+5':ctrl_func_dict5,
                      'ctrl+6':ctrl_func_dict6,
                      'ctrl+7':ctrl_func_dict7,
                      'ctrl+8':ctrl_func_dict8,
                      'ctrl+9':ctrl_func_dict9,}

    mplstyle.use('fast')
    plt.interactive(False)
    fig = plt.figure()
    val_fig = plt.figure()


    ############################### 必ず2次元listの各要素に画像が入ってる状態にする
    if isinstance(target_img_list,list)==False:
        target_img_list = [[target_img_list]]
    elif isinstance(target_img_list[0],list)==False:
        target_img_list = [target_img_list]

    ############################### 画像のプロファイル整理
    class all_img_property:
        sub_y_size = 0
        sub_x_size = 0

        img_minmax_list = []
        img_hw_list = []

        all_img_num = 0
        all_img_min = np.Inf
        all_img_max = -np.Inf
        all_img_h_max = 0
        all_img_w_max = 0

        def init_aip(self,target_img_list):
            self.sub_y_size = len(target_img_list)
            temp = []

            for y in range(self.sub_y_size):
                temp.append(len(target_img_list[y]))
                for x in range(len(target_img_list[y])):
                    self.all_img_num = self.all_img_num + 1
                    self.img_minmax_list.append([np.nanmin(target_img_list[y][x]),np.nanmax(target_img_list[y][x])])
                    self.img_hw_list.append([np.shape(target_img_list[y][x])[0],np.shape(target_img_list[y][x])[1]])

            self.sub_x_size = np.max(temp)
            self.all_img_min = np.min(np.array(self.img_minmax_list)[:,0])
            self.all_img_max = np.max(np.array(self.img_minmax_list)[:,1])
            self.all_img_h_max = np.max(np.array(self.img_hw_list)[:,0])
            self.all_img_w_max = np.max(np.array(self.img_hw_list)[:,1])
            pass

        def update_aip(self,target_img,i):
            self.img_minmax_list[i] = [np.nanmin(target_img),np.nanmax(target_img)]
            self.img_hw_list[i] = [np.shape(target_img)[0],np.shape(target_img)[1]]

            self.all_img_min = np.min(np.array(self.img_minmax_list)[:,0])
            self.all_img_max = np.max(np.array(self.img_minmax_list)[:,1])
            self.all_img_h_max = np.max(np.array(self.img_hw_list)[:,0])
            self.all_img_w_max = np.max(np.array(self.img_hw_list)[:,1])
            pass

        pass

    aip = all_img_property()
    aip.init_aip(target_img_list)

    ############################### subplot縦横サイズ計測 & 各画像に対する設定を保持する2次元list作成
    caxis = []
    if isinstance(coloraxis,tuple):#coloraxisが一つだけ → 初期はすべてのcoloraxisを同じで
        print(coloraxis)
        if coloraxis[0]>=coloraxis[1]:#(min,max)の指定が同数もしくは逆転している → coloraxis=(全画像の最小,全画像の最大)
            temp = (aip.all_img_min,aip.all_img_max)
        else:
            temp = (coloraxis[0],coloraxis[1])
        for y in range(aip.sub_y_size):
            caxis.append([temp for x in range(len(target_img_list[y]))])

    else:# coloraxisがlist=2つ以上 → 初期からバラバラ
        if isinstance(coloraxis[0],list)==False:
            coloraxis = [coloraxis]
        i = 0
        for y in range(aip.sub_y_size):
            caxis.append([])
            for x in range(len(target_img_list[y])):
                if coloraxis[0]>=coloraxis[1]:#(min,max)の指定が同数も育は逆転している → coloraxis=(全画像の最小,全画像の最大)
                    caxis.append( (aip.img_minmax_list[i][0], aip.img_minmax_list[i][1]) )
                else:
                    caxis.append( (coloraxis[y][x][0], coloraxis[y][x][1]) )
                i = i + 1

    ############################### 操作系関数群
    # main figが閉じられた場合の動作
    def main_figure_close(fig,val_fig):
        def close_event(event):
            val_fig.clf()
            plt.close(val_fig)
            pass
        fig.canvas.mpl_connect('close_event', close_event)
        pass

    # 十字カーソル
    def mouse_cross_cursor(fig,axhline,axvline):
        def mouse_cursor(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                for i,now_axh in enumerate(axhline):
                    now_axh.set_ydata(y)
                    axvline[i].set_xdata(x)
            fig.canvas.draw()
            pass
        fig.canvas.mpl_connect('motion_notify_event',mouse_cursor)
        pass

    # マウスクリック系イベント統合関数
    def mouse_click_event(fig,ax_list,im):
        def click_event(event):
            # ダブルクリックで最も大きい画像に合わせて表示領域リセット
            if event.dblclick:
                ax_list[0].set_xlim(-0.5, aip.all_img_w_max-0.5)
                ax_list[0].set_ylim(aip.all_img_h_max-0.5, -0.5)


            # クリックした画像を着目画像(current axes)に指定
            for ax_cand in ax_list:
                if event.inaxes == ax_cand:
                    fig.sca(ax_cand)
                    ax_cand.spines['bottom'].set_color("#2B8B96")
                    ax_cand.spines['top'].set_color("#2B8B96")
                    ax_cand.spines['left'].set_color("#2B8B96")
                    ax_cand.spines['right'].set_color("#2B8B96")
                    ax_cand.spines['bottom'].set_linewidth(4)
                    ax_cand.spines['top'].set_linewidth(4)
                    ax_cand.spines['left'].set_linewidth(4)
                    ax_cand.spines['right'].set_linewidth(4)
                else:
                    ax_cand.spines['bottom'].set_color("black")
                    ax_cand.spines['top'].set_color("black")
                    ax_cand.spines['left'].set_color("black")
                    ax_cand.spines['right'].set_color("black")
                    ax_cand.spines['bottom'].set_linewidth(1)
                    ax_cand.spines['top'].set_linewidth(1)
                    ax_cand.spines['left'].set_linewidth(1)
                    ax_cand.spines['right'].set_linewidth(1)
            fig.canvas.draw()
        fig.canvas.mpl_connect('button_press_event',click_event)
        pass

    # ツールを使わず移動、拡大
    def mouse_drag_move(fig,ax_list):
        mouse_data = {'x_s':0,
                      'y_s':0,
                      'xdata_s':0,
                      'ydata_s':0,
                      'inaxes_flag':False,
                      'shift_flag':False,
                      'zoom_rect':[]}

        for ax_num,ax_cand in enumerate(ax_list):
            mouse_data['zoom_rect'].append(patches.Rectangle(xy=(-1, -1),width=0,height=0,ec='#ff1493', fill=False))
            ax_cand.add_patch(mouse_data['zoom_rect'][-1])

        def button_press(event):
            if event.key=="shift":
                mouse_data['shift_flag']=True
            if event.inaxes:
                mouse_data['x_s'] = event.x
                mouse_data['y_s'] = event.y
                mouse_data['xdata_s'] = event.xdata
                mouse_data['ydata_s'] = event.ydata
                mouse_data['inaxes_flag'] = True
            pass

        def button_drag(event):
            if mouse_data['shift_flag'] and mouse_data['inaxes_flag']:
                if event.inaxes:
                    for ax_num,ax_cand in enumerate(ax_list):
                        mouse_data['zoom_rect'][ax_num].set_bounds((mouse_data['xdata_s'], mouse_data['ydata_s'],
                                                                    event.xdata-mouse_data['xdata_s'], event.ydata-mouse_data['ydata_s']))
                else:
                    for ax_num,ax_cand in enumerate(ax_list):
                        mouse_data['zoom_rect'][ax_num].set_bounds((-1,-1,0,0))
                    pass
                fig.canvas.draw()
            pass

        def button_release(event):
            if mouse_data['inaxes_flag']:
                ax_x_px = int( (ax_list[0].bbox.x1-ax_list[0].bbox.x0) )
                move_x = mouse_data['x_s'] - event.x
                lim_x = ax_list[0].get_xlim()
                ax_img_pix_x = lim_x[1]-lim_x[0]
                move_x_pix = move_x/ax_x_px*ax_img_pix_x

                ax_y_px = int( (ax_list[0].bbox.y1-ax_list[0].bbox.y0) )
                move_y = mouse_data['y_s'] - event.y
                lim_y = ax_list[0].get_ylim()
                ax_img_pix_y = lim_y[1]-lim_y[0]
                move_y_pix = move_y/ax_y_px*ax_img_pix_y

                if mouse_data['shift_flag']:
                    x_lim = np.sort([mouse_data['xdata_s'],mouse_data['xdata_s']-move_x_pix])
                    y_lim = np.sort([mouse_data['ydata_s'],mouse_data['ydata_s']-move_y_pix])
                    ax_list[0].set_xlim(x_lim[0],x_lim[1])
                    ax_list[0].set_ylim(y_lim[1],y_lim[0])
                else:
                    ax_list[0].set_xlim(lim_x[0]+move_x_pix,lim_x[1]+move_x_pix)
                    ax_list[0].set_ylim(lim_y[0]+move_y_pix,lim_y[1]+move_y_pix)
                fig.canvas.draw()

            mouse_data['x_s']=0
            mouse_data['y_s']=0
            mouse_data['xdata_s']=0
            mouse_data['ydata_s']=0
            mouse_data['inaxes_flag']=False
            mouse_data['shift_flag']=False
            for ax_num,ax_cand in enumerate(ax_list):
                mouse_data['zoom_rect'][ax_num].set_bounds((-1,-1,0,0))

            pass

        fig.canvas.mpl_connect('button_press_event',button_press)
        fig.canvas.mpl_connect('motion_notify_event', button_drag)
        fig.canvas.mpl_connect('button_release_event',button_release)
        pass

    # PNG保存
    def keyboard_shortcut_onlyPNG(fig):
        def keyboard_onlyPNG(event):
            if event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png')
            pass
        fig.canvas.mpl_connect('key_press_event',keyboard_onlyPNG)
        pass

    # climマニュアル操作のショートカットで使う共通関数
    def update_clim(ax_list,im_list,mode,gain=None,plusminus=None):
        if mode=='manual':
            for ax_num,ax_cand in enumerate(ax_list):
                if plt.gca() == ax_cand:
                    caxis_min,caxis_max = im_list[ax_num].get_clim()
                    diff = caxis_max-caxis_min
                    min_change = plusminus[0]*diff*gain
                    max_change = plusminus[1]*diff*gain
                    im_list[ax_num].set_clim(float(caxis_min)+min_change,float(caxis_max)+max_change)

        elif mode=='auto':
            for ax_num,ax_cand in enumerate(ax_list):
                if plt.gca() == ax_cand:
                    int_x = np.clip((np.array(plt.gca().get_xlim()) + 0.5).astype(int),0,None)
                    int_y = np.clip((np.array(plt.gca().get_ylim()) + 0.5).astype(int),0,None)
                    temp = im_list[ax_num].get_array()[int_y[1]:int_y[0],int_x[0]:int_x[1]]
                    im_list[ax_num].set_clim(np.min(temp),np.max(temp))

        elif mode=='sync':
            for ax_num,ax_cand in enumerate(ax_list):
                if plt.gca() == ax_cand:
                    caxis_min,caxis_max = im_list[ax_num].get_clim()
                    for num,now_im in enumerate(im_list):
                        now_im.set_clim(float(caxis_min),float(caxis_max))
                    break

        elif mode=='whole':
            for num,now_im in enumerate(im_list):
                now_im.set_clim(aip.all_img_min,aip.all_img_max)

        pass

    def make_lineprofile(ax_list,im_list,mode,xdata=None,ydata=None,roi_list=None):

        img_xlim = ax_list[0].get_xlim()
        img_ylim = ax_list[0].get_ylim()
        img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
        img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

        if (mode=='horizontal') or (mode=='vertical'):
            mouse_x, mouse_y = int(xdata + 0.5), int(ydata + 0.5)

        elif (mode=='horizontal_mean') or (mode=='vertical_mean'):
            roi_x_s,roi_y_s = roi_list[0].get_xy()
            roi_x_s,roi_y_s = np.clip(roi_x_s+0.5,0,None).astype(int),np.clip(roi_y_s+0.5,0,None).astype(int)
            roi_x_e,roi_y_e = (roi_x_s+roi_list[0].get_width()).astype(int),(roi_y_s+roi_list[0].get_height()).astype(int)

        ana_fig = plt.figure()
        plt_sub_ax = ana_fig.add_subplot(2,len(ax_list),(1,len(ax_list)),picker=True)
        img_sub_ax = []
        img_sub_im = []
        legend_str = []
        for ax_num,ax_cand in enumerate(ax_list):
            temp_img = im_list[ax_num].get_array()
            img_xlim2 = [np.clip(img_xlim[0],0,np.shape(temp_img)[1]),np.clip(img_xlim[1],0,np.shape(temp_img)[1])]
            img_ylim2 = [np.clip(img_ylim[0],0,np.shape(temp_img)[0]),np.clip(img_ylim[1],0,np.shape(temp_img)[0])]

            caxis_min,caxis_max = im_list[ax_num].get_clim()
            img_sub_ax.append(ana_fig.add_subplot(2,len(ax_list),len(ax_list)+ax_num+1,picker=True))
            imshow_block(ax=img_sub_ax[-1],
                         im=img_sub_im,
                         im_index=-1,
                         target_img=temp_img,
                         caxis_min=caxis_min,
                         caxis_max=caxis_max)
            img_sub_ax[-1].set_xlim(img_xlim)
            img_sub_ax[-1].set_ylim(img_ylim)
            img_sub_im[-1].set_clim(caxis_min,caxis_max)
            legend_str.append(str(ax_num))
            if mode=='horizontal':
                mouse_y2 = np.clip(mouse_y,0,np.shape(temp_img)[0])
                plt_sub_ax.plot(np.arange(img_xlim2[0],img_xlim2[1]),
                                temp_img[mouse_y2,img_xlim2[0]:img_xlim2[1]])
                img_sub_ax[-1].axhline(y=mouse_y2,color='pink')
            elif mode=='vertical':
                mouse_x2 = np.clip(mouse_x,0,np.shape(temp_img)[1])
                plt_sub_ax.plot(np.arange(img_ylim2[1],img_ylim2[0]),
                                temp_img[img_ylim2[1]:img_ylim2[0],mouse_x2])
                img_sub_ax[-1].axvline(x=mouse_x2,color='pink')
            elif mode=='horizontal_mean':
                roi_y_s2 = np.clip(roi_y_s,0,np.shape(temp_img)[0])
                roi_y_e2 = np.clip(roi_y_e,0,np.shape(temp_img)[0])
                plt_sub_ax.plot(np.arange(img_xlim2[0],img_xlim2[1]),
                                np.nanmean(temp_img[roi_y_s2:roi_y_e2,img_xlim2[0]:img_xlim2[1]],axis=0))
                img_sub_ax[-1].axhline(y=roi_y_s2,color='pink')
                img_sub_ax[-1].axhline(y=roi_y_e2,color='pink')
            elif mode=='vertical_mean':
                roi_x_s2 = np.clip(roi_x_s,0,np.shape(temp_img)[1])
                roi_x_e2 = np.clip(roi_x_e,0,np.shape(temp_img)[1])
                plt_sub_ax.plot(np.arange(img_ylim2[1],img_ylim2[0]),
                                np.nanmean(temp_img[img_ylim2[1]:img_ylim2[0]:,roi_x_s2:roi_x_e2],axis=1))
                img_sub_ax[-1].axvline(x=roi_x_s2-0.5,color='pink')
                img_sub_ax[-1].axvline(x=roi_x_e2-0.5,color='pink')

        plt_sub_ax.legend(legend_str)
        if mode=='horizontal':
            ana_fig.suptitle('lineprofile : x='+str(img_xlim[0])+'~'+str(img_xlim[1])+', y='+str(mouse_y))
        elif mode=='vertical':
            ana_fig.suptitle('lineprofile : x='+str(mouse_x)+', y='+str(img_ylim[1])+'~'+str(img_ylim[0]))
        elif mode=='horizontal_mean':
            ana_fig.suptitle('lineprofile : x='+str(img_xlim[0])+'~'+str(img_xlim[1])+', y(mean)='+str(roi_y_s)+'~'+str(roi_y_e))
        elif mode=='vertical_mean':
            ana_fig.suptitle('lineprofile : x(mean)='+str(roi_x_s)+'~'+str(roi_x_e)+', y='+str(img_xlim[0])+'~'+str(img_xlim[1]))

        ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
        ana_fig.show()
        keyboard_shortcut_onlyPNG(ana_fig)
        main_figure_close(fig,ana_fig)
        pass


    # keyboard_shortcut関数
    temp_state_refnum_clim = ["normal",0,]
    def keyboard_shortcut_sum(fig, ax_list, im_list):
        def keyboard_shortcut(event):
            print(event.key)
            ################################## image diffarence ##################################
            if event.key=='D':# diff
                if temp_state_refnum_clim[0] == 'normal':
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            temp_state_refnum_clim[1] = ax_num
                    for num,now_im in enumerate(im_list):
                        temp_state_refnum_clim.append((now_im.get_clim()))
                        if num!=temp_state_refnum_clim[1]:
                            set_img = im_list[temp_state_refnum_clim[1]].get_array() - now_im.get_array()
                            now_im.set_array(set_img)
                            caxis_min,caxis_max = np.min(set_img),np.max(set_img)
                            now_im.set_clim(caxis_min,caxis_max)
                            if caxis_min==caxis_max==0:
                                print('img'+str(temp_state_refnum_clim[1])+'(ref-img) and img' + str(num) + ' are identical.')
                            elif caxis_min==caxis_max!=0:
                                print('img'+str(temp_state_refnum_clim[1])+'(ref-img) and img' + str(num) + ' are identical except for the OFFSET component.')
                    temp_state_refnum_clim[0] = 'diff'

                elif temp_state_refnum_clim[0] == 'diff':
                    for num,now_im in enumerate(im_list):
                        if int(temp_state_refnum_clim[1])!=num:
                            now_im.set_array(im_list[int(temp_state_refnum_clim[1])].get_array() - now_im.get_array())
                            caxis_min,caxis_max = temp_state_refnum_clim[int(num+2)]
                            now_im.set_clim(float(caxis_min),float(caxis_max))
                    temp_state_refnum_clim[0] = 'normal'
                    temp_state_refnum_clim[1] = 0
                    del temp_state_refnum_clim[2:]

            ################################## update clim with AUTOMATIC adjustments ##################################
            elif event.key=='A':# auto clim
                update_clim(ax_list,im_list,mode='auto')
            elif event.key=='S':# sync clim
                update_clim(ax_list,im_list,mode='sync')
            elif event.key=='W':# set clim(<all image min>, <all image max>)
                update_clim(ax_list,im_list,mode='whole')

            ################################## update clim with MANUAL adjustments ##################################
            elif event.key=='left':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[0,-1])
            elif event.key=='alt+left':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[0,+1])
            elif event.key=='right':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[+1,0])
            elif event.key=='alt+right':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[-1,0])
            elif event.key=='ctrl+left':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[0,-1])
            elif event.key=='ctrl+alt+left':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[0,+1])
            elif event.key=='ctrl+right':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[+1,0])
            elif event.key=='ctrl+alt+right':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[-1,0])
            elif event.key=='up':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[+1,+1])
            elif event.key=='down':
                update_clim(ax_list,im_list,mode='manual',gain=0.01,plusminus=[-1,-1])
            elif event.key=='ctrl+up':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[+1,+1])
            elif event.key=='ctrl+down':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[-1,-1])

            ################################## line ROI analyze ##################################
            elif event.key=='-':
                if event.inaxes:
                    make_lineprofile(ax_list,im_list,mode='horizontal',xdata=event.xdata,ydata=event.ydata,roi_list=None)
            elif event.key=='i':
                if event.inaxes:
                    make_lineprofile(ax_list,im_list,mode='vertical',xdata=event.xdata,ydata=event.ydata,roi_list=None)

            elif event.key=='r': #set ROI
                if event.inaxes:
                    mouse_x, mouse_y = int(event.xdata + 0.5)-0.5, int(event.ydata + 0.5)-0.5
                    for ax_num,ax_cand in enumerate(ax_list):
                        roi_list[ax_num].set_xy((mouse_x,mouse_y))
            elif event.key=='alt+r': #reset ROI
                if event.inaxes:
                    for ax_num,ax_cand in enumerate(ax_list):
                        roi_list[ax_num].set_xy((-11.5, -11.5))
                        roi_list[ax_num].set_height(11)
                        roi_list[ax_num].set_width(11)
            elif event.key=='>':
                for ax_num,ax_cand in enumerate(ax_list):
                    roi_list[ax_num].set_width(roi_list[ax_num].get_width() + 2)
                print('roi size(x,y)='+str(roi_list[0].get_width())+', '+str(roi_list[0].get_height()))
            elif event.key=='alt+>':
                for ax_num,ax_cand in enumerate(ax_list):
                    roi_list[ax_num].set_width(np.clip(roi_list[ax_num].get_width() - 2,3,None))
                print('roi size(x,y)='+str(roi_list[0].get_width())+', '+str(roi_list[0].get_height()))
            elif event.key=='<':
                for ax_num,ax_cand in enumerate(ax_list):
                    roi_list[ax_num].set_height(roi_list[ax_num].get_height() + 2)
                print('roi size(x,y)='+str(roi_list[0].get_width())+', '+str(roi_list[0].get_height()))
            elif event.key=='alt+<':
                for ax_num,ax_cand in enumerate(ax_list):
                    roi_list[ax_num].set_height(np.clip(roi_list[ax_num].get_height() - 2,3,None))
                print('roi size(x,y)='+str(roi_list[0].get_width())+', '+str(roi_list[0].get_height()))

            elif event.key=='=':
                make_lineprofile(ax_list,im_list,mode='horizontal_mean',roi_list=roi_list)

            elif event.key=='I':
                make_lineprofile(ax_list,im_list,mode='vertical_mean',roi_list=roi_list)

            elif event.key=='$':
                roi_x_s,roi_y_s = roi_list[0].get_xy()
                roi_x_s,roi_y_s = np.clip(roi_x_s+0.5,0,None).astype(int),np.clip(roi_y_s+0.5,0,None).astype(int)
                roi_x_e,roi_y_e = (roi_x_s+roi_list[0].get_width()).astype(int),(roi_y_s+roi_list[0].get_height()).astype(int)
                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = im_list[ax_num].get_array()[roi_y_s:roi_y_e, roi_x_s:roi_x_e]
                    print( '--- img'+str(ax_num)+' x='+str(roi_x_s)+'~'+str(roi_x_e)+', y='+str(roi_y_s)+'~'+str(roi_y_e)+ ' ---\n'
                           +'mean     :'+str(np.nanmean(temp_img))   +'\n'
                           +'max      :'+str(np.nanmax(temp_img))    +'\n'
                           +'min      :'+str(np.nanmin(temp_img))    +'\n'
                           +'median   :'+str(np.nanmedian(temp_img)) +'\n'
                           +'std      :'+str(np.nanstd(temp_img))
                           )

            elif event.key=='m':
                roi_x_s,roi_y_s = roi_list[0].get_xy()
                roi_x_s,roi_y_s = np.clip(roi_x_s+0.5,0,None).astype(int),np.clip(roi_y_s+0.5,0,None).astype(int)
                roi_x_e,roi_y_e = (roi_x_s+roi_list[0].get_width()).astype(int),(roi_y_s+roi_list[0].get_height()).astype(int)
                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = im_list[ax_num].get_array()[roi_y_s:roi_y_e,roi_x_s:roi_x_e]
                    val_ax_list[ax_num].cla()
                    val_ax_list[ax_num].imshow(temp_img,aspect='auto')

                    ys, xs = np.meshgrid(range(temp_img.shape[0]), range(temp_img.shape[1]), indexing='ij')
                    for (xi, yi, val) in zip(xs.flatten(), ys.flatten(), temp_img.flatten()):
                        val_ax_list[ax_num].text(xi, yi, '{0:.3f}'.format(val),
                                                 horizontalalignment='center', verticalalignment='center', color='brown',fontweight='bold')

                val_fig.suptitle('x='+str(roi_x_s)+'~'+str(roi_x_e)+', y='+str(roi_y_s)+'~'+str(roi_y_e)
                                 +'\n(Values are rounded to three decimal places.)')
                val_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                val_fig.canvas.draw()
                val_fig.show()

            elif event.key=='h':
                roi_x_s,roi_y_s = roi_list[0].get_xy()
                roi_x_s,roi_y_s = np.clip(roi_x_s+0.5,0,None).astype(int),np.clip(roi_y_s+0.5,0,None).astype(int)
                roi_x_e,roi_y_e = (roi_x_s+roi_list[0].get_width()).astype(int),(roi_y_s+roi_list[0].get_height()).astype(int)

                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = im_list[ax_num].get_array()
                    roi_y_s2=np.clip(roi_y_s,0,np.shape(temp_img)[0])
                    roi_y_e2=np.clip(roi_y_e,0,np.shape(temp_img)[0])
                    roi_x_s2=np.clip(roi_x_s,0,np.shape(temp_img)[1])
                    roi_x_e2=np.clip(roi_x_e,0,np.shape(temp_img)[1])
                    val_ax_list[ax_num].cla()
                    val_ax_list[ax_num].axis('on')
                    val_ax_list[ax_num].hist(temp_img[roi_y_s2:roi_y_e2,roi_x_s2:roi_x_e2].flatten()
                                             ,bins=512,range=(aip.all_img_min,aip.all_img_max))

                val_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                val_fig.canvas.draw()
                val_fig.show()

            elif event.key=='H':
                img_xlim = ax_list[0].get_xlim()
                img_ylim = ax_list[0].get_ylim()
                img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
                img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = im_list[ax_num].get_array()
                    img_xlim2=[np.clip(img_xlim[0],0,np.shape(temp_img)[1]),np.clip(img_xlim[1],0,np.shape(temp_img)[1])]
                    img_ylim2=[np.clip(img_ylim[0],0,np.shape(temp_img)[0]),np.clip(img_ylim[1],0,np.shape(temp_img)[0])]
                    val_ax_list[ax_num].cla()
                    val_ax_list[ax_num].axis('on')
                    val_ax_list[ax_num].hist(temp_img[img_ylim2[1]:img_ylim2[0],img_xlim2[0]:img_xlim2[1]].flatten(),
                                             bins=512,range=(aip.all_img_min,aip.all_img_max))

                val_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                val_fig.canvas.draw()
                val_fig.show()

            ################################## image processing ##################################
            elif (event.key=='ctrl+1') or (event.key=='ctrl+2') or (event.key=='ctrl+3') or \
                    (event.key=='ctrl+4') or (event.key=='ctrl+5') or (event.key=='ctrl+6') or \
                    (event.key=='ctrl+7') or (event.key=='ctrl+8') or (event.key=='ctrl+9'):

                try:
                    eval_str = ctrl_func_dict[event.key]
                    new_target_img_list = []
                    for target_img_y_list in target_img_list:
                        new_target_img_list.append([])
                        for target_img in target_img_y_list:
                            new_target_img_list[-1].append( eval(eval_str) )
                    imageq(target_img_list=new_target_img_list)
                except:
                    print('imageq-Warning: The ctrl+(1-9) shortcut failed to execute the specified string.')


            ################################## file IO ##################################
            elif event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png')

            elif event.key=='ctrl+v':# paste clipboard image
                import win32clipboard # win32clipboard is included in the pywin32 package.
                win32clipboard.OpenClipboard()
                try:
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            caxis_min,caxis_max = im_list[ax_num].get_clim()
                            plt.gca().images[-1].colorbar.remove()
                            plt.gca().clear()
                            clipboard_img = imread(win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)[0])
                            aip.update_aip(clipboard_img,ax_num)
                            imshow_block(ax_list[-1], im_list, ax_num, clipboard_img, caxis_min, caxis_max)

                except:
                    pass
                win32clipboard.CloseClipboard()

            fig.canvas.draw()
            pass
        fig.canvas.mpl_connect('key_press_event',keyboard_shortcut)
        pass


    # layer切り替え関数
    def layer_numkey_switch(fig,ax_list,im):
        def layer_switch(event):
            if ((event.key).isdigit()) and (int(event.key)!=0) and (int(event.key))<len(im):
                # im[0].set_data(im[int(event.key)].get_array().copy())
                im[0].set_array(im[int(event.key)].get_array().copy())
                im[0].set_clim((im[int(event.key)].get_clim())[0], (im[int(event.key)].get_clim())[1])
                ax_list[0].figure.canvas.draw()
                # fig.canvas.draw()
                print("layer-"+event.key)
            pass
        fig.canvas.mpl_connect('key_press_event',layer_switch)
        pass

    # 一つのsubplotの描画セット
    def imshow_block(ax,im,im_index,target_img,caxis_min,caxis_max):

        if np.ndim(target_img) == 1:
            print("imageq-Warning: Drawing is not possible because a 1-channel image has been input.")
        elif np.ndim(target_img) == 2:
            pass
        elif np.ndim(target_img) == 3:
            print("imageq-Warning: Value range limited to 0-255 (uint8) due to 3-channel image input.")
            target_img = target_img.astype(np.uint8)
        else:
            print("imageq-Warning: Drawing is not possible because a 4-channel image has been input.")

        if im_index == -1:
            im.append(ax.imshow(target_img, interpolation='nearest', cmap=colormap, vmin=aip.all_img_min,vmax=aip.all_img_max))
        else:
            im[im_index] = ax.imshow(target_img, interpolation='nearest', cmap=colormap, vmin=aip.all_img_min,vmax=aip.all_img_max)
        im[im_index].set_clim(caxis_min,caxis_max)
        ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax.tick_params(bottom=False,left=False,right=False,top=False)

        if colorbar:# colorbar表示
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.075)
            ax.figure.add_axes(ax_cb)
            fig.colorbar(im[im_index], cax=ax_cb)
        if val_view and view_mode=='tile':# 画素値を重畳テキスト表示(小数2桁まで)
            ys, xs = np.meshgrid(range(target_img.shape[0]), range(target_img.shape[1]), indexing='ij')
            for (xi, yi, val) in zip(xs.flatten(), ys.flatten(), target_img.flatten()):
                ax.text(xi, yi, '{0:.2f}'.format(val), horizontalalignment='center', verticalalignment='center', color='deeppink')
        pass


    # それぞれのモードで描画
    ax_list = []
    im_list = []
    val_ax_list = []
    hst_ax_list = []
    axhline_list = []
    axvline_list = []
    roi_list = []
    if view_mode=='tile':
        for y in range(aip.sub_y_size):
            for x in range(len(target_img_list[y])):
                if (y+x)==0:
                    ax_list.append(fig.add_subplot(aip.sub_y_size,aip.sub_x_size,1,picker=True))
                    val_ax_list.append(val_fig.add_subplot(aip.sub_y_size,aip.sub_x_size,1,picker=True))
                else:
                    ax_list.append(fig.add_subplot(aip.sub_y_size,aip.sub_x_size, aip.sub_x_size*y+x+1 ,sharex=ax_list[0],sharey=ax_list[0]))
                    val_ax_list.append(val_fig.add_subplot(aip.sub_y_size,aip.sub_x_size, aip.sub_x_size*y+x+1 ,sharex=val_ax_list[0],sharey=val_ax_list[0]))

                imshow_block(ax_list[-1],im_list,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1])
                axhline_list.append(ax_list[-1].axhline(y=-0.5,color='pink'))
                axvline_list.append(ax_list[-1].axvline(x=-0.5,color='pink'))
                roi_list.append(patches.Rectangle(xy=(-11.5, -11.5), width=11, height=11, ec='red', fill=False))
                ax_list[-1].add_patch(roi_list[-1])

                # 最初の画像を着目画像に指定
                fig.sca(ax_list[0])
                ax_list[0].spines['bottom'].set_color("#2B8B96")
                ax_list[0].spines['top'].set_color("#2B8B96")
                ax_list[0].spines['left'].set_color("#2B8B96")
                ax_list[0].spines['right'].set_color("#2B8B96")
                ax_list[0].spines['bottom'].set_linewidth(4)
                ax_list[0].spines['top'].set_linewidth(4)
                ax_list[0].spines['left'].set_linewidth(4)
                ax_list[0].spines['right'].set_linewidth(4)

    elif view_mode=='layer':
        main_ax_colspan = 10
        ax_list.append(fig.subplot2grid((aip.all_img_num, main_ax_colspan+1), (0, 0), rowspan=aip.all_img_num, colspan=main_ax_colspan))
        imshow_block(ax_list[-1],im_list,-1,target_img_list[0][0],caxis[0][0][0],caxis[0][0][1])

        i = 0
        for y in range(len(target_img_list)):
            for x in range(len(target_img_list[y])):
                ax_list.append(fig.subplot2grid((aip.all_img_num, main_ax_colspan+1), (i, main_ax_colspan), rowspan=1, colspan=1))
                imshow_block(ax_list[-1],im_list,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1])
                i = i + 1
        # layer切り替え関数と接続
        layer_numkey_switch(fig,ax_list,im_list)

    # マウス、キーボード操作の関数連携
    mouse_click_event(fig,ax_list,im_list)
    keyboard_shortcut_sum(fig,ax_list,im_list)
    mouse_drag_move(fig,ax_list)
    if cross_cursor:
        mouse_cross_cursor(fig,axhline_list,axvline_list)

    keyboard_shortcut_onlyPNG(val_fig)


    # status barの表示変更
    def format_coord(x, y):
        int_x = int(x + 0.5)
        int_y = int(y + 0.5)
        return_str = 'x='+str(int_x)+', y='+str(int_y)+' |  '
        for k,now_im in enumerate(im_list):
            if 0 <= int_x < now_im.get_size()[1] and 0 <= int_y < now_im.get_size()[0]:
                now_img_val = now_im.get_array()[int_y,int_x]
                if np.ndim(now_img_val)==0:
                    return_str = return_str+str(k)+': '+'{:.3f}'.format(now_img_val)+'  '
                else:
                    return_str = return_str+str(k)+': <'+'{:.3f}'.format(now_img_val[0])+', '+'{:.3f}'.format(now_img_val[1])+', '+'{:.3f}'.format(now_img_val[2])+'>  '
            else:
                return_str = return_str+str(k)+': ###'+'  '
        # 対処には、https://stackoverflow.com/questions/47082466/matplotlib-imshow-formatting-from-cursor-position
        # のような実装が必要になり、別の関数＋matplotlibの関数を叩くが必要ありめんどくさい
        return return_str

    for now_ax in ax_list:
        now_ax.format_coord = format_coord


    # 表示範囲調整
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
    val_fig.tight_layout()
    main_figure_close(fig,val_fig)

    # 表示
    fig.show()
    pass
