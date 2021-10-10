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



def imageq(target_img_list, singlecoloraxis=True, coloraxis=(0,0), colormap='viridis', colorbar=True, val_view=False, view_mode='tile', cross_cursor=False):
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
    :param coloraxis:
    :param colormap:
    :param colorbar:
    :param val_view:
    :param view_mode:
    :param cross_cursor:
    :return:
    """
    mplstyle.use('fast')
    plt.interactive(False)


    print("""
    =======================================================================
    ------- ビューイング系操作 ------- 
    Drag                : 画像の移動
    shift+Drag          : 画像の部分拡大
    Double click        : 画像全体表示
    Click on the image  : クリックした画像を”着目画像”に指定
    D                   : ”着目画像”と他の画像の差分を表示
    A                   : ”着目画像”の現在描画されている領域でclimを自動スケーリング
    < (+alt)            : ”着目画像”のclim上限を1%小さく(大きく)
    > (+alt)            : ”着目画像”のclim下限を1%大きく(小さく)
    up(↑)/down(↓)       : ”着目画像”のclim範囲を1% 正/負 側にずらす
    S                   : ”着目画像”のclimを他の画像にも同期
    T                   : 全画像の最大/最小を用いて全画像のclimを設定
    ------------ 解析 ------------ 
    i / -               : 縦/横方向のラインプロファイルをを別ウィンドウで表示
    M                   : 11x11画素(マウス位置座標が左上)の画素値を別ウィンドウで表示
    ----- 画像書き出し、読み込み ----- 
    P                   : 現在のfigureをPNGで保存
    ctrl+v              : コピーした画像を着目画像領域に貼り付け
    """)

    # print("""
    # =======================================================================
    # Drag                : move the image.
    # shift+Drag          : zoom-in on the image.
    # Click on the image  : specify the clicked image as the "focus image".
    # D                   : Difference between the "focus image" and other images.
    # A                   : Auto-scaling the clim in the currently drawn area of the "focus image".
    # < (+alt)            : decrease (increase) the upper clim limit of the "focus image" by 1%.
    # > (+alt)            : increase (decrease) the lower clim limit of the "focus image" by 1%.
    # up(↑)/down(↓)       : shifts the clim range of the "focus image" by 1% to the positive/negative side.
    # S                   : Synchronize the "focus image" clim with other images.
    # T                   : set the clim of all images using the maximum/minimum of all images.
    # P                   : save the current figure as PNG.
    # """)

    val_fig = plt.figure(figsize=(13.0, 7.0))

    class all_img_property:
        img_num = 0
        img_min = np.Inf
        img_max = -np.Inf
        img_h_max = 0
        img_w_max = 0

        def update_minmax(self,target_img):
            target_img_min = np.nanmin(target_img)
            target_img_max = np.nanmax(target_img)
            target_img_hmax = np.shape(target_img)[0]
            target_img_wmax = np.shape(target_img)[1]
            self.img_min = target_img_min if target_img_min<self.img_min else self.img_min
            self.img_max = target_img_max if target_img_max>self.img_max else self.img_max
            self.img_h_max = target_img_hmax if target_img_hmax>self.img_h_max else self.img_h_max
            self.img_w_max = target_img_wmax if target_img_wmax>self.img_w_max else self.img_w_max
            pass

        pass

    # 必ず2次元listの各要素に画像が入ってる状態にする
    if isinstance(target_img_list,list)==False:
        target_img_list = [[target_img_list]]
    elif isinstance(target_img_list[0],list)==False:
        target_img_list = [target_img_list]

    if isinstance(coloraxis,list)==False:
        coloraxis = [[coloraxis]]
    elif isinstance(coloraxis[0],list)==False:
        coloraxis = [coloraxis]

    aip = all_img_property()
    for y in range(len(target_img_list)):
        for x in range(len(target_img_list[y])):
            aip.img_num = aip.img_num + 1
            aip.update_minmax(target_img_list[y][x])

    # subplot縦横サイズ計測 & 各画像に対する設定を保持する2次元list作成
    sub_y_size = len(target_img_list)
    sub_x_size = 0
    caxis = []
    if singlecoloraxis:
        if coloraxis[0][0][0]>=coloraxis[0][0][1]:
            temp = (aip.img_min,aip.img_max)
        else:
            temp = (coloraxis[0][0][0],coloraxis[0][0][1])
        for y in range(sub_y_size):
            sub_x_size = np.maximum(sub_x_size, len(target_img_list[y]))
            caxis.append([temp for x in range(len(target_img_list[y]))])
    else:
        for y in range(sub_y_size):
            sub_x_size = np.maximum(sub_x_size, len(target_img_list[y]))
            caxis.append([])
            for x in range(len(target_img_list[y])):
                if coloraxis[y][x][0]>=coloraxis[y][x][1]:
                    temp = (np.nanmin(np.nanmin(target_img_list[y][x])), np.nanmax(np.nanmax(target_img_list[y][x])))
                else:
                    temp = (coloraxis[y][x][0], coloraxis[y][x][1])
                caxis[-1].append(temp)


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
                ax_list[0].set_xlim(-0.5, aip.img_w_max-0.5)
                ax_list[0].set_ylim(aip.img_h_max-0.5, -0.5)


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
        global imageq_x1,imageq_y1,imageq_x1data,imageq_y1data,imageq_inaxes_flag,imageq_ctrl_flag,imageq_zoom_rect
        imageq_inaxes_flag=False
        imageq_ctrl_flag =False
        imageq_zoom_rect = patches.Rectangle(xy=(-1, -1),
                                             width=0,
                                             height=0,
                                             ec='#ff1493', fill=False)
        ax_list[0].add_patch(imageq_zoom_rect)

        def button_press(event):
            global imageq_x1,imageq_y1,imageq_x1data,imageq_y1data,imageq_inaxes_flag,imageq_ctrl_flag
            imageq_inaxes_flag=event.inaxes
            imageq_ctrl_flag =False
            if event.key=="shift":
                imageq_ctrl_flag=True
            imageq_x1 = event.x
            imageq_y1 = event.y
            imageq_x1data = event.xdata
            imageq_y1data = event.ydata
            pass

        def button_drag(event):
            global imageq_x1data,imageq_y1data,imageq_inaxes_flag,imageq_ctrl_flag,imageq_zoom_rect
            if imageq_ctrl_flag:
                if event.inaxes:
                    imageq_zoom_rect.set_bounds((imageq_x1data, imageq_y1data, event.xdata - imageq_x1data, event.ydata - imageq_y1data))
                else:
                    imageq_zoom_rect.set_bounds((imageq_x1data, imageq_y1data, 0,0))
                    pass
                fig.canvas.draw()
            pass

        def button_release(event):
            global imageq_x1,imageq_y1,imageq_x1data,imageq_y1data,imageq_inaxes_flag,imageq_ctrl_flag

            if imageq_inaxes_flag:
                ax_x_px = int( (ax_list[0].bbox.x1-ax_list[0].bbox.x0) )
                move_x = imageq_x1 - event.x
                lim_x = ax_list[0].get_xlim()
                ax_img_pix_x = lim_x[1]-lim_x[0]
                move_x_pix = move_x/ax_x_px*ax_img_pix_x

                ax_y_px = int( (ax_list[0].bbox.y1-ax_list[0].bbox.y0) )
                move_y = imageq_y1 - event.y
                lim_y = ax_list[0].get_ylim()
                ax_img_pix_y = lim_y[1]-lim_y[0]
                move_y_pix = move_y/ax_y_px*ax_img_pix_y

                if imageq_ctrl_flag:
                    x_lim = np.sort([imageq_x1data,imageq_x1data-move_x_pix])
                    y_lim = np.sort([imageq_y1data,imageq_y1data-move_y_pix])
                    ax_list[0].set_xlim(x_lim[0],x_lim[1])
                    ax_list[0].set_ylim(y_lim[1],y_lim[0])
                else:
                    ax_list[0].set_xlim(lim_x[0]+move_x_pix,lim_x[1]+move_x_pix)
                    ax_list[0].set_ylim(lim_y[0]+move_y_pix,lim_y[1]+move_y_pix)

                fig.canvas.draw()
                imageq_inaxes_flag = False
                imageq_ctrl_flag = False
                imageq_zoom_rect.set_bounds((-1, -1, 0, 0))
            pass

        fig.canvas.mpl_connect('button_press_event',button_press)
        fig.canvas.mpl_connect('motion_notify_event', button_drag)
        fig.canvas.mpl_connect('button_release_event',button_release)
        pass


    def keyboard_shortcut_onlyPNG(fig):
        def keyboard_onlyPNG(event):
            # print(event.key)
            if event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('imageq-LP-%Y_%m_%d_%H_%M_%S')+'.png')
            pass
        fig.canvas.mpl_connect('key_press_event',keyboard_onlyPNG)
        pass


    # keyboard_shortcut関数
    def keyboard_shortcut_sum(fig,ax_list,im):
        def keyboard_shortcut(event):
            # print(event.key)
            if event.key=='D':# diff
                s_state,s_ref,s_caxis = fig._suptitle.get_text().split('_')
                # state
                if s_state == 'normal':
                    print('ref-target')
                    gca_num = 0
                    caxis_text = ''
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            gca_num = ax_num
                    for num,now_im in enumerate(im):
                        caxis_text = caxis_text +str((now_im.get_clim())[0])+','+str((now_im.get_clim())[1])+'c'
                        if num!=gca_num:
                            set_img = im[gca_num].get_array().copy()-now_im.get_array().copy()
                            now_im.set_array(set_img)
                            now_im.set_clim(np.min(set_img),np.max(set_img))
                    fig._suptitle.set_text('diff_'+str(gca_num)+'_'+caxis_text)

                elif s_state == 'diff':
                    caxis_list = s_caxis.split('c')
                    for num,now_im in enumerate(im):
                        if int(s_ref)!=num:
                            now_im.set_array(im[int(s_ref)].get_array().copy()-now_im.get_array().copy())
                            caxis_min,caxis_max = caxis_list[num].split(',')
                            now_im.set_clim(float(caxis_min),float(caxis_max))
                    print(caxis_list)
                    fig._suptitle.set_text('normal_dummy_dummy')
                fig.canvas.draw()

            elif event.key=='A':# auto clim
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        int_x = np.clip((np.array(plt.gca().get_xlim()) + 0.5).astype(int),0,None)
                        int_y = np.clip((np.array(plt.gca().get_ylim()) + 0.5).astype(int),0,None)
                        temp = im[ax_num].get_array()[int_y[1]:int_y[0],int_x[0]:int_x[1]]
                        plt.clim(np.min(temp),np.max(temp))
                fig.canvas.draw()

            elif event.key=='S':# sync clim
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        for num,now_im in enumerate(im):
                            now_im.set_clim(float(caxis_min),float(caxis_max))
                        break
                fig.canvas.draw()

            elif event.key=='T':# set clim(<all image min>, <all image max>)

                for num,now_im in enumerate(im):
                    now_im.set_clim(aip.img_min,aip.img_max)
                fig.canvas.draw()

            elif event.key=='<':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min),float(caxis_max)-diff*0.01)
                fig.canvas.draw()
            elif event.key=='alt+<':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min),float(caxis_max)+diff*0.01)
                fig.canvas.draw()

            elif event.key=='>':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min)+diff*0.01,float(caxis_max))
                fig.canvas.draw()
            elif event.key=='alt+>':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min)-diff*0.01,float(caxis_max))
                fig.canvas.draw()

            elif event.key=='up':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min)+diff*0.01,float(caxis_max)+diff*0.01)
                fig.canvas.draw()
            elif event.key=='down':# set clim(<all image min>, <all image max>)
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        diff = caxis_max-caxis_min
                        im[ax_num].set_clim(float(caxis_min)-diff*0.01,float(caxis_max)-diff*0.01)
                fig.canvas.draw()

            elif event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png')

            elif event.key=='ctrl+v':# paste clipboard image
                import win32clipboard
                win32clipboard.OpenClipboard()
                try:
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            caxis_min,caxis_max = im[ax_num].get_clim()
                            plt.gca().images[-1].colorbar.remove()
                            plt.gca().clear()
                            clipboard_img = imread(win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)[0])
                            aip.update_minmax(clipboard_img)
                            imshow_block(ax[-1],im,ax_num,clipboard_img,caxis_min,caxis_max)
                            fig.canvas.draw()
                            print(aip.img_w_max,aip.img_h_max)
                except:
                    pass
                win32clipboard.CloseClipboard()

            elif event.key=='-':
                if event.inaxes:
                    mouse_x, mouse_y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                    img_xlim = ax[0].get_xlim()
                    img_ylim = ax[0].get_ylim()
                    img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
                    img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

                    ana_fig = plt.figure()
                    plt_sub_ax = ana_fig.add_subplot(2,len(ax_list),(1,len(ax_list)),picker=True)
                    img_sub_ax = []
                    img_sub_im = []
                    legend_str = []
                    for ax_num,ax_cand in enumerate(ax_list):
                        temp_img = im[ax_num].get_array()
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        plt_sub_ax.plot(np.arange(img_xlim[0],img_xlim[1]),temp_img[mouse_y,img_xlim[0]:img_xlim[1]])
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
                        img_sub_ax[-1].axhline(y=mouse_y,color='pink')
                        legend_str.append(str(ax_num))

                    plt_sub_ax.legend(legend_str)
                    ana_fig.suptitle('x='+str(img_xlim[0])+'~'+str(img_xlim[1])+', y='+str(mouse_y))
                    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                    ana_fig.show()
                    keyboard_shortcut_onlyPNG(ana_fig)

            elif event.key=='i':
                if event.inaxes:
                    mouse_x, mouse_y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                    img_xlim = ax[0].get_xlim()
                    img_ylim = ax[0].get_ylim()
                    img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
                    img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

                    ana_fig = plt.figure()
                    plt_sub_ax = ana_fig.add_subplot(2,len(ax_list),(1,len(ax_list)),picker=True)
                    img_sub_ax = []
                    img_sub_im = []
                    legend_str = []
                    for ax_num,ax_cand in enumerate(ax_list):
                        temp_img = im[ax_num].get_array()
                        caxis_min,caxis_max = im[ax_num].get_clim()
                        plt_sub_ax.plot(np.arange(img_ylim[0],img_ylim[1],-1),temp_img[img_ylim[0]:img_ylim[1]:-1,mouse_x])
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
                        img_sub_ax[-1].axvline(x=mouse_x,color='pink')
                        legend_str.append(str(ax_num))

                    plt_sub_ax.legend(legend_str)
                    ana_fig.suptitle('x='+str(mouse_x)+', y='+str(img_ylim[1])+'~'+str(img_ylim[0]))
                    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                    ana_fig.show()
                    keyboard_shortcut_onlyPNG(ana_fig)

            elif event.key=='M':
                if event.inaxes:
                    mouse_x, mouse_y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                    val_view_size = 11
                    mouse_x_s, mouse_y_s = np.clip(mouse_x,0,None), np.clip(mouse_y,0,None)
                    mouse_x_e, mouse_y_e = np.clip(mouse_x+val_view_size,0,None), np.clip(mouse_y+val_view_size,0,None)
                    print('x=',mouse_x_s,'~',mouse_x_e,', y=',mouse_y_s,'~',mouse_y_e)
                    for ax_num,ax_cand in enumerate(ax_list):
                        temp_img = im[ax_num].get_array()
                        val_ax[ax_num].cla()
                        val_ax[ax_num].axis('off')
                        temp_table=val_ax[ax_num].table(cellText=np.round(temp_img[mouse_y_s:mouse_y_e,mouse_x_s:mouse_x_e],3),
                                                        colLabels=np.arange(mouse_x_s,mouse_x_e),
                                                        rowLabels=np.arange(mouse_y_s,mouse_y_e),
                                                        loc="center")
                        temp_table.auto_set_font_size(False)
                        temp_table.set_fontsize(12)
                        temp_table.scale(1, 1)
                        cell_num = 0
                        for pos, cell in temp_table.get_celld().items():
                            cell.set_height(1/(val_view_size+2))
                            if cell_num>(val_view_size*val_view_size-1):
                                cell.set_facecolor('gray')
                            cell_num = cell_num+1

                    val_fig.suptitle('x='+str(mouse_x_s)+'~'+str(mouse_x_e)+', y='+str(mouse_y_s)+'~'+str(mouse_y_e)
                                     +'\n(Values are rounded to three decimal places.)')
                    val_fig.canvas.draw()
                    val_fig.show()
                    keyboard_shortcut_onlyPNG(val_fig)
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


    def main_figure_close(fig,val_fig):
        def close_event(event):
            val_fig.clf()
            plt.close(val_fig)
            pass
        fig.canvas.mpl_connect('close_event', close_event)
        pass


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
            im.append(plt.imshow(target_img, interpolation='nearest', cmap=colormap, vmin=aip.img_min,vmax=aip.img_max))
        else:
            im[im_index] = plt.imshow(target_img, interpolation='nearest', cmap=colormap, vmin=aip.img_min,vmax=aip.img_max)
        plt.clim(caxis_min,caxis_max)
        ax.tick_params(labelbottom=False,
                       labelleft=False,
                       labelright=False,
                       labeltop=False)
        ax.tick_params(bottom=False,
                       left=False,
                       right=False,
                       top=False)

        if colorbar:# colorbar表示
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.075)
            ax.figure.add_axes(ax_cb)
            plt.colorbar(im[im_index], cax=ax_cb)
        if val_view and view_mode=='tile':# 画素値を重畳テキスト表示(小数2桁まで)
            ys, xs = np.meshgrid(range(target_img.shape[0]), range(target_img.shape[1]), indexing='ij')
            for (xi, yi, val) in zip(xs.flatten(), ys.flatten(), target_img.flatten()):
                plt.text(xi, yi, '{0:.2f}'.format(val), horizontalalignment='center', verticalalignment='center', color='deeppink')
        pass


    # それぞれのモードで描画
    fig = plt.figure()
    fig.suptitle('normal_dummy_dummy',alpha=0)
    ax = []
    im = []
    val_ax = []
    axhline = []
    axvline = []
    if view_mode=='tile':
        for y in range(sub_y_size):
            for x in range(len(target_img_list[y])):
                if (y+x)==0:
                    ax.append(fig.add_subplot(sub_y_size,sub_x_size,1,picker=True))
                    val_ax.append(val_fig.add_subplot(sub_y_size,sub_x_size,1,picker=True))
                else:
                    ax.append(fig.add_subplot(sub_y_size,sub_x_size, sub_x_size*y+x+1 ,sharex=ax[0],sharey=ax[0]))
                    val_ax.append(val_fig.add_subplot(sub_y_size,sub_x_size, sub_x_size*y+x+1 ,picker=True))
                imshow_block(ax[-1],im,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1])
                axhline.append(ax[-1].axhline(y=-0.5,color='pink'))
                axvline.append(ax[-1].axvline(x=-0.5,color='pink'))

                # 最初の画像を着目画像に指定
                fig.sca(ax[0])
                ax[0].spines['bottom'].set_color("#2B8B96")
                ax[0].spines['top'].set_color("#2B8B96")
                ax[0].spines['left'].set_color("#2B8B96")
                ax[0].spines['right'].set_color("#2B8B96")
                ax[0].spines['bottom'].set_linewidth(4)
                ax[0].spines['top'].set_linewidth(4)
                ax[0].spines['left'].set_linewidth(4)
                ax[0].spines['right'].set_linewidth(4)

    elif view_mode=='layer':
        main_ax_colspan = 10
        ax.append(plt.subplot2grid((aip.img_num, main_ax_colspan+1), (0, 0), rowspan=aip.img_num, colspan=main_ax_colspan))
        imshow_block(ax[-1],im,-1,target_img_list[0][0],caxis[0][0][0],caxis[0][0][1])

        i = 0
        for y in range(len(target_img_list)):
            for x in range(len(target_img_list[y])):
                ax.append(plt.subplot2grid((aip.img_num, main_ax_colspan+1), (i, main_ax_colspan), rowspan=1, colspan=1))
                imshow_block(ax[-1],im,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1])
                i = i + 1
        # layer切り替え関数と接続
        layer_numkey_switch(fig,ax,im)

    # マウス、キーボード操作の関数連携
    mouse_click_event(fig,ax,im)
    keyboard_shortcut_sum(fig,ax,im)
    mouse_drag_move(fig,ax)
    if cross_cursor:
        mouse_cross_cursor(fig,axhline,axvline)

    # status barの表示変更
    def format_coord(x, y):
        int_x = int(x + 0.5)
        int_y = int(y + 0.5)
        return_str = 'x='+str(int_x)+', y='+str(int_y)+' |  '
        for k,now_im in enumerate(im):
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

    for now_ax in ax:
        now_ax.format_coord = format_coord


    # 表示範囲調整
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
    val_fig.tight_layout()
    main_figure_close(fig,val_fig)

    # 表示
    fig.show()
    pass
