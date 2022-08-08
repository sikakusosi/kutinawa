"""
表示系の関数群.
"""
import datetime
import numpy as np

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

from .kutinawa_num2num import rgb_to_hex
from .kutinawa_io import imread
from .kutinawa_filter import fast_boxfilter,fast_box_variance_filter


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


def clim(cmin,cmax):
    for im in plt.gca().get_images():
        im.set_clim(cmin,cmax)
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

def close_all():
    for i in plt.get_fignums():
        plt.gcf()
        plt.clf()
        plt.close()
    pass

def tolist_0dim(target_array):
    return [i for i in target_array]

def cmap_out_range_color(cmap_name='viridis',over_color='white',under_color='black',bad_color='red'):
    cm = pylab.cm.get_cmap(cmap_name)
    colors = cm.colors
    out_cmap = ListedColormap(colors,name='custom',N=255)
    out_cmap.set_over(over_color)
    out_cmap.set_under(under_color)
    out_cmap.set_bad(bad_color)
    return out_cmap

def q_basic(fig,init_xy_pos,yud_mode=0):

    local_ax_list = [i for i in fig.get_axes() if i.get_navigate()]
    # マウスクリック系イベント統合関数
    def mouse_click_event(fig):
        def click_event(event):
            # ダブルクリックで最も大きい画像に合わせて表示領域リセット
            if (event.dblclick) and (event.button==1):
                local_ax_list[0].set_xlim(init_xy_pos[0][0], init_xy_pos[0][1])
                local_ax_list[0].set_ylim(init_xy_pos[1][0], init_xy_pos[1][1])
            elif event.button==3:
                now_xlim = local_ax_list[0].get_xlim()
                now_ylim = local_ax_list[0].get_ylim()
                zoom_out_x = np.abs(now_xlim[0]-now_xlim[1])*0.05
                zoom_out_y = np.abs(now_ylim[0]-now_ylim[1])*0.05
                if yud_mode==0:
                    local_ax_list[0].set_xlim(now_xlim[0]-zoom_out_x, now_xlim[1]+zoom_out_x)
                    local_ax_list[0].set_ylim(now_ylim[0]-zoom_out_y, now_ylim[1]+zoom_out_y)
                elif yud_mode==1:
                    local_ax_list[0].set_xlim(now_xlim[0]-zoom_out_x, now_xlim[1]+zoom_out_x)
                    local_ax_list[0].set_ylim(now_ylim[0]+zoom_out_y, now_ylim[1]-zoom_out_y)

            # クリックした画像を着目画像(current axes)に指定
            for ax_cand in fig.get_axes():
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
    def mouse_drag_move(fig):
        mouse_data = {'x_s':0,
                      'y_s':0,
                      'xdata_s':0,
                      'ydata_s':0,
                      'inaxes_flag':False,
                      'shift_flag':False,
                      'zoom_rect':[]}

        for ax_num,ax_cand in enumerate(local_ax_list):
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
                    for ax_num,ax_cand in enumerate(local_ax_list):
                        mouse_data['zoom_rect'][ax_num].set_bounds((mouse_data['xdata_s'], mouse_data['ydata_s'],
                                                                    event.xdata-mouse_data['xdata_s'], event.ydata-mouse_data['ydata_s']))
                else:
                    for ax_num,ax_cand in enumerate(local_ax_list):
                        mouse_data['zoom_rect'][ax_num].set_bounds((-1,-1,0,0))
                    pass
                fig.canvas.draw()
            pass

        def button_release(event):
            if mouse_data['inaxes_flag']:
                ax_x_px = int( (local_ax_list[0].bbox.x1-local_ax_list[0].bbox.x0) )
                move_x = mouse_data['x_s'] - event.x
                lim_x = local_ax_list[0].get_xlim()
                ax_img_pix_x = lim_x[1]-lim_x[0]
                move_x_pix = move_x/ax_x_px*ax_img_pix_x

                ax_y_px = int( (local_ax_list[0].bbox.y1-local_ax_list[0].bbox.y0) )
                move_y = mouse_data['y_s'] - event.y
                lim_y = local_ax_list[0].get_ylim()
                ax_img_pix_y = lim_y[1]-lim_y[0]
                move_y_pix = move_y/ax_y_px*ax_img_pix_y

                if mouse_data['shift_flag']:
                    x_lim = np.sort([mouse_data['xdata_s'],mouse_data['xdata_s']-move_x_pix])
                    y_lim = np.sort([mouse_data['ydata_s'],mouse_data['ydata_s']-move_y_pix])
                    local_ax_list[0].set_xlim(x_lim[0],x_lim[1])
                    local_ax_list[0].set_ylim(y_lim[int(bool(0-yud_mode))],y_lim[int(bool(1-yud_mode))])

                else:
                    local_ax_list[0].set_xlim(lim_x[0]+move_x_pix,lim_x[1]+move_x_pix)
                    local_ax_list[0].set_ylim(lim_y[0]+move_y_pix,lim_y[1]+move_y_pix)
                    for ax_num,ax_cand in enumerate(local_ax_list):
                        mouse_data['zoom_rect'][ax_num].set_bounds((lim_x[0]+move_x_pix,lim_y[0]+move_y_pix,
                                                                    lim_x[1]-lim_x[0]  ,lim_y[1]-lim_y[0]   ))

                fig.canvas.draw()

            mouse_data['x_s']=0
            mouse_data['y_s']=0
            mouse_data['xdata_s']=0
            mouse_data['ydata_s']=0
            mouse_data['inaxes_flag']=False
            mouse_data['shift_flag']=False

            pass

        fig.canvas.mpl_connect('button_press_event',button_press)
        fig.canvas.mpl_connect('motion_notify_event', button_drag)
        fig.canvas.mpl_connect('button_release_event',button_release)
        pass

    # PNG保存
    def keyboard_shortcut_onlyPNG(fig):
        def keyboard_onlyPNG(event):
            if event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('kutinawa-%Y_%m_%d_%H_%M_%S')+'.png',bbox_inches='tight')
            pass
        fig.canvas.mpl_connect('key_press_event',keyboard_onlyPNG)
        pass

    mouse_click_event(fig)
    mouse_drag_move(fig)
    keyboard_shortcut_onlyPNG(fig)

    return fig

def q_input_shaping_flattening(input_list):
    if isinstance(input_list, list)==False:
        if (np.squeeze(input_list)).ndim>2:
            print("kutinawa-Warning:The input data is flattened and displayed because it had more than two dimensions.")
        input_list=[np.squeeze(np.reshape(input_list, (1, -1)))]
    else:
        temp_in = []
        for input_i in input_list:
            if (np.squeeze(input_i)).ndim>2:
                print("kutinawa-Warning:The input data is flattened and displayed because it had more than two dimensions.")
            temp_in.append(np.squeeze(np.reshape(input_i,(1,-1))))
        input_list = temp_in
    return input_list

def q_option_shaping(plotq_in, plotq_x_len):
    plot_option = []
    if isinstance(plotq_in,list)==False:
        for i in np.arange(plotq_x_len):
            plot_option.append(plotq_in)
    else:
        if plotq_x_len==len(plotq_in):
            plot_option = plotq_in
        else:
            print("Warning: Incorrect plot option designation. Therefore the default option is used.")
    return plot_option


def q_color_shaping(color, plotq_x_len):

    matplotlib_colormap_list = [#Perceptually Uniform Sequential
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        #Sequential
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        #Sequential (2)
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
        'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
        'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
        #Diverging
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
        #Cyclic
        'twilight', 'twilight_shifted', 'hsv',
        #Qualitative
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
        #Miscellaneous
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
        'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
        'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
        'turbo', 'nipy_spectral', 'gist_ncar']

    kutinawa_color_list = ['#f50035','#06b137','#00aff5','#ffca09','#00e0b4','#3a00cc','#e000a1','#9eff5d','#8d00b8','#7e5936',
                           '#367d4c','#364c7d','#f27993','#85de9f','#7acef0','#fce281','#97ded0','#937dc9','#9496ff','#8c8c8c',]

    plot_color = []
    if (isinstance(color,list)==False) and (type(color)==str):
        if color=='kutinawa_color':
            for i in np.arange(plotq_x_len):
                plot_color.append(kutinawa_color_list[i%len(kutinawa_color_list)])
        elif color in matplotlib_colormap_list:
            cm = plt.cm.get_cmap(color)
            for i in np.arange(plotq_x_len):
                plot_color.append(cm(i/np.clip(plotq_x_len-1,1,None)))
        else:
            for i in np.arange(plotq_x_len):
                plot_color.append(color)
    else:
        if plotq_x_len==len(color):
            plot_color = color
        else:
            print("Warning: Incorrect color designation. Therefore the default kutinawa_color is used.")

    return plot_color


def plotq(x_list, y_list,
          color='kutinawa_color',
          marker='None',
          markersize=7,
          linestyle='-',
          linewidth=2,
          label='None'):
    """
    折れ線グラフを一行で表示し、マウス操作・スクリーンショット保存等のショートカットキー機能を提供する関数。

    :param x_list     :
    :param y_list     : 入力データ。
                        1次元list内に、1つ以上のndarray形式のデータを格納して渡す。
                        略記法として、 1つのndarray形式のデータをそのまま渡す 事が可能。
    :param color      : 【※list指定可能】線、マーカーの色を指定する。
                        matplotlibで指定可能なカラーマップ文字列を 単独 もしくは データ個数分含むlist
                        （参考：https://matplotlib.org/stable/tutorials/colors/colormaps.html）
    :param marker     : 【※list指定可能】マーカー種類を指定する。
                        matplotlibで指定可能なマーカー文字列を 単独 もしくは データ個数分含むlist
                        （参考：https://matplotlib.org/stable/api/markers_api.html）
    :param markersize : 【※list指定可能】マーカーサイズを指定する。
                        マーカーサイズを指定する数値を 単独 もしくは データ個数分含むlist
    :param linestyle  : 【※list指定可能】線のスタイルを指定する。'-', '--', '.', '-.'のいずれか。
                        matplotlibで指定可能なlinestyle文字列を 単独 もしくは データ個数分含むlist
                        （参考：https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html）
    :param linewidth  : 【※list指定可能】線幅を指定する。
                        線幅を指定する数値を 単独 もしくは データ個数分含むlist
    :param label      : 【※list指定可能】データのラベル文字列を指定する。

    ※list指定可能
    　・単一指定の場合
    　　　すべてのデータに同じ引数が適用される。
    　・list指定の場合
    　　　list格納順に指定した引数が適用される。

    :return: なし
    """

    #################################################################################################################### データ整形
    # 必ずx,yデータがlistに入るように
    x_list = q_input_shaping_flattening(x_list)
    y_list = q_input_shaping_flattening(y_list)

    # x,yの要素数が違う場合エラー
    if len(x_list)!=len(y_list):
        print("plotq-Warning: Plotting is not possible because the x-y data do not have a one-to-one correspondence.")
        return

    ####################################################################################################################  plot時の線の色、太さ、マーカー等を整形
    plot_color = q_color_shaping(color, len(x_list))
    plot_marker = q_option_shaping(marker, len(x_list))
    plot_markersize = q_option_shaping(markersize, len(x_list))
    plot_linestyle = q_option_shaping(linestyle, len(x_list))
    plot_linewidth = q_option_shaping(linewidth, len(x_list))
    plot_label = np.arange(len(x_list)) if (isinstance(label, list) == False) or (len(label) != len(x_list)) else label

    #################################################################################################################### 描画
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,picker=True)
    plot_list = []
    for i in np.arange(len(x_list)):
        plot_list.append(ax.plot(x_list[i], y_list[i],
                                 color=plot_color[i],
                                 marker=plot_marker[i],
                                 markersize=plot_markersize[i],
                                 linestyle=plot_linestyle[i],
                                 linewidth=plot_linewidth[i],
                                 label=plot_label[i]
                                 )
                         )
    plot_legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})

    plot_map = {}
    for plot_legend_text, tgt_plot in zip(plot_legend.get_texts(),plot_list):
        plot_legend_text.set_picker(True)
        plot_map[plot_legend_text] = tgt_plot[0]

    def legend_switch(fig):
        def on_pick(event):
            plot_legend_line = event.artist
            if plot_legend_line in plot_map.keys():
                tgt_plot = plot_map[plot_legend_line]
                visible = not tgt_plot.get_visible()
                tgt_plot.set_visible(visible)
                plot_legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
        pass

    # q_basic機能付加
    fig = q_basic(fig=fig,init_xy_pos=[ax.get_xlim(),ax.get_ylim()] )
    legend_switch(fig)
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
    fig.show()

    pass


def histq(input_list,
          interval=1.0,
          color='kutinawa_color',
          edgecolor='k',
          alpha=0.75,
          label=None,
          density=False,
          cumulative=False,
          histtype='step',
          orientation='vertical',
          log=False,
          mode='hold',
          ):
    """
    ヒストグラムを一行で表示し、マウス操作・スクリーンショット保存等のショートカットキー機能を提供する関数。

    :param input_list : 入力データ。
                        1次元list内に、1つ以上の1次元ndarray形式のデータを格納して渡す。
                        略記法として、 1つのndarray形式のデータをそのまま渡す 事が可能。
    :param interval   : 【全データ共通】各binの値幅。
                        例 ) interval=0.5、全データの最小値～最大値=3.1~7.9 の場合
                        　　　3.1, 3.6, 4.1, 4.6, 5.1, 5.6, 6.1, 6.6, 7.1, 7.6, 8.1間をbinとする

    :param color      : 【※list指定可能】塗りつぶし部の色を指定する。
    :param edgecolor  : 【※list指定可能】線部の色を指定する。
                         color edgecolorは、下記を受け付ける。
                         ・matplotlibで指定可能なカラーマップ文字列 単独
                         　この場合、指定されたカラーマップに従いデータ毎に色が指定される。
                         　（参考：https://matplotlib.org/stable/tutorials/colors/colormaps.html）
                         ・matplotlibで指定可能な色を示す文字列('red'、'r'等) もしくは matplotlibで指定可能な色を示す16進数を データ個数分含むlist
                         　この場合、list格納順に指定色が適用される。
    :param alpha      : 【※list指定可能】透明度を指定する。
                        透明度を指定する数値を 単独 もしくは データ個数分含むlist
    :param label      : 【※list指定可能】データのラベルを指定する。
    :param density    : 【全データ共通】Trueの場合、確率密度として表示する。
    :param cumulative : 【全データ共通】Trueの場合、累積ヒストグラムを表示する。
    :param histtype   : 【全データ共通】表示形式を指定する。
                        ‘bar’ (通常のヒストグラム), ‘barstacked’ (積み上げヒストグラム),‘step’ (線), ‘stepfilled ‘ (塗りつぶしありの線) から選択。
    :param orientation: 【全データ共通】ヒストグラムの方向を指定する。　’horizontal’ (水平方向), ‘vertical’ (垂直方向) から選択。
    :param log        : 【全データ共通】Trueの場合、縦軸を対数目盛で表示する。

    ※list指定可能
    　・単一指定の場合
    　　　すべてのデータに同じ引数が適用される。
    　・list指定の場合
    　　　list格納順に指定した引数が適用される。

    :return:　なし
    """

    input_list = q_input_shaping_flattening(input_list)

    temp_min = np.Inf
    temp_max = -np.Inf
    for input_i in input_list:
        temp_min = np.minimum(np.nanmin(input_i),temp_min)
        temp_max = np.maximum(np.nanmax(input_i),temp_max)
    hist_bins = np.arange(temp_min, temp_max + interval*2, interval)

    ####################################################################################################################
    plot_color = q_color_shaping(color, len(input_list))
    plot_edgecolor = q_color_shaping(edgecolor, len(input_list))
    plot_alpha = q_option_shaping(alpha, len(input_list))
    plot_label = np.arange(len(input_list)) if (isinstance(label, list) == False) or (len(label) != len(input_list)) else label

    ####################################################################################################################
    fig = plt.figure()
    ax_list = []
    view_dict = {'hold':0,'sub':1}
    if mode=='hold':
        ax_list = [fig.add_subplot(1,1,1,picker=True)]
        hist_list = []
        for i in np.arange(len(input_list)):
            hist_list.append(ax_list[0].hist(input_list[i],
                                             bins=hist_bins,
                                             color=plot_color[i],
                                             alpha=plot_alpha[i],
                                             label=plot_label[i],
                                             edgecolor=plot_edgecolor[i],
                                             density=density,
                                             cumulative=cumulative,
                                             histtype=histtype,
                                             orientation=orientation,
                                             log=log,
                                             )
                             )
        hist_legend=ax_list[0].legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})

        if (histtype=='step') or (histtype=='stepfilled'):
            hist_map = {}
            for plot_legend_text, tgt_plot in zip(hist_legend.get_texts(),hist_list):
                plot_legend_text.set_picker(True)
                hist_map[plot_legend_text] = tgt_plot[2][0]

            def legend_switch(fig):
                def on_pick(event):
                    plot_legend_line = event.artist
                    if plot_legend_line in hist_map.keys():
                        tgt_plot = hist_map[plot_legend_line]
                        visible = not tgt_plot.get_visible()
                        tgt_plot.set_visible(visible)
                        plot_legend_line.set_alpha(1.0 if visible else 0.2)
                        fig.canvas.draw()
                fig.canvas.mpl_connect('pick_event', on_pick)
                pass

            legend_switch(fig)

    elif mode=='sub':
        sub_x=sub_y=0
        if orientation=='vertical':
            sub_x = 1
            sub_y = len(input_list)
        elif orientation=='horizontal':
            sub_x = len(input_list)
            sub_y = 1

        hist_list = []
        for i in np.arange(len(input_list)):
            if i==0:
                ax_list.append(fig.add_subplot(sub_y,sub_x,i+1,picker=True))
            else:
                ax_list.append(fig.add_subplot(sub_y,sub_x,i+1,picker=True,sharex=ax_list[0],sharey=ax_list[0]))

            hist_list.append(ax_list[i*view_dict[mode]].hist(input_list[i],
                                                             bins=hist_bins,
                                                             color=plot_color[i],
                                                             alpha=plot_alpha[i],
                                                             label=plot_label[i],
                                                             edgecolor=plot_edgecolor[i],
                                                             density=density,
                                                             cumulative=cumulative,
                                                             histtype=histtype,
                                                             orientation=orientation,
                                                             log=log,
                                                             )
                             )

    # q_basic機能付加
    fig = q_basic(fig=fig,init_xy_pos=[ax_list[0].get_xlim(),ax_list[0].get_ylim()] )
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
    fig.show()

    pass


def imageq(target_img_list,
           coloraxis   = (0,0),
           colormap    = 'viridis',
           colorbar    = True,
           val_view    = False,
           view_mode   = 'tile',
           help_print  = True,
           **kwargs
           ):
    """
    ショートカットキーで色々できる、画像ビューワー。

    :param target_img_list: 表示したい画像。
                            2次元list内に、1つ以上のndarray形式の画像を格納して渡す。
                            view_mode=tileの場合、2次元listの0次元目が縦、1次元目が横に対応して表示される。
                            例：target_img_list=[[img_a, img_b],
                            　　                 [img_c, img_d]]
                                表示は下記のようになる。
                                ┌── figure ───────────┐
                                │ ┌───────┐ ┌───────┐ │
                                │ │ img_a │ │ img_b │ │
                                │ └───────┘ └───────┘ │
                                │ ┌───────┐ ┌───────┐ │
                                │ │ img_c │ │ img_d │ │
                                │ └───────┘ └───────┘ │
                                └─────────────────────┘
                            略記法として、1次元listに1つ以上のndarray形式画像を格納して渡す / 1つのndarray形式の画像をそのまま渡す 事が可能。
                            例えば、入力を [img_a, img_b] とした場合は、[[img_a, img_b]]と等価となる。

    :param coloraxis:       表示に使う疑似カラーの範囲。
                            (最小値,最大値)という構成のtuple　もしくは　target_img_listと同様の構成のリストに格納された(最小値,最大値)のtupleを受け付ける。
                            全画像に対して同じ疑似カラーの範囲を適用する場合は、単一のtupleを設定する。
                            各画像に対して個別の疑似カラーの範囲を適用する場合は、target_img_listと同様の構成のリストに格納された(最小値,最大値)のtupleを設定する。
                            またtupleの最小値・最大値が同一の値である場合、画像の最小値最大値から自動で疑似カラーが設定される。
                                       │                                        [[(min_a, max_a), (min_b, min_b)],
                                       │         (min, max)                      [(min_c, max_c), (min_d, min_d)]]
                            ───────────┼──────────────────────────────────────────────────────────────────────────────────
                            min != max │ 全画像に(min,max)が適用される          個別の画像に、設定された(min_ , max_)が適用される
                            ───────────┼──────────────────────────────────────────────────────────────────────────────────
                            min == max │ 全画像に全画像の                       個別の画像ごとに、個別の画像の最小・最大の
                                       │ 最小・最大の(min,max)が適用される       (min,max)が適用される

    :param colormap:        表示に用いる疑似カラーマップを指定する。
                            matplotlibで指定可能なカラーマップ文字列、もしくはLinearSegmentedColormap関数等を用いて作成したLUT形式のカラーマップを受け付ける。
                            wa.cmap_out_range_color関数を用いることも可能。
                            matplotlibで指定可能なカラーマップ文字列は右記URLを参照。https://matplotlib.org/stable/tutorials/colors/colormaps.html

    :param colorbar:        colorbarを表示するかどうかを指定する。boolを受け付ける。

    :param val_view:        画素値を画像上にオーバーレイ表示するかどうかを指定する。boolを受け付ける。

    :param view_mode:       表示方式を選択する。'tile'もしくは'layer'を受け付ける。
                            tileの場合は、縦横に画像を並べた形で表示する。
                            layerの場合は、メインの表示部にF1~F12のキーで選択された画像を表示する。右側には表示候補の画像が常に表示されている。

    :return:                なし。
    """

    if help_print:
        print("""
        ======= kutinawa imageq =======
        ------------ マウス操作 ------------------------------------------------------------------------------------------ 
         Drag                    : 画像の移動
         shift + Drag            : 選択領域を拡大表示
         Right click             : 現在の表示範囲から5%ズームアウト
         Double click            : 画像全体表示
         Left click on the image : クリックした画像を”着目画像”に指定
        ------------ clim 自動調整 ---------------------------------------------------------------------------------------
         shift + a               : ”着目画像”のみ現在描画されている領域でclimを自動スケーリング
         shift + e               : 各画像それぞれ現在描画されている領域でclimを自動スケーリング
         shift + w               : 全画像の最大-最小を用いて全画像のclimを設定
         shift + s               : ”着目画像”のclimを他の画像にも同期
        ------------ clim 手動調整 ---------------------------------------------------------------------------------------
         left(←),right(→) (+alt) : ”着目画像”のclim上限を1%小さく(←),下限を1%大きく(→)、
                                   (+alt)時はclim上限を1%大きく(<),下限を1%小さく(>)
         up(↑),down(↓)           : ”着目画像”のclim範囲を1% 正(up),負(down)側にずらす
         ※(←,→,↑,↓(+ctrl))       　(+ ctrl)時は←,→,↑,↓の操作の変動量が5%となる
        ------------ 画像比較 -------------------------------------------------------------------------------------------- 
         shift + d               : ”着目画像”と他の画像の差分を表示
         :                       : ”着目画像”と他の画像のSNR,PSNR,MSSIMをコンソールに表示、SSIMを別ウィンドウで表示
         F1 ~  F12               : 表示画像の切り替え(mode='layer'時のみ), F1=image0、F2=image1、…と対応している
        ------------ 画像全体に対する解析 ---------------------------------------------------------------------------------- 
         shift + h               : 表示範囲内(shift + h)の画素値ヒストグラム表示 
        ------------ lineを用いた解析 ------------------------------------------------------------------------------------ 
         i, -                    : キー押下時のマウス位置における、縦(i),横(-)方向のラインプロファイルをを別ウィンドウで表示
        ------------ ROIを用いた解析 ------------------------------------------------------------------------------------- 
         r (+ alt)               : キー押下時のマウス位置を左上としたROIを設定 
                                   (+ alt)時は、ROIサイズをデフォルトサイズ(11x11)に戻し画像外に移動
         >, < (+ alt)            : ROIの水平(>),垂直(<)サイズ拡大((+ alt)時は縮小)を行う
         m                       : ROI内の画素値を別ウィンドウで表示
         h                       : ROI内の画素値ヒストグラム表示 
         $                       : ROI内の画素値の統計量表示
         shift + i, =            : ROIの水平範囲を平均した縦(shift + i),
                                   垂直範囲を平均した横(=)方向のラインプロファイルを別ウィンドウで表示
        ------------ 画像書き出し、読み込み -------------------------------------------------------------------------------- 
         shift + p               : 現在のfigureをPNGで保存
         ctrl  + v               : コピーした画像を着目画像領域に貼り付け
        """)

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
                if coloraxis[y][x][0]>=coloraxis[y][x][1]:#(min,max)の指定が同数も育は逆転している → coloraxis=(全画像の最小,全画像の最大)
                    caxis[-1].append( (aip.img_minmax_list[i][0], aip.img_minmax_list[i][1]) )
                else:
                    caxis[-1].append( (coloraxis[y][x][0], coloraxis[y][x][1]) )
                i = i + 1

    cmap = []
    if isinstance(colormap,list)==False:#colormapが一つだけ → すべてのcolormapを同じで
        for y in range(aip.sub_y_size):
            cmap.append([colormap for x in range(len(target_img_list[y]))])

    else:# coloraxisがlist=2つ以上 → バラバラのcolormap
        if isinstance(colormap[0],list)==False:
            colormap = [colormap]
        print(cmap)
        for y in range(aip.sub_y_size):
            cmap.append([])
            for x in range(len(target_img_list[y])):
                cmap[-1].append( colormap[y][x])

    ############################### 操作系関数群
    # main figが閉じられた場合の動作
    def main_figure_close(fig,val_fig):
        def close_event(event):
            val_fig.clf()
            plt.close(val_fig)
            pass
        fig.canvas.mpl_connect('close_event', close_event)
        pass

    # PNG保存
    def keyboard_shortcut_onlyPNG(fig):
        def keyboard_onlyPNG(event):
            if event.key=='P':# save figure in PNG
                fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png',bbox_inches='tight')
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

        elif mode=='each':
            for ax_num,ax_cand in enumerate(ax_list):
                int_x = np.clip((np.array(ax_cand.get_xlim()) + 0.5).astype(int),0,None)
                int_y = np.clip((np.array(ax_cand.get_ylim()) + 0.5).astype(int),0,None)
                temp = im_list[ax_num].get_array()[int_y[1]:int_y[0],int_x[0]:int_x[1]]
                im_list[ax_num].set_clim(np.min(temp),np.max(temp))

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
            cmap = im_list[ax_num].get_cmap()
            img_sub_ax.append(ana_fig.add_subplot(2,len(ax_list),len(ax_list)+ax_num+1,picker=True))
            imshow_block(ax=img_sub_ax[-1],
                         im=img_sub_im,
                         im_index=-1,
                         target_img=temp_img,
                         caxis_min=caxis_min,
                         caxis_max=caxis_max,
                         cmap=cmap)
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

    def get_img_in_roi(target_img,roi_x_s,roi_y_s,roi_x_e,roi_y_e):
        roi_y_s2=np.clip(roi_y_s,0,np.shape(target_img)[0])
        roi_y_e2=np.clip(roi_y_e,0,np.shape(target_img)[0])
        roi_x_s2=np.clip(roi_x_s,0,np.shape(target_img)[1])
        roi_x_e2=np.clip(roi_x_e,0,np.shape(target_img)[1])


        if (roi_y_s2==roi_y_e2) or (roi_x_s2==roi_x_e2):
            out_img = np.array([[0]])
        else:
            if np.ndim(target_img)==2:
                out_img = np.zeros((roi_y_e-roi_y_s,roi_x_e-roi_x_s))*np.NaN
                out_img[roi_y_s2-roi_y_s:(roi_y_s2-roi_y_s)+(roi_y_e2-roi_y_s2),roi_x_s2-roi_x_s:(roi_x_s2-roi_x_s)+(roi_x_e2-roi_x_s2)] = target_img[roi_y_s2:roi_y_e2,roi_x_s2:roi_x_e2]
            elif np.ndim(target_img)==3:
                out_img = np.zeros((roi_y_e-roi_y_s,roi_x_e-roi_x_s,np.shape(target_img)[2]))*np.NaN
                out_img[roi_y_s2-roi_y_s:(roi_y_s2-roi_y_s)+(roi_y_e2-roi_y_s2),roi_x_s2-roi_x_s:(roi_x_s2-roi_x_s)+(roi_x_e2-roi_x_s2),:] = target_img[roi_y_s2:roi_y_e2,roi_x_s2:roi_x_e2,:]

        return out_img

    # keyboard_shortcut関数
    temp_state_refnum_clim = ["normal",0,]
    def keyboard_shortcut_sum(fig, ax_list, im_list):
        def keyboard_shortcut(event):
            ################################## image diffarence ##################################
            if event.key=='D':# diff
                if temp_state_refnum_clim[0] == 'normal':
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            temp_state_refnum_clim[1] = ax_num
                    for num,now_im in enumerate(im_list):
                        temp_state_refnum_clim.append((now_im.get_clim()))
                        if num!=temp_state_refnum_clim[1]:
                            if np.ndim(im_list[temp_state_refnum_clim[1]].get_array()) == np.ndim(now_im.get_array()):
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

            elif event.key==':':# image quality Evaluation
                ref_ax_num = 0
                for ax_num,ax_cand in enumerate(ax_list):
                    if plt.gca() == ax_cand:
                        ref_ax_num = ax_num

                ref_img = im_list[ref_ax_num].get_array()
                fil_h=15
                fil_w=15
                ref_img_range = np.nanmax(ref_img)-np.nanmin(ref_img)
                C1 = (ref_img_range*0.01)*(ref_img_range*0.01)
                C2 = (ref_img_range*0.03)*(ref_img_range*0.03)
                ref_local_mean = fast_boxfilter(ref_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w
                ref_local_var  = fast_box_variance_filter(ref_img,fil_h=fil_h,fil_w=fil_w)
                ssim_list = []
                coloraxis_list=[]
                for ax_num,ax_cand in enumerate(ax_list):
                    eva_img = im_list[ax_num].get_array()
                    if np.shape(ref_img)==np.shape(eva_img):
                        # MSE
                        diff_img = ref_img-eva_img
                        mse = np.nanmean(diff_img*diff_img)
                        if mse != 0:
                            # SNR
                            signal_range = np.nanmax(ref_img)-np.nanmin(ref_img)
                            snr = 10*np.log10(signal_range*signal_range/mse)
                            # PSNR
                            psnr = 10*np.log10(255*255/mse)
                            # SSIM
                            eva_local_mean = fast_boxfilter(eva_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w
                            eva_local_var  = fast_box_variance_filter(eva_img,fil_h=fil_h,fil_w=fil_w)
                            ref_eva_cov    = fast_boxfilter(ref_img*eva_img,fil_h=fil_h,fil_w=fil_w)/fil_h/fil_w - (ref_local_mean*eva_local_mean)
                            ssim_list.append(
                                ( (2*ref_local_mean*eva_local_mean+C1)*(2*ref_eva_cov+C2) )
                                /( (ref_local_mean*ref_local_mean+eva_local_mean*eva_local_mean+C1)*(ref_local_var+eva_local_var+C2) )
                            )
                            coloraxis_list.append((0,1))
                            mssim = np.nanmean(ssim_list[-1])
                        else:
                            ssim_list.append(eva_img)
                            coloraxis_list.append((np.nanmin(eva_img),np.nanmax(eva_img)))
                            snr,psnr,mssim = np.Inf,np.Inf,1

                        title_str = ('Ref-> img'+str(ref_ax_num)+'  :Evaluate-> img'+str(ax_num)+
                                     '\nSNR   : ' + '{0:.5f}'.format(snr)   + '(dB)' +
                                     '\nPSNR  : ' + '{0:.5f}'.format(psnr)  + '(dB)' +
                                     '\nMSSIM : ' + '{0:.5f}'.format(mssim)  +
                                     '\n')
                        print(title_str)

                imageq(ssim_list,coloraxis=coloraxis_list,help_print=False)


            ################################## update clim with AUTOMATIC adjustments ##################################
            elif event.key=='A':# auto clim
                update_clim(ax_list,im_list,mode='auto')
            elif event.key=='S':# sync clim
                update_clim(ax_list,im_list,mode='sync')
            elif event.key=='W':# set clim(<all image min>, <all image max>)
                update_clim(ax_list,im_list,mode='whole')
            elif event.key=='E':# set clim(<all image min>, <all image max>)
                update_clim(ax_list,im_list,mode='each')


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
            elif (event.key=='ctrl+alt+left') or (event.key=='alt+ctrl+left'):
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[0,+1])
            elif event.key=='ctrl+right':
                update_clim(ax_list,im_list,mode='manual',gain=0.05,plusminus=[+1,0])
            elif (event.key=='ctrl+alt+right') or (event.key=='alt+ctrl+right'):
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
                           +'mean     : '+str(np.nanmean(temp_img))   +'\n'
                           +'max      : '+str(np.nanmax(temp_img))    +'\n'
                           +'min      : '+str(np.nanmin(temp_img))    +'\n'
                           +'median   : '+str(np.nanmedian(temp_img)) +'\n'
                           +'std      : '+str(np.nanstd(temp_img))
                           )

            elif event.key=='m':
                roi_x_s,roi_y_s = roi_list[0].get_xy()
                roi_x_s,roi_y_s = int(roi_x_s+0.5),int(roi_y_s+0.5)
                roi_x_e,roi_y_e = int(roi_x_s+roi_list[0].get_width()),int(roi_y_s+roi_list[0].get_height())
                xticklabel = [str(i) for i in np.arange(roi_x_s,roi_x_e)]
                yticklabel = [str(i) for i in np.arange(roi_y_s,roi_y_e)]

                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = get_img_in_roi(im_list[ax_num].get_array(),roi_x_s,roi_y_s,roi_x_e,roi_y_e)

                    val_ax_list[ax_num].cla()
                    val_ax_list[ax_num].imshow(temp_img,aspect='auto',cmap=colormap)
                    val_ax_list[ax_num].set_xticks(np.arange(0,roi_x_e-roi_x_s))
                    val_ax_list[ax_num].set_xticklabels(xticklabel)
                    val_ax_list[ax_num].set_yticks(np.arange(0,roi_y_e-roi_y_s))
                    val_ax_list[ax_num].set_yticklabels(yticklabel)
                    ys, xs = np.meshgrid(range(temp_img.shape[0]), range(temp_img.shape[1]), indexing='ij')

                    if np.ndim(temp_img)==2:
                        for (xi, yi, val) in zip(xs.flatten(), ys.flatten(), temp_img.flatten()):
                            val_ax_list[ax_num].text(xi, yi, '{0:.3f}'.format(val),horizontalalignment='center', verticalalignment='center', color='#ff21cb',fontweight='bold')
                    elif np.ndim(temp_img)==3:
                        for (xi, yi) in zip(xs.flatten(), ys.flatten()):
                            val_ax_list[ax_num].text(xi, yi-0.25, '{0:.3f}'.format(temp_img[yi,xi,0]),horizontalalignment='center', verticalalignment='center', color='red',fontweight='bold')
                            val_ax_list[ax_num].text(xi, yi, '{0:.3f}'.format(temp_img[yi,xi,1]),horizontalalignment='center', verticalalignment='center', color='green',fontweight='bold')
                            val_ax_list[ax_num].text(xi, yi+0.25, '{0:.3f}'.format(temp_img[yi,xi,2]),horizontalalignment='center', verticalalignment='center', color='blue',fontweight='bold')


                val_fig.suptitle('x='+str(roi_x_s)+'~'+str(roi_x_e)+', y='+str(roi_y_s)+'~'+str(roi_y_e)
                                 +'\n(Values are rounded to three decimal places.)')
                val_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                val_fig.canvas.draw()
                val_fig.show()

            elif event.key=='h':
                roi_x_s,roi_y_s = roi_list[0].get_xy()
                roi_x_s,roi_y_s = int(roi_x_s+0.5),int(roi_y_s+0.5)
                roi_x_e,roi_y_e = int(roi_x_s+roi_list[0].get_width()),int(roi_y_s+roi_list[0].get_height())

                for ax_num,ax_cand in enumerate(ax_list):
                    temp_img = get_img_in_roi(im_list[ax_num].get_array(),roi_x_s,roi_y_s,roi_x_e,roi_y_e)

                    val_ax_list[ax_num].cla()
                    val_ax_list[ax_num].axis('on')
                    val_ax_list[ax_num].hist(temp_img.flatten(),bins=512,range=(aip.all_img_min,aip.all_img_max))
                    val_ax_list[ax_num].set_aspect('auto')

                val_fig.suptitle('x='+str(roi_x_s)+'~'+str(roi_x_e)+', y='+str(roi_y_s)+'~'+str(roi_y_e)
                                 +'\nHistogram')
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

                val_fig.suptitle('x='+str(img_xlim[0])+'~'+str(img_xlim[1])+', y='+str(img_ylim[1])+'~'+str(img_ylim[0])
                                 +'\nHistogram')
                val_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)
                val_fig.canvas.draw()
                val_fig.show()

            ################################## file IO ##################################
            # elif event.key=='P':# save figure in PNG
            #     fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png',bbox_inches='tight',)

            elif event.key=='ctrl+v':# paste clipboard image
                import win32clipboard # win32clipboard is included in the pywin32 package.
                win32clipboard.OpenClipboard()
                try:
                    for ax_num,ax_cand in enumerate(ax_list):
                        if plt.gca() == ax_cand:
                            caxis_min,caxis_max = im_list[ax_num].get_clim()
                            cmap = im_list[ax_num].get_cmap()
                            plt.gca().images[-1].colorbar.remove()
                            plt.gca().clear()
                            clipboard_img = imread(win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)[0])
                            aip.update_aip(clipboard_img,ax_num)
                            imshow_block(ax_list[-1], im_list, ax_num, clipboard_img, caxis_min, caxis_max,cmap=cmap)

                except:
                    pass
                win32clipboard.CloseClipboard()

            ################################## layer ##################################
            elif (event.key=='f1' or event.key=='f2' or event.key=='f3' or event.key=='f4' or event.key=='f5' or event.key=='f6' or
                  event.key=='f7' or event.key=='f8' or event.key=='f9' or event.key=='f10' or event.key=='f11' or event.key=='f12') and (view_mode=='layer'):# layer

                layer_num = int(event.key[1:])
                if layer_num<len(im_list):
                    l0_img = im_list[layer_num].get_array().copy()
                    im_list[0].set_array(l0_img)
                    im_list[0].set_clim((im_list[layer_num].get_clim())[0], (im_list[layer_num].get_clim())[1])
                    print("layer-"+str(layer_num))

            ###########################################################################
            fig.canvas.draw()
            pass
        fig.canvas.mpl_connect('key_press_event',keyboard_shortcut)
        pass

    # 一つのsubplotの描画セット
    def imshow_block(ax,im,im_index,target_img,caxis_min,caxis_max,cmap):

        flag_1dim_img = True
        if np.ndim(target_img) == 1:
            print("imageq-Warning: Drawing is not possible because a 1-dimensional data has been input.")
        elif np.ndim(target_img) == 2:
            pass
        elif np.ndim(target_img) == 3:
            # flag_1dim_img = False
            if target_img.dtype in ['int16','int32','int64','uint16','uint32','uint64','float16','float32','float64','bool']:
                print("imageq-Warning: Value range normalized to 0-1(float64) due to 3-dim image input.")
                target_img = target_img.astype('float64')
                min_val = np.nanmin(target_img)
                max_val = np.nanmax(target_img)
                target_img = (target_img-min_val)/(max_val-min_val)
        else:
            print("imageq-Warning: Drawing is not possible because a 4-dimensional data has been input.")

        if im_index == -1:
            im.append(ax.imshow(target_img, interpolation='nearest', cmap=cmap, vmin=aip.all_img_min,vmax=aip.all_img_max))
        else:
            im[im_index] = ax.imshow(target_img, interpolation='nearest', cmap=cmap, vmin=aip.all_img_min,vmax=aip.all_img_max)
        im[im_index].set_clim(caxis_min,caxis_max)
        ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        ax.tick_params(bottom=False,left=False,right=False,top=False)

        if colorbar and flag_1dim_img:# colorbar表示
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

                imshow_block(ax_list[-1],im_list,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1],cmap[y][x])
                axhline_list.append(ax_list[-1].axhline(y=-0.5,color='pink'))
                axvline_list.append(ax_list[-1].axvline(x=-0.5,color='pink'))
                roi_list.append(patches.Rectangle(xy=(-11.5, -11.5), width=11, height=11, ec='red', fill=False))
                ax_list[-1].add_patch(roi_list[-1])

                # 最初の画像を着目画像に指定
                fig.sca(ax_list[0])

    elif view_mode=='layer':
        spec = gridspec.GridSpec(nrows=aip.all_img_num, ncols=2, width_ratios=[10,1])
        ax_list.append(fig.add_subplot(spec[:,0],picker=True))
        val_ax_list.append(val_fig.add_subplot(spec[:,0],picker=True))
        imshow_block(ax_list[-1],im_list,-1,target_img_list[0][0],caxis[0][0][0],caxis[0][0][1],cmap[0][0])

        i = 0
        for y in range(len(target_img_list)):
            for x in range(len(target_img_list[y])):
                ax_list.append(fig.add_subplot(spec[i,1]))
                val_ax_list.append(val_fig.add_subplot(spec[i,1]))
                imshow_block(ax_list[-1],im_list,-1,target_img_list[y][x],caxis[y][x][0],caxis[y][x][1],cmap[y][x])
                i = i + 1

    # マウス、キーボード操作の関数連携
    keyboard_shortcut_sum(fig,ax_list,im_list)
    keyboard_shortcut_onlyPNG(val_fig)
    fig = q_basic(fig=fig,init_xy_pos=[[-0.5, aip.all_img_w_max-0.5],[aip.all_img_h_max-0.5, -0.5]],yud_mode=1)


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

    if 'save_png' in kwargs:
        fig.set_size_inches(kwargs['save_png'][1],kwargs['save_png'][0])
        fig.savefig(kwargs['save_png'][2]+'.png',bbox_inches='tight', )

    mplstyle.use('fast')

    pass
