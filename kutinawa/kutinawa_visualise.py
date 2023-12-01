"""
表示系の関数群.
"""
import itertools

import datetime
import numpy as np

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pylab


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



########################################################################################################################
#   ██████   █████  ███████ ███████
#   ██   ██ ██   ██ ██      ██
#   ██████  ███████ ███████ █████
#   ██   ██ ██   ██      ██ ██
#   ██████  ██   ██ ███████ ███████
########################################################################################################################
def q_basic(config_list,init_xy_pos,yud_mode=0,keyboard_dict=None):
    def mouse_key_event(fig):
        event_dict = {'btn_prs_x':0,'btn_prs_y':0,
                      'btn_prs_xdata':0,'btn_prs_ydata':0,
                      'key_prs_xdata':0,'key_prs_ydata':0,
                      'btn_prs_inaxes_flag':False,
                      'btn_prs_shift_flag':False,
                      'wheel_shift_flag':False,
                      'wheel_xlim':(0,0),
                      'wheel_ylim':(0,0),
                      }
        zoom_dict = {'up':0.95,'down':1.05}

        def button_press(event):
            event_dict['btn_prs_shift_flag']  =(event.key=="shift")
            event_dict['btn_prs_inaxes_flag'] = event.inaxes
            event_dict['btn_prs_x'],    event_dict['btn_prs_y']     = event.x, event.y
            event_dict['btn_prs_xdata'],event_dict['btn_prs_ydata'] = event.xdata, event.ydata
            # クリックした画像を着目画像(current axes)に指定
            for ax_cand in fig.get_axes():
                if event.inaxes == ax_cand:
                    fig.sca(ax_cand)
                    ax_cand.spines['left'].set(color="#FF4500",linewidth=6)
                else:
                    ax_cand.spines['left'].set(color="black",linewidth=1)
            # ダブルクリックで最も大きい画像に合わせて表示領域リセット
            if (event.dblclick) and (event.button==1):
                config_list[0].ax.set_xlim(init_xy_pos[0][0], init_xy_pos[0][1])
                config_list[0].ax.set_ylim(init_xy_pos[1][0], init_xy_pos[1][1])
            pass


        def button_release(event):
            if event_dict['btn_prs_inaxes_flag']:
                ax_x_px,ax_y_px = int( (config_list[0].ax.bbox.x1-config_list[0].ax.bbox.x0) ), int( (config_list[0].ax.bbox.y1-config_list[0].ax.bbox.y0) )
                move_x,move_y = event_dict['btn_prs_x'] - event.x, event_dict['btn_prs_y'] - event.y
                lim_x,lim_y = config_list[0].ax.get_xlim(), config_list[0].ax.get_ylim()
                ax_img_pix_x,ax_img_pix_y = lim_x[1]-lim_x[0], lim_y[1]-lim_y[0]
                move_x_pix,move_y_pix = move_x/ax_x_px*ax_img_pix_x, move_y/ax_y_px*ax_img_pix_y

                if event_dict['btn_prs_shift_flag']:
                    x_lim = np.sort([event_dict['btn_prs_xdata'],event_dict['btn_prs_xdata']-move_x_pix])
                    y_lim = np.sort([event_dict['btn_prs_ydata'],event_dict['btn_prs_ydata']-move_y_pix])
                    config_list[0].ax.set_xlim(x_lim[0],x_lim[1])
                    config_list[0].ax.set_ylim(y_lim[int(bool(0-yud_mode))],y_lim[int(bool(1-yud_mode))])
                else:
                    config_list[0].ax.set_xlim(lim_x[0]+move_x_pix,lim_x[1]+move_x_pix)
                    config_list[0].ax.set_ylim(lim_y[0]+move_y_pix,lim_y[1]+move_y_pix)

                fig.canvas.draw()
                event_dict['btn_prs_x'],event_dict['btn_prs_y'],event_dict['btn_prs_xdata'],event_dict['btn_prs_ydata']=0,0,0,0
                event_dict['btn_prs_inaxes_flag'],event_dict['btn_prs_shift_flag']=False,False

        def key_press(event):
            event_dict['wheel_xlim']=config_list[0].ax.get_xlim()
            event_dict['wheel_ylim']=config_list[0].ax.get_ylim()
            event_dict['key_prs_xdata']=event.xdata
            event_dict['key_prs_ydata']=event.ydata
            keyboard_dict.get(event.key,q_hotkey__dummy)(event,config_list)
            pass

        def wheel_act(event):
            if (event.key=="shift") and (event.inaxes):
                event_dict['wheel_shift_flag'] = True
                # x
                event_dict['wheel_xlim'] = (event_dict['key_prs_xdata']-(event_dict['key_prs_xdata']-event_dict['wheel_xlim'][0])*(zoom_dict[event.button]),
                                            event_dict['key_prs_xdata']+(event_dict['wheel_xlim'][1]-event_dict['key_prs_xdata'])*(zoom_dict[event.button]))
                # y
                ylim_temp  = np.sort(event_dict['wheel_ylim'])
                event_dict['wheel_ylim'] = (event_dict['key_prs_ydata']-(event_dict['key_prs_ydata']-ylim_temp[0])*(zoom_dict[event.button]),
                                            event_dict['key_prs_ydata']+(ylim_temp[1]-event_dict['key_prs_ydata'])*(zoom_dict[event.button]))
            pass

        def key_release(event):
            if event_dict['wheel_shift_flag']:
                event_dict['wheel_shift_flag']=False
                config_list[0].ax.set_xlim(event_dict['wheel_xlim'][0],event_dict['wheel_xlim'][1])
                config_list[0].ax.set_ylim(event_dict['wheel_ylim'][int(bool(0-yud_mode))],event_dict['wheel_ylim'][int(bool(1-yud_mode))])
                fig.canvas.draw()
            pass

        fig.canvas.mpl_connect('button_press_event',button_press)
        fig.canvas.mpl_connect('button_release_event',button_release)
        fig.canvas.mpl_connect('key_press_event',key_press)
        fig.canvas.mpl_connect('key_release_event',key_release)
        fig.canvas.mpl_connect('scroll_event', wheel_act)

    mouse_key_event(config_list[0].fig)
    # return fig
    pass


class q_config:
    def __init__(self,
                 # すべての描画に共通した情報
                 fig, ax, ax_id, y_id, x_id,y_id_max, x_id_max,
                 # 画像描画に関する情報
                 data=None, data2=None, cmin=0, cmax=0, cmap=None, cbar=None, roi=None,
                 # 画像そのものに関する情報
                 min=None, max=None, h=None, w=None, h_max=None, w_max=None,
                 # グラフ描画にかかわる情報
                 label=None,alpha=None,color=None,edgecolor=None,
                 # plotにかかわる情報
                 marker=None,marker_size=None,linestyle=None,linewidth=None,
                # histにかかわる情報
                 interval=None,histtype=None):
        # すべての描画に共通した情報
        self.fig   = fig
        self.ax    = ax
        self.ax_id = ax_id
        self.y_id  = y_id
        self.x_id  = x_id
        self.y_id_max = y_id_max
        self.x_id_max = x_id_max
        self.graph = None

        # データに関する情報
        self.data  = data
        self.data2 = data2
        self.min   = min   # データの最小値
        self.max   = max   # データの最大値
        self.h     = h     # 画像の高さ
        self.w     = w     # 画像の幅
        self.h_max = h_max # すべての画像の高さの最大値
        self.w_max = w_max # すべての画像の幅の最小値

        # 画像描画に関する情報
        self.state     = 'normal'
        self.cmin      = {self.state:cmin} if cmin<cmax else {self.state:min}
        self.cmax      = {self.state:cmax} if cmin<cmax else {self.state:max}
        self.cmap      = cmap if (cmap in matplotlib_colormap_list) or (isinstance(cmap,matplotlib.colors.ListedColormap)) else 'viridis'
        self.cbar      = cbar
        self.roi       = roi
        if roi:
            self.ax.add_patch(self.roi)

        # グラフ描画にかかわる情報
        self.label     = label
        self.alpha     = alpha
        self.color     = color
        self.edgecolor = edgecolor
        self.graph_v = None
        self.legend  = None

        # plotにかかわる情報
        self.marker      = marker
        self.markersize  = marker_size
        self.linestyle   = linestyle
        self.linewidth   = linewidth

        # histにかかわる情報
        self.interval = interval
        self.histtype = histtype

        pass

    def update_clim(self,cmin,cmax):
        self.cmin[self.state]  = cmin
        self.cmax[self.state]  = cmax
        self.graph.set_clim(cmin, cmax)
        pass

    def get_roi_pos(self):
        roi_x_s,roi_y_s = self.roi.get_xy()
        roi_x_s,roi_y_s = np.clip(roi_x_s+0.5,0,None).astype(int),np.clip(roi_y_s+0.5,0,None).astype(int)
        roi_x_e,roi_y_e = (roi_x_s+self.roi.get_width()).astype(int),(roi_y_s+self.roi.get_height()).astype(int)
        return [roi_x_s,roi_x_e,roi_y_s,roi_y_e]

    def get_roi_img(self):
        roi_x_s,roi_x_e,roi_y_s,roi_y_e = self.get_roi_pos()
        return self.data[roi_y_s:roi_y_e, roi_x_s:roi_x_e], [roi_x_s, roi_x_e, roi_y_s, roi_y_e]

    def get_xlim_pos_for_img(self):
        img_xlim = self.ax.get_xlim()
        img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
        return img_xlim

    def get_ylim_pos_for_img(self):
        img_ylim = self.ax.get_ylim()
        img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]
        return img_ylim

    def get_xlim_pos(self):
        img_xlim = self.ax.get_xlim()
        return img_xlim

    def get_ylim_pos(self):
        img_ylim = self.ax.get_ylim()
        return img_ylim

    def get_part_img(self,x_pos=None,y_pos=None):
        x_pos = [np.clip(x_pos[0],0,self.w),np.clip(x_pos[1],0,self.w)]
        y_pos = np.sort([np.clip(y_pos[0],0,self.h),np.clip(y_pos[1],0,self.h)])
        return self.data[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1]], [x_pos, y_pos]


########################################################################################################################
#   ██████   █████  ████████  █████      ███████ ██   ██  █████  ██████  ██ ███    ██  ██████
#   ██   ██ ██   ██    ██    ██   ██     ██      ██   ██ ██   ██ ██   ██ ██ ████   ██ ██
#   ██   ██ ███████    ██    ███████     ███████ ███████ ███████ ██████  ██ ██ ██  ██ ██   ███
#   ██   ██ ██   ██    ██    ██   ██          ██ ██   ██ ██   ██ ██      ██ ██  ██ ██ ██    ██
#   ██████  ██   ██    ██    ██   ██     ███████ ██   ██ ██   ██ ██      ██ ██   ████  ██████
########################################################################################################################
def q_util_shaping_2dlist(input_data):
    if not isinstance(input_data, list):
        output_list_for_shape = [[0]]
        input_data = [[input_data]]
    else:
        lb = np.sum(np.array([isinstance(id, list) for id in input_data]))
        if lb==0:
            output_list_for_shape = [[0]*len(input_data)]
            input_data = [input_data]
        else:
            output_list_for_shape = [[0]*len(id) if isinstance(id, list) else [0] for id in input_data]
            input_data = [id if isinstance(id, list) else [id] for id in input_data]
    return input_data,output_list_for_shape

def q_util_shaping_2dlist_sub(input_data, main_list_for_shape):
    output_data = []
    if not isinstance(input_data, list):
        for y in np.arange(len(main_list_for_shape)):
            output_data.append([input_data for x in range(len(main_list_for_shape[y]))])
    else:
        temp2 = 0
        for y in np.arange(len(input_data)):
            if not isinstance(input_data[y],list):
                temp2 = temp2+1

        if len(input_data)==temp2:#input_dataが1次元listだった
            output_data = [input_data]
        elif temp2==0:#input_dataが2次元配列だった
            output_data = input_data

        for y in np.arange(len(main_list_for_shape)):
            if len(main_list_for_shape[y]) != len(output_data[y]):
                print('warning:list shape error!')

    return output_data


def q_util_shaping_2dlist_color(input_color, main_list_for_shape):
    qutinawa_color_list = ['#f50035','#06b137','#00aff5',
                           '#ffca09','#00e0b4','#3a00cc',
                           '#e000a1','#9eff5d','#8d00b8',
                           '#7e5936','#367d4c','#364c7d',
                           '#f27993','#85de9f','#7acef0',
                           '#fce281','#97ded0','#937dc9',
                           '#9496ff','#8c8c8c',]

    output_color = []
    if input_color in matplotlib_colormap_list:
        cm = plt.cm.get_cmap(input_color)
        main_data_num = sum(len(v) for v in main_list_for_shape)
        i = 0
        for y in np.arange(len(main_list_for_shape)):
            temp=[]
            for x in range(len(main_list_for_shape[y])):
                temp.append(cm(i/np.clip(main_data_num-1,1,None)))
                i = i+1
            output_color.append(temp)
    elif input_color=='qutinawa_color':
        i = 0
        for y in np.arange(len(main_list_for_shape)):
            temp=[]
            for x in range(len(main_list_for_shape[y])):
                temp.append(qutinawa_color_list[i%len(qutinawa_color_list)])
                i = i+1
            output_color.append(temp)
    else:
        if isinstance(input_color,list):
            output_color = input_color
        else:
            for y in np.arange(len(main_list_for_shape)):
                output_color.append([input_color for x in range(len(main_list_for_shape[y]))])

    return output_color

def q_util_shaping_2dlist_label_arange(main_list_for_shape):
    output_data = []
    i = 0
    for y in np.arange(len(main_list_for_shape)):
        temp = []
        for x in range(len(main_list_for_shape[y])):
            temp.append(str(i))
            i = i+1
        output_data.append(temp)
    return output_data

def q_util_shaping_2dlist_arange(input_data,main_list_for_shape):
    output_data = []
    i = 0
    for y in np.arange(len(main_list_for_shape)):
        temp = []
        for x in range(len(main_list_for_shape[y])):
            temp.append(np.arange(np.shape(np.squeeze(np.reshape(input_data[y][x],(1,-1))))[0]))
        output_data.append(temp)
    return output_data


########################################################################################################################
#   ██   ██  ██████  ████████ ██   ██ ███████ ██    ██
#   ██   ██ ██    ██    ██    ██  ██  ██       ██  ██
#   ███████ ██    ██    ██    █████   █████     ████
#   ██   ██ ██    ██    ██    ██  ██  ██         ██
#   ██   ██  ██████     ██    ██   ██ ███████    ██
########################################################################################################################
def q_hotkey__dummy(event,config_list):
    pass

def q_hotkey__png_save(event,config_list):
    config_list[0].fig.savefig(datetime.datetime.now().strftime('imageq-%Y_%m_%d_%H_%M_%S')+'.png',bbox_inches='tight')
    pass

def q_util_main_figure_close(fig, ana_fig):
    def close_event(event):
        ana_fig.clf()
        plt.close(ana_fig)
        pass
    fig.canvas.mpl_connect('close_event', close_event)
    pass

def q_hotkey__visible(event,config_list):
    f_dict = {'f1': 0,'f2': 1, 'f3': 2,
              'f4': 3,'f5': 4, 'f6': 5,
              'f7': 6,'f8': 7, 'f9': 8,
              'f10':9,'f11':10, 'f12':11,
              'shift+f1': 12,'shift+f2': 13,'shift+f3': 14,
              'shift+f4': 15,'shift+f5': 16,'shift+f6': 17,
              'shift+f7': 18,'shift+f8': 19,'shift+f9': 20,
              'shift+f10':21,'shift+f11':22,'shift+f12':23,}
    # print(config_list[f_dict[event.key]].graph_v)
    visible = not config_list[f_dict[event.key]].graph_v.get_visible()
    config_list[f_dict[event.key]].graph_v.set_visible(visible)
    config_list[f_dict[event.key]].fig.canvas.draw()
    pass

def q_hotkey_util__gca(config_list):
    return [cl for cl in config_list if plt.gca()==cl.ax][0]

############################################################################ add diff mul div
def q_hotkey_util__admd(event,config_list,mode):
    def shape_arrange(img1,img2):
        s1 = np.shape(img1)
        s2 = np.shape(img2)
        if s1 != s2:
            s1 = np.pad(s1,(0, 3-len(s1)))
            s2 = np.pad(s2,(0, 3-len(s2)))
            ims = np.maximum(s1,s2)
            img1 = np.pad(np.atleast_3d(img1), ((0,ims[0]-s1[0]),(0,ims[1]-s1[1]),(0,np.clip(ims[2]-s1[2]-1,0,None))), mode='edge')
            img2 = np.pad(np.atleast_3d(img2), ((0,ims[0]-s2[0]),(0,ims[1]-s2[1]),(0,np.clip(ims[2]-s2[2]-1,0,None))), mode='edge')
        return img1,img2

    def add(img1,img2):
        return img1+img2
    def diff(img1,img2):
        return img1-img2
    def mul(img1,img2):
        return img1*img2
    def div(img1,img2):
        return img2/img1

    finc_dict = {'add':add,'diff':diff,'mul':mul,'div':div}
    if config_list[0].state=='normal':
        gca_config = q_hotkey_util__gca(config_list)
        for now_config in config_list:
            now_config.state=mode
            if now_config!=gca_config:
                admded_img = finc_dict[mode](*shape_arrange(gca_config.data, now_config.data))
                if np.ndim(admded_img) in [3, 4]:
                    admded_img_min = np.nanmin(admded_img)
                    admded_img_max = np.nanmax(admded_img)
                    admded_img = np.clip((admded_img.astype('float64') - admded_img_min) / (admded_img_max - admded_img_min), 0, 1)
                now_config.graph.set_array(admded_img)
                cmin,cmax = np.nanmin(admded_img),np.nanmax(admded_img)
                now_config.update_clim(cmin,cmax)
        config_list[0].fig.canvas.draw()

    elif config_list[0].state==mode:
        for now_config in config_list:
            now_config.state='normal'
            now_config.graph.set_array(now_config.data)
            cmin=now_config.cmin.get('normal', np.min(now_config.data))
            cmax=now_config.cmax.get('normal', np.max(now_config.data))
            now_config.update_clim(cmin,cmax)
            now_config.update_clim(cmin,cmax)#2重に書いておかないと何故かうまくいかない
        config_list[0].fig.canvas.draw()
    pass

def q_hotkey__add(event,config_list):
    q_hotkey_util__admd(event,config_list,mode='add')
def q_hotkey__diff(event,config_list):
    q_hotkey_util__admd(event,config_list,mode='diff')
def q_hotkey__mul(event,config_list):
    q_hotkey_util__admd(event,config_list,mode='mul')
def q_hotkey__div(event,config_list):
    q_hotkey_util__admd(event,config_list,mode='div')


############################################################################ clim
def q_hotkey_util__climMANUAL(gca_config, plusminus, gain):
    caxis_min,caxis_max = gca_config.cmin.get(gca_config.state, np.min(gca_config.data)), gca_config.cmax.get(gca_config.state, np.max(gca_config.data))
    diff = caxis_max-caxis_min
    min_change = plusminus[0]*diff*gain
    max_change = plusminus[1]*diff*gain
    gca_config.update_clim(caxis_min+min_change,caxis_max+max_change)
    gca_config.fig.canvas.draw()
    pass

def q_hotkey__climMANUAL_top_down(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[0, -1], gain=0.025)
def q_hotkey__climMANUAL_btm_up(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[1,  0], gain=0.025)
def q_hotkey__climMANUAL_slide_up(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[1, 1], gain=0.025)
def q_hotkey__climMANUAL_slide_down(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[-1, -1], gain=0.025)
def q_hotkey__climMANUAL_top_up(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[0, 1], gain=0.025)
def q_hotkey__climMANUAL_btm_down(event, config_list):
    q_hotkey_util__climMANUAL(q_hotkey_util__gca(config_list), plusminus=[-1, 0], gain=0.025)

def q_hotkey__climAUTO(event,config_list):
    gca_config = q_hotkey_util__gca(config_list)
    now_lim_x = np.clip((np.array(gca_config.ax.get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(gca_config.ax.get_ylim()) + 0.5).astype(int),0,None)
    temp = gca_config.data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
    gca_config.update_clim(np.min(temp),np.max(temp))
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__climWHOLE(event,config_list):
    now_lim_x = np.clip((np.array(config_list[0].ax.get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(config_list[0].ax.get_ylim()) + 0.5).astype(int),0,None)
    whole_max = np.max(np.array([np.max(cl.data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for cl in config_list]))
    whole_min = np.min(np.array([np.min(cl.data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for cl in config_list]))
    for now_config in config_list:
        now_config.update_clim(whole_min,whole_max)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__climEACH(event,config_list):
    now_lim_x = np.clip((np.array(config_list[0].ax.get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(config_list[0].ax.get_ylim()) + 0.5).astype(int),0,None)
    for now_config in config_list:
        temp = now_config.data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
        now_config.update_clim(np.min(temp),np.max(temp))
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__climSYNC(event,config_list):
    gca_config = q_hotkey_util__gca(config_list)
    new_cmin = gca_config.cmin[gca_config.state]
    new_cmax = gca_config.cmax[gca_config.state]
    for now_config in config_list:
        now_config.update_clim(new_cmin,new_cmax)
    config_list[0].fig.canvas.draw()
    pass


############################################################################ roi
def q_hotkey__roiset(event,config_list):
    if event.inaxes:
        mouse_x, mouse_y = int(event.xdata + 0.5)-0.5, int(event.ydata + 0.5)-0.5
        for now_config in config_list:
            now_config.roi.set_xy((mouse_x,mouse_y))
        config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roiwidthUP(event,config_list):
    for now_config in config_list:
        now_config.roi.set_width(now_config.roi.get_width() + 2)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roiheightUP(event,config_list):
    for now_config in config_list:
        now_config.roi.set_height(now_config.roi.get_height() + 2)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roiwidthDOWN(event,config_list):
    for now_config in config_list:
        now_config.roi.set_width(now_config.roi.get_width() - 2)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roiheightDOWN(event,config_list):
    for now_config in config_list:
        now_config.roi.set_height(now_config.roi.get_height() - 2)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roireset(event,config_list):
    for now_config in config_list:
        now_config.roi.set_xy((-11.5, -11.5))
        now_config.roi.set_height(11)
        now_config.roi.set_width(11)
    config_list[0].fig.canvas.draw()
    pass

def q_hotkey__roistats(event,config_list):
    pos = config_list[0].get_roi_pos()
    print('=== ROI stats === ' + ' (x='+str(pos[0])+'~'+str(pos[1])+', y='+str(pos[2])+'~'+str(pos[3])+')' )
    print( 'img#'.ljust(4)+'\t'
          +'mean'.ljust(20)+'\t'
          +'max'.ljust(20)+'\t'
          +'min'.ljust(20)+'\t'
          +'median'.ljust(20)+'\t'
          +'std'.ljust(20)+'\t')
    for now_config in config_list:
        temp_img,pos = now_config.get_roi_img()
        print( str(now_config.ax_id      ).ljust(4)+'\t'
              +str(np.nanmean(temp_img)  ).ljust(20)+'\t'
              +str(np.nanmax(temp_img)   ).ljust(20)+'\t'
              +str(np.nanmin(temp_img)   ).ljust(20)+'\t'
              +str(np.nanmedian(temp_img)).ljust(20)+'\t'
              +str(np.nanstd(temp_img)   ).ljust(20)+'\t')
    print('')
    pass


def q_hotkey__roipixval(event,config_list):
    ana_fig = plt.figure()
    pos = config_list[0].get_roi_pos()
    for now_config in config_list:
        temp_img,pos = now_config.get_roi_img()

        ax = ana_fig.add_subplot(config_list[-1].y_id+1, config_list[-1].x_id+1, now_config.ax_id, picker=True)
        ax.imshow(temp_img, interpolation='nearest',cmap=now_config.cmap)

        ys, xs = np.meshgrid(range(temp_img.shape[0]), range(temp_img.shape[1]), indexing='ij')

        if np.ndim(temp_img)==2:
            v_a = {0:'bottom',1:'top'}
            for (xi, yi, val) in zip(xs.flatten(), ys.flatten(), temp_img.flatten()):
                ax.text(xi, yi, '{0:.2f}'.format(val),
                        horizontalalignment='center', verticalalignment=v_a[xi%2],
                        color='#ff21cb',fontweight='bold',fontsize=13)
        elif np.ndim(temp_img)==3:
            for (xi, yi, val_0ch, val_1ch, val_2ch) in zip(xs.flatten(), ys.flatten(),
                                                           temp_img[:,:,0].flatten(), temp_img[:,:,1].flatten(), temp_img[:,:,2].flatten()):
                ax.text(xi, yi-0.25, '{0:.2f}'.format(val_0ch),
                        horizontalalignment='center', verticalalignment='center',
                        color='#ff2a00',fontweight='bold',fontsize=13)
                ax.text(xi, yi, '{0:.2f}'.format(val_1ch),
                        horizontalalignment='center', verticalalignment='center',
                        color='#00ff40',fontweight='bold',fontsize=13)
                ax.text(xi, yi+0.25, '{0:.2f}'.format(val_2ch),
                        horizontalalignment='center', verticalalignment='center',
                        color='#0048ff',fontweight='bold',fontsize=13)

    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    q_util_main_figure_close(config_list[0].fig, ana_fig)
    ana_fig.show()
    pass

############################################################################ hist
def q_hotkey_util__hist(event,config_list,mode):
    i = 0
    input_list = []
    interval_list = []
    edgecolor = []
    for y in np.arange(config_list[0].y_id_max):
        temp_input = []
        temp_inter = []
        temp_ec = []
        for x in np.arange(config_list[0].x_id_max):
            if (i<len(config_list)) and (y==config_list[i].y_id) and (x==config_list[i].x_id):
                if mode=='roi':
                    temp_img = (config_list[i].get_roi_img())[0]
                elif mode=='view':
                    temp_img = (config_list[i].get_part_img(config_list[i].get_xlim_pos_for_img(), config_list[i].get_ylim_pos_for_img()))[0]

                il = (np.nanmax(temp_img)-np.nanmin(temp_img))/128
                if np.ndim(temp_img)==2:
                    temp_input.append(temp_img)
                    temp_inter.append(il)
                    temp_ec.append('k')
                elif np.ndim(temp_img)==3:
                    cl = ['r','g','b']
                    for ch in np.arange(np.shape(temp_img)[2]):
                        temp_input.append(temp_img[:,:,ch])
                        temp_inter.append(il)
                        temp_ec.append(cl[ch])
                i = i + 1
        input_list.append(temp_input)
        interval_list.append(temp_inter)
        edgecolor.append(temp_ec)

    ana_fig = histq(input_list,interval_list=interval_list,edgecolor_list=edgecolor,overlay=False)
    q_util_main_figure_close(config_list[0].fig, ana_fig)
    pass

def q_hotkey__roihist(event,config_list):
    q_hotkey_util__hist(event,config_list,mode='roi')

def q_hotkey__hist(event,config_list):
    q_hotkey_util__hist(event,config_list,mode='view')


############################################################################ lineprof
def q_hotkey_util__lineprof(config_list,event,mode):
    img_xlim = config_list[0].get_xlim_pos_for_img()
    img_ylim = config_list[0].get_ylim_pos_for_img()

    if mode=='H':
        mouse_y = int(event.ydata + 0.5)
        line_img_xpos = img_xlim
        line_img_ypos = [mouse_y, mouse_y+1]
        line_plot_idx = 0
        ax_line_xy1   = (img_xlim[0],mouse_y)
        ax_line_xy2   = (img_xlim[1],mouse_y)
        ax_line_xy3   = (img_xlim[0],mouse_y)
        ax_line_xy4   = (img_xlim[1],mouse_y)
    elif mode=='V':
        mouse_x = int(event.xdata + 0.5)
        line_img_xpos = [mouse_x, mouse_x+1]
        line_img_ypos = img_ylim
        line_plot_idx = 1
        ax_line_xy1   = (mouse_x, img_ylim[0])
        ax_line_xy2   = (mouse_x, img_ylim[1])
        ax_line_xy3   = (mouse_x, img_ylim[0])
        ax_line_xy4   = (mouse_x, img_ylim[1])
    elif mode=='roiH':
        roi_x_s,roi_x_e,roi_y_s,roi_y_e = config_list[0].get_roi_pos()
        line_img_xpos = img_xlim
        line_img_ypos = [roi_y_s,roi_y_e]
        line_plot_idx = 0
        ax_line_xy1   = (img_xlim[0],roi_y_s)
        ax_line_xy2   = (img_xlim[1],roi_y_s)
        ax_line_xy3   = (img_xlim[0],roi_y_e)
        ax_line_xy4   = (img_xlim[1],roi_y_e)
    elif mode=='roiV':
        roi_x_s,roi_x_e,roi_y_s,roi_y_e = config_list[0].get_roi_pos()
        line_img_xpos = [roi_x_s,roi_x_e]
        line_img_ypos = img_ylim
        line_plot_idx = 1
        ax_line_xy1   = (roi_x_s, img_ylim[0])
        ax_line_xy2   = (roi_x_s, img_ylim[1])
        ax_line_xy3   = (roi_x_e, img_ylim[0])
        ax_line_xy4   = (roi_x_e, img_ylim[1])

    ana_fig = plt.figure()
    plot_sub_ax = ana_fig.add_subplot(2,len(config_list),(1,len(config_list)),picker=True)
    for cn,now_config in enumerate(config_list):
        line_img,pos = now_config.get_part_img(x_pos=line_img_xpos,y_pos=line_img_ypos)
        # print(line_img,pos)
        ####################### plot
        if np.ndim(line_img)==2:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(np.mean(line_img,axis=line_plot_idx)),
                             label=now_config.ax_id)
        elif np.ndim(line_img)==3:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(np.mean(line_img[:,:,0],axis=line_plot_idx)),
                             label=str(now_config.ax_id)+'(0ch)')
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(np.mean(line_img[:,:,1],axis=line_plot_idx)),
                             label=str(now_config.ax_id)+'(1ch)')
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(np.mean(line_img[:,:,2],axis=line_plot_idx)),
                             label=str(now_config.ax_id)+'(2ch)')

        ####################### image
        part_img,aaa   = now_config.get_part_img(x_pos=img_xlim,y_pos=img_ylim)
        img_sub_ax = ana_fig.add_subplot(2,len(config_list),len(config_list)+cn+1,picker=True)
        img_sub_ax.axline(xy1=ax_line_xy1,xy2=ax_line_xy2,color='pink')
        img_sub_ax.axline(xy1=ax_line_xy3,xy2=ax_line_xy4,color='pink')
        img_sub_im = img_sub_ax.imshow(part_img,interpolation='nearest', cmap=now_config.cmap,
                                       extent=[aaa[0][0],aaa[0][1],aaa[1][1],aaa[1][0]],
                                       aspect='equal')
        img_sub_im.set_clim(now_config.cmin[now_config.state],now_config.cmax[now_config.state])

    plot_sub_ax.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    q_util_main_figure_close(config_list[0].fig, ana_fig)
    ana_fig.show()
    pass


def q_hotkey__lineprofH(event,config_list):
    if event.inaxes:
        q_hotkey_util__lineprof(config_list,event,mode='H')

def q_hotkey__lineprofV(event,config_list):
    if event.inaxes:
        q_hotkey_util__lineprof(config_list,event,mode='V')

def q_hotkey__lineprofHmean(event,config_list):
    q_hotkey_util__lineprof(config_list,event,mode='roiH')

def q_hotkey__lineprofVmean(event,config_list):
    q_hotkey_util__lineprof(config_list,event,mode='roiV')

############################################################################ WaveformMonitor
def q_hotkey__WaveformMonitor(event,config_list):
    i = 0
    x_list=[]
    y_list=[]
    c_list = []
    for y in np.arange(config_list[0].y_id_max):
        temp_x = []
        temp_y = []
        temp_c = []
        for x in np.arange(config_list[0].x_id_max):
            if (i<len(config_list)) and (y==config_list[i].y_id) and (x==config_list[i].x_id):
                if np.ndim(config_list[i].data)==2:
                    temp_x.append(np.reshape(np.tile(np.arange(config_list[i].w)[np.newaxis,:],(config_list[i].h,1)),(1,-1)))
                    temp_y.append(np.reshape(config_list[i].data,(1,-1)))
                    temp_c.append('k')
                elif np.ndim(config_list[i].data)==3:
                    cc = ['r','g','b']
                    for j in np.arange(np.shape(config_list[i].data)[2]):
                        temp_x.append(np.reshape(np.tile(np.arange(config_list[i].w)[np.newaxis,:],(config_list[i].h,1)),(1,-1)))
                        temp_y.append(np.reshape(config_list[i].data[:,:,j],(1,-1)))
                        temp_c.append(cc[j])
                i = i + 1
        x_list.append(temp_x)
        y_list.append(temp_y)
        c_list.append(temp_c)

    ana_fig = plotq(x_list,y_list,color_list=c_list,alpha_list=0.1,linewidth_list=0,marker_list='o',markersize_list=2,overlay=False)
    q_util_main_figure_close(config_list[0].fig, ana_fig)
    pass

def q_hotkey__posprint(event,config_list):
    if config_list[0].roi:# 画像
        view_pos = [config_list[0].get_xlim_pos_for_img(),config_list[0].get_ylim_pos_for_img()]
        mouse_pos = [int(event.xdata + 0.5),int(event.ydata + 0.5)]
        roi_pos = config_list[0].get_roi_pos()
        roi_pos = [roi_pos[0:2],roi_pos[2:]]
        print(  ('view[[xs,xe],[ys,ye]]:'+str(view_pos)).ljust(50)+'\t'
                +('mouse[x,y]:'+str(mouse_pos)).ljust(40)+'\t'
                +('roi[[xs,xe],[ys,ye]]:'+str(roi_pos)).ljust(50)+'\t')
    else:
        view_pos = [config_list[0].get_xlim_pos(),config_list[0].get_ylim_pos()]
        mouse_pos = [event.xdata ,event.ydata]
        print(  ('view[[xs,xe],[ys,ye]]:'+str(view_pos)).ljust(120)+'\t'
                +('mouse[x,y]:'+str(mouse_pos)).ljust(80)+'\t')
    pass


########################################################################################################################
#   ██████  ██       ██████  ████████
#   ██   ██ ██      ██    ██    ██
#   ██████  ██      ██    ██    ██
#   ██      ██      ██    ██    ██
#   ██      ███████  ██████     ██
########################################################################################################################
def q_util_plot_for_plotq(q_conf):
    q_conf.graph = q_conf.ax.plot(np.squeeze(np.reshape(q_conf.data,(1,-1))),
                                  np.squeeze(np.reshape(q_conf.data2,(1,-1))),
                                  label     = q_conf.label,
                                  color     = q_conf.color,
                                  alpha     = q_conf.alpha,
                                  marker    = q_conf.marker,
                                  markersize= q_conf.markersize,
                                  linestyle = q_conf.linestyle,
                                  linewidth = q_conf.linewidth,
                                  )
    q_conf.graph_v = q_conf.graph[0]
    pass

def plotq(input_list_x,
          input_list_y=None,
          marker_list = 'None',
          markersize_list = 7,
          linestyle_list='-',
          linewidth_list=2,
          color_list = 'qutinawa_color',
          alpha_list = 1.0,
          label_list = None,
          overlay=True,
          fig = None,
          ):

    ############################### 準備
    plt.interactive(False)
    if not fig:
        fig = plt.figure()

    ############################### 必ず2次元listの形状にする
    if input_list_y:
        input_list_x,input_list_for_shape = q_util_shaping_2dlist(input_list_x)
        input_list_y,_ = q_util_shaping_2dlist(input_list_y)
    else:
        input_list_y,input_list_for_shape = q_util_shaping_2dlist(input_list_x)
        input_list_x = q_util_shaping_2dlist_arange(input_list_y,input_list_for_shape)

    marker_list      = q_util_shaping_2dlist_sub(input_data=marker_list,
                                                 main_list_for_shape=input_list_for_shape)
    markersize_list  = q_util_shaping_2dlist_sub(input_data=markersize_list,
                                                 main_list_for_shape=input_list_for_shape)
    linestyle_list   = q_util_shaping_2dlist_sub(input_data=linestyle_list,
                                                 main_list_for_shape=input_list_for_shape)
    linewidth_list   = q_util_shaping_2dlist_sub(input_data=linewidth_list,
                                                 main_list_for_shape=input_list_for_shape)
    alpha_list       = q_util_shaping_2dlist_sub(input_data=alpha_list,
                                                 main_list_for_shape=input_list_for_shape)
    label_list       = q_util_shaping_2dlist_sub(input_data=label_list,
                                                 main_list_for_shape=input_list_for_shape) if label_list else q_util_shaping_2dlist_label_arange(main_list_for_shape=input_list_for_shape)
    color_list       = q_util_shaping_2dlist_color(input_color=color_list,
                                                   main_list_for_shape=input_list_for_shape)

    ############################### 各描画
    y_id_max = len(input_list_for_shape)
    x_id_max = np.max(np.array([len(i) for i in input_list_for_shape]))
    config_list = []
    xxx = [np.inf, -np.inf]
    yyy = [np.inf, -np.inf]
    if overlay:
        ax_id = 1
        ax = fig.add_subplot(1,1,1,picker=True)
        y_id_max = 1
        x_id_max = 1

    for y_id,temp_list in enumerate(input_list_for_shape):
        for x_id,temp2_data in enumerate(temp_list):
            if not overlay:
                ax_id = x_id_max*y_id+x_id+1
                if ax_id==1:
                    ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True)
                else:
                    ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True,sharex=config_list[0].ax,sharey=config_list[0].ax)
                    # ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True,sharex=fig.get_axes()[0],sharey=fig.get_axes()[0])


            config_list.append(q_config(fig=fig, ax=ax, ax_id=ax_id, y_id=y_id, x_id=x_id,y_id_max=y_id_max,x_id_max=x_id_max,
                                        data =input_list_x[y_id][x_id].astype(float),
                                        data2=input_list_y[y_id][x_id].astype(float),
                                        label=label_list[y_id][x_id], color=color_list[y_id][x_id],alpha=alpha_list[y_id][x_id],
                                        marker=marker_list[y_id][x_id], marker_size=markersize_list[y_id][x_id],
                                        linestyle=linestyle_list[y_id][x_id], linewidth=linewidth_list[y_id][x_id],
                                        ))
            q_util_plot_for_plotq(q_conf=config_list[-1])
            xxx[0] = np.minimum(xxx[0], np.nanmin(input_list_x[y_id][x_id].astype(float)))
            xxx[1] = np.maximum(xxx[1], np.nanmax(input_list_x[y_id][x_id].astype(float)))
            yyy[0] = np.minimum(yyy[0], np.nanmin(input_list_y[y_id][x_id].astype(float)))
            yyy[1] = np.maximum(yyy[1], np.nanmax(input_list_y[y_id][x_id].astype(float)))
            x_spc = (xxx[1]-xxx[0])*0.0495
            y_spc = (yyy[1]-yyy[0])*0.0495
    ############################### キーボードショートカット追加
    q_basic(config_list=config_list,init_xy_pos=[[xxx[0]-x_spc, xxx[1]+x_spc], [yyy[0]-y_spc, yyy[1]+y_spc]],yud_mode=0,
            keyboard_dict={# png保存
                'P':q_hotkey__png_save,
                # visible
                'f1':q_hotkey__visible,'f2':q_hotkey__visible,'f3':q_hotkey__visible,'f4':q_hotkey__visible,'f5':q_hotkey__visible,'f6':q_hotkey__visible,
                'f7':q_hotkey__visible,'f8':q_hotkey__visible,'f9':q_hotkey__visible,'f10':q_hotkey__visible,'f11':q_hotkey__visible,'f12':q_hotkey__visible,
                'shift+f1': q_hotkey__visible,'shift+f2': q_hotkey__visible,'shift+f3': q_hotkey__visible,'shift+f4': q_hotkey__visible,'shift+f5': q_hotkey__visible,'shift+f6': q_hotkey__visible,
                'shift+f7': q_hotkey__visible,'shift+f8': q_hotkey__visible,'shift+f9': q_hotkey__visible,'shift+f10':q_hotkey__visible,'shift+f11':q_hotkey__visible,'shift+f12':q_hotkey__visible,
                '@':q_hotkey__posprint
            }
            )

    ############################### 表示
    for conf in config_list:
        conf.legend=conf.ax.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    fig.show()

    return fig



########################################################################################################################
#   ██   ██ ██ ███████ ████████  ██████   ██████  ██████   █████  ███    ███
#   ██   ██ ██ ██         ██    ██    ██ ██       ██   ██ ██   ██ ████  ████
#   ███████ ██ ███████    ██    ██    ██ ██   ███ ██████  ███████ ██ ████ ██
#   ██   ██ ██      ██    ██    ██    ██ ██    ██ ██   ██ ██   ██ ██  ██  ██
#   ██   ██ ██ ███████    ██     ██████   ██████  ██   ██ ██   ██ ██      ██
########################################################################################################################
def q_util_hist_for_histq(q_conf):
    hist_bins = np.arange(q_conf.min, q_conf.max + q_conf.interval*2, q_conf.interval)
    q_conf.graph = q_conf.ax.hist(np.squeeze(np.reshape(q_conf.data,(1,-1))),
                                  bins=hist_bins,
                                  label=q_conf.label,
                                  # color=q_conf.color,
                                  edgecolor=q_conf.edgecolor,
                                  alpha=q_conf.alpha,
                                  histtype=q_conf.histtype)
    q_conf.graph_v = q_conf.graph[2][0]
    pass

def histq(input_list,
          interval_list=1.0,
          label_list=None,
          alpha_list = 0.75,
          edgecolor_list = 'qutinawa_color',
          color_list = 'qutinawa_color',
          histtype='step',
          overlay=True
          ):

    ############################### 準備
    plt.interactive(False)
    fig = plt.figure()

    ############################### 必ず2次元listの形状にする
    input_list,input_list_for_shape = q_util_shaping_2dlist(input_list)
    interval_list   = q_util_shaping_2dlist_sub(input_data=interval_list,
                                                main_list_for_shape=input_list_for_shape)
    label_list   = q_util_shaping_2dlist_sub(input_data=label_list,
                                             main_list_for_shape=input_list_for_shape) if label_list else q_util_shaping_2dlist_label_arange(main_list_for_shape=input_list_for_shape)
    alpha_list   = q_util_shaping_2dlist_sub(input_data=alpha_list,
                                             main_list_for_shape=input_list_for_shape)
    edgecolor_list = q_util_shaping_2dlist_color(input_color=edgecolor_list,
                                                 main_list_for_shape=input_list_for_shape)
    color_list     = q_util_shaping_2dlist_color(input_color=color_list,
                                                 main_list_for_shape=input_list_for_shape)

    ############################### 各描画
    y_id_max = len(input_list)
    x_id_max = np.max(np.array([len(i) for i in input_list]))
    config_list = []
    if overlay:
        ax_id = 1
        ax = fig.add_subplot(1,1,1,picker=True)
        y_id_max = 1
        x_id_max = 1

    for y_id,temp_list in enumerate(input_list):
        for x_id,target_data in enumerate(temp_list):
            if not overlay:
                ax_id = x_id_max*y_id+x_id+1
                if ax_id==1:
                    ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True)
                else:
                    ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True,sharex=config_list[0].ax,sharey=config_list[0].ax)

            config_list.append(q_config(fig=fig, ax=ax, ax_id=ax_id, y_id=y_id, x_id=x_id, y_id_max=y_id_max,x_id_max=x_id_max,
                                        data=target_data.astype(float),
                                        min=np.nanmin(target_data), max=np.nanmax(target_data),
                                        interval=interval_list[y_id][x_id],label=label_list[y_id][x_id],
                                        alpha=alpha_list[y_id][x_id],edgecolor=edgecolor_list[y_id][x_id],color=color_list[y_id][x_id],
                                        histtype=histtype
                                        ))
            q_util_hist_for_histq(q_conf=config_list[-1])

    x_min = []
    x_max = []
    y_max = []
    for conf in config_list:
        x_min.append(conf.min)
        x_max.append(conf.max)
        y_max.append(np.max(conf.graph[0]))
    x_min = np.min(np.array(x_min))
    x_max = np.max(np.array(x_max))
    y_max = np.max(np.array(y_max))
    x_spc = (x_max-x_min)*0.0495

    ############################### キーボードショートカット追加
    q_basic(config_list=config_list,init_xy_pos=[[x_min-x_spc, x_max+x_spc],[0, y_max*1.05]],yud_mode=0,
            keyboard_dict={# png保存
                'P':q_hotkey__png_save,
                # visible
                'f1':q_hotkey__visible,'f2':q_hotkey__visible,'f3':q_hotkey__visible,'f4':q_hotkey__visible,'f5':q_hotkey__visible,'f6':q_hotkey__visible,
                'f7':q_hotkey__visible,'f8':q_hotkey__visible,'f9':q_hotkey__visible,'f10':q_hotkey__visible,'f11':q_hotkey__visible,'f12':q_hotkey__visible,
                'shift+f1': q_hotkey__visible,'shift+f2': q_hotkey__visible,'shift+f3': q_hotkey__visible,'shift+f4': q_hotkey__visible,'shift+f5': q_hotkey__visible,'shift+f6': q_hotkey__visible,
                'shift+f7': q_hotkey__visible,'shift+f8': q_hotkey__visible,'shift+f9': q_hotkey__visible,'shift+f10':q_hotkey__visible,'shift+f11':q_hotkey__visible,'shift+f12':q_hotkey__visible,
            }
            )

    ############################### 表示
    for conf in config_list:
        conf.legend=conf.ax.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    fig.show()

    return fig

########################################################################################################################
#   ██ ███    ███  █████   ██████  ███████
#   ██ ████  ████ ██   ██ ██       ██
#   ██ ██ ████ ██ ███████ ██   ███ █████
#   ██ ██  ██  ██ ██   ██ ██    ██ ██
#   ██ ██      ██ ██   ██  ██████  ███████
########################################################################################################################
def q_util_imshow_for_imageq(q_conf):
    # 3,4ch画像をcmin~cmaxのレンジで正規化するため、値域調整 & 4ch以上の画像は表示できないのでエラー返して終了
    if np.ndim(q_conf.data) in [3, 4]:
        print("imageq-Warning: The image was normalized to 0-1 and clipped in the cmin-cmax range for a 3-channel image.")
        q_conf.data = np.clip((q_conf.data.astype('float64') - q_conf.cmin[q_conf.state]) / (q_conf.cmax[q_conf.state] - q_conf.cmin[q_conf.state]), 0, 1)
    elif np.ndim(q_conf.data)==1 or np.ndim(q_conf.data)>4:
        print("imageq-Warning: Cannot draw data other than 2-, 3-, and 4-dimensional.")
        return -1

    q_conf.graph = q_conf.ax.imshow(q_conf.data, interpolation='nearest', cmap=q_conf.cmap)
    q_conf.update_clim(q_conf.cmin[q_conf.state],q_conf.cmax[q_conf.state])
    q_conf.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    q_conf.ax.tick_params(bottom=False, left=False, right=False, top=False)

    # colorbar表示
    if q_conf.cbar:
        divider = make_axes_locatable(q_conf.ax)
        ax_cbar = divider.new_horizontal(size="5%", pad=0.075)
        q_conf.ax.figure.add_axes(ax_cbar)
        q_conf.fig.colorbar(q_conf.graph, cax=ax_cbar)
        pass

    pass

def imageq(target_img_list,caxis_list=(0,0),cmap_list='viridis',disp_cbar_list=True):

    ############################### 準備
    plt.interactive(False)
    fig = plt.figure()

    ############################### 必ず2次元listの形状にする
    target_img_list,target_img_list_for_shape = q_util_shaping_2dlist(target_img_list)
    caxis_list     = q_util_shaping_2dlist_sub(caxis_list,      target_img_list_for_shape)
    cmap_list      = q_util_shaping_2dlist_sub(cmap_list,       target_img_list_for_shape)
    disp_cbar_list = q_util_shaping_2dlist_sub(disp_cbar_list,  target_img_list_for_shape)

    ############################### 各imshow描画
    
    y_id_max = len(target_img_list)
    x_id_max = np.max(np.array([len(i) for i in target_img_list]))
    h_max,w_max = np.max(np.array([np.shape(j)[0:2] for i in target_img_list for j in i]),axis=0)
    config_list = []
    for y_id,temp_list in enumerate(target_img_list):
        for x_id,target_img in enumerate(temp_list):
            ax_id = x_id_max*y_id+x_id+1
            if ax_id==1:
                ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True)
            else:
                ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True,sharex=config_list[0].ax,sharey=config_list[0].ax)
            config_list.append(q_config(fig=fig, ax=ax, ax_id=x_id_max*y_id+x_id+1, y_id=y_id, x_id=x_id,y_id_max=y_id_max,x_id_max=x_id_max,
                                        data=target_img.astype(float),
                                        cmin=caxis_list[y_id][x_id][0], cmax=caxis_list[y_id][x_id][1],
                                        cmap=cmap_list[y_id][x_id], cbar=disp_cbar_list[y_id][x_id],
                                        roi=patches.Rectangle(xy=(-11.5, -11.5), width=11, height=11, ec='red', fill=False),
                                        min=np.nanmin(target_img), max=np.nanmax(target_img),
                                        h=np.shape(target_img)[0], w=np.shape(target_img)[1], h_max=h_max, w_max=w_max, ))
            q_util_imshow_for_imageq(q_conf=config_list[-1])

    ############################### キーボードショートカット追加
    q_basic(config_list=config_list,init_xy_pos=[[-0.5, w_max-0.5],[h_max-0.5, -0.5]],yud_mode=1,
            keyboard_dict={'P':q_hotkey__png_save,
                           'D':q_hotkey__diff,'+':q_hotkey__add,'*':q_hotkey__mul,'/':q_hotkey__div,
                           'left':q_hotkey__climMANUAL_top_down, 'right':q_hotkey__climMANUAL_btm_up, 'up':q_hotkey__climMANUAL_slide_up, 'down':q_hotkey__climMANUAL_slide_down,
                           'alt+left':q_hotkey__climMANUAL_top_up, 'alt+right':q_hotkey__climMANUAL_btm_down,
                           'A':q_hotkey__climAUTO,'W':q_hotkey__climWHOLE,'E':q_hotkey__climEACH,'S':q_hotkey__climSYNC,
                           'r':q_hotkey__roiset,'alt+r':q_hotkey__roireset,'>':q_hotkey__roiwidthUP, '<':q_hotkey__roiheightUP,'alt+>':q_hotkey__roiwidthDOWN,'alt+<':q_hotkey__roiheightDOWN,
                           '$':q_hotkey__roistats,'m':q_hotkey__roipixval,'h':q_hotkey__roihist,
                           '-':q_hotkey__lineprofH,'=':q_hotkey__lineprofHmean,'i':q_hotkey__lineprofV,'I':q_hotkey__lineprofVmean,
                           'H':q_hotkey__hist,'~':q_hotkey__WaveformMonitor,'@':q_hotkey__posprint})


    # status barの表示変更
    def format_coord(x, y):
        int_x = int(x + 0.5)
        int_y = int(y + 0.5)
        return_str = 'x='+str(int_x)+', y='+str(int_y)+' |  '
        for k,conf in enumerate(config_list):
            if 0 <= int_x < conf.w and 0 <= int_y < conf.h:
                now_img_val = conf.data[int_y,int_x]
                if np.sum(np.isnan(now_img_val)) or np.sum(np.isinf(now_img_val)):
                    return_str = return_str+str(k)+': ###'+'  '
                else:
                    if np.ndim(now_img_val)==0:
                        return_str = return_str+str(k)+': '+'{:.3f}'.format(now_img_val)+'  '
                    else:
                        return_str = return_str+str(k)+': <'+'{:.3f}'.format(now_img_val[0])+', '+'{:.3f}'.format(now_img_val[1])+', '+'{:.3f}'.format(now_img_val[2])+'>  '
            else:
                return_str = return_str+str(k)+': ###'+'  '
        # 対処には、https://stackoverflow.com/questions/47082466/matplotlib-imshow-formatting-from-cursor-position
        # のような実装が必要になり、別の関数＋matplotlibの関数を叩くが必要ありめんどくさい
        return return_str

    for conf in config_list:
        conf.ax.format_coord = format_coord

    ############################### 表示
    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    fig.show()
    pass


# def imageq2(target_img_list, caxis_list, cmap_list, disp_cbar_list, view_mode='tile'):

import kutinawa as wa
# target_img_list = [[wa.imread('lena_bw.tiff'),wa.imread('lena.tiff'),wa.imread('fig_ready-made_10.png')]]
target_img_list = [[wa.imread('lena_bw.tiff')],
                    wa.imread('lena_bw.tiff')+20,
                    wa.imread('lena_bw.tiff')+wa.generate_gaussian_random_array((512,512))*30]
caxis_list = [[(64,196),(128,196),(64,196)]]
cmap_list = [['viridis','viridis','viridis']]
disp_cbar_list = [[True,True,True]]
# imageq(target_img_list,cmap_list=wa.cmap_out_range_color(over_color='red'))

# target_img_list = [[wa.imread('lena_bw.tiff'),wa.imread('lena.tiff'),wa.imread('fig_ready-made_10.png')]]
target_img_list = [[wa.imread('lena_bw.tiff')],
                   [wa.imread('lena_bw.tiff')+20,
                   wa.imread('lena_bw.tiff')+wa.generate_gaussian_random_array((512,512))*30]]
# histq(input_list=target_img_list,overlay=False,label_list=[['a'],['b','c']])
# plotq(input_list_x=target_img_list,overlay=True,label_list=[['a'],['b','c']],marker_list='o',linewidth_list=0)
# plotq(input_list_x=[[np.random.randn(10),np.array([])],[]],overlay=False,label_list=[['a',' '],[]],marker_list='o',linewidth_list=0)
# plotq(input_list_x=[[],[np.random.randn(100)]],overlay=False,label_list=[[],['b']],marker_list='o',linewidth_list=0,fig=fig)
#
# plotq(input_list_x=[[np.random.randn(10),np.random.randn(10),np.random.randn(10)],[]],overlay=False,label_list=[['a','b','c'],[]],marker_list='o',linewidth_list=0)

aaa = wa.imread('lena.tiff')
bbb = wa.imread('lena_bw.tiff')
# imageq([[aaa],[aaa,bbb],[aaa,bbb,aaa]],cmap_list=wa.cmap_out_range_color(over_color='red'))
# imageq([[aaa,bbb,aaa],[aaa],[aaa,bbb],],cmap_list=wa.cmap_out_range_color(over_color='red'))
#
# imageq([[bbb],[bbb,bbb]],cmap_list=wa.cmap_out_range_color(over_color='red'))

imageq([aaa,bbb],caxis_list = (0,255),cmap_list=wa.cmap_out_range_color(over_color='red'))

# wa.imageq([[aaa],[aaa,bbb]],cmap_list=wa.cmap_out_range_color(over_color='red'))



























