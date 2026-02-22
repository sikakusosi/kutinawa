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

from matplotlib.widgets import RectangleSelector

# https://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20      Font "Doh"

"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                                                                                                                                                                                                        
        CCCCCCCCCCCCC                           lllllll                                                     lllllll        iiii                                      tttt          
     CCC::::::::::::C                           l:::::l                                                     l:::::l       i::::i                                  ttt:::t          
   CC:::::::::::::::C                           l:::::l                                                     l:::::l        iiii                                   t:::::t          
  C:::::CCCCCCCC::::C                           l:::::l                                                     l:::::l                                               t:::::t          
 C:::::C       CCCCCC        ooooooooooo         l::::l         ooooooooooo        rrrrr   rrrrrrrrr         l::::l      iiiiiii          ssssssssss        ttttttt:::::ttttttt    
C:::::C                    oo:::::::::::oo       l::::l       oo:::::::::::oo      r::::rrr:::::::::r        l::::l      i:::::i        ss::::::::::s       t:::::::::::::::::t    
C:::::C                   o:::::::::::::::o      l::::l      o:::::::::::::::o     r:::::::::::::::::r       l::::l       i::::i      ss:::::::::::::s      t:::::::::::::::::t    
C:::::C                   o:::::ooooo:::::o      l::::l      o:::::ooooo:::::o     rr::::::rrrrr::::::r      l::::l       i::::i      s::::::ssss:::::s     tttttt:::::::tttttt    
C:::::C                   o::::o     o::::o      l::::l      o::::o     o::::o      r:::::r     r:::::r      l::::l       i::::i       s:::::s  ssssss            t:::::t          
C:::::C                   o::::o     o::::o      l::::l      o::::o     o::::o      r:::::r     rrrrrrr      l::::l       i::::i         s::::::s                 t:::::t          
C:::::C                   o::::o     o::::o      l::::l      o::::o     o::::o      r:::::r                  l::::l       i::::i            s::::::s              t:::::t          
 C:::::C       CCCCCC     o::::o     o::::o      l::::l      o::::o     o::::o      r:::::r                  l::::l       i::::i      ssssss   s:::::s            t:::::t    tttttt
  C:::::CCCCCCCC::::C     o:::::ooooo:::::o     l::::::l     o:::::ooooo:::::o      r:::::r                 l::::::l     i::::::i     s:::::ssss::::::s           t::::::tttt:::::t
   CC:::::::::::::::C     o:::::::::::::::o     l::::::l     o:::::::::::::::o      r:::::r                 l::::::l     i::::::i     s::::::::::::::s            tt::::::::::::::t
     CCC::::::::::::C      oo:::::::::::oo      l::::::l      oo:::::::::::oo       r:::::r                 l::::::l     i::::::i      s:::::::::::ss               tt:::::::::::tt
        CCCCCCCCCCCCC        ooooooooooo        llllllll        ooooooooooo         rrrrrrr                 llllllll     iiiiiiii       sssssssssss                   ttttttttttt                                                                                                                                                
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
matplotlib_colormap_list = [  # Perceptually Uniform Sequential
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    # Sequential
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    # Sequential (2)
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    # Cyclic
    'twilight', 'twilight_shifted', 'hsv',
    # Qualitative
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
    # Miscellaneous
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    'turbo', 'nipy_spectral', 'gist_ncar']

# kutinawa_color = ['#FF7171','#9DDD15','#57B8FF','#FF8D44','#51B883','#7096F8','#FFC700','#BB87FF','#2BC8E4','#F661F6',
#                   '#EC0000','#0066BE','#618E00','#C74700','#0031D8','#197A48','#A58000','#5C10BE','#008299','#AA00AA',
#                   '#FFDADA','#DCF0FF','#D0F5A2','#FFDFCA','#D9E6FF','#C2E5D1','#FFF0B3','#ECDDFF','#C8F8FF','#FFD0FF',]
kutinawa_color = ['#EC0000', '#1fec00', '#0014ec',
                  '#C74700', '#7ac402', '#008bc7',
                  '#ff5454', '#c0ff54', '#5471ff',
                  '#ff9b54', '#54ffb8', '#54fffc',
                  '#a60000', '#0ea600', '#2f00a6',
                  '#ff6200', '#00ffbb', '#00b7ff',
                  '#ff8a93', '#a9ff8a', '#8ac5ff',
                  '#ffb98a', '#8aff92', '#d48aff',
                  '#d9026d', '#83d902', '#7802d9',
                  '#ab0030', '#00ab7b', '#0083ab', ]

"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                                                                                                                                                                                                        
PPPPPPPPPPPPPPPPP                                   iiii                                       tttt          
P::::::::::::::::P                                 i::::i                                   ttt:::t          
P::::::PPPPPP:::::P                                 iiii                                    t:::::t          
PP:::::P     P:::::P                                                                        t:::::t          
  P::::P     P:::::P     rrrrr   rrrrrrrrr        iiiiiii      nnnn  nnnnnnnn         ttttttt:::::ttttttt    
  P::::P     P:::::P     r::::rrr:::::::::r       i:::::i      n:::nn::::::::nn       t:::::::::::::::::t    
  P::::PPPPPP:::::P      r:::::::::::::::::r       i::::i      n::::::::::::::nn      t:::::::::::::::::t    
  P:::::::::::::PP       rr::::::rrrrr::::::r      i::::i      nn:::::::::::::::n     tttttt:::::::tttttt    
  P::::PPPPPPPPP          r:::::r     r:::::r      i::::i        n:::::nnnn:::::n           t:::::t          
  P::::P                  r:::::r     rrrrrrr      i::::i        n::::n    n::::n           t:::::t          
  P::::P                  r:::::r                  i::::i        n::::n    n::::n           t:::::t          
  P::::P                  r:::::r                  i::::i        n::::n    n::::n           t:::::t    tttttt
PP::::::PP                r:::::r                 i::::::i       n::::n    n::::n           t::::::tttt:::::t
P::::::::P                r:::::r                 i::::::i       n::::n    n::::n           tt::::::::::::::t
P::::::::P                r:::::r                 i::::::i       n::::n    n::::n             tt:::::::::::tt
PPPPPPPPPP                rrrrrrr                 iiiiiiii       nnnnnn    nnnnnn               ttttttttttt                                                                                                                                                                                                                                                     
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def table_print(data, headers=[], table_mode='adapt', format_alignment='<', format_min_w=12,
                format_significant_digits=5):
    """
    ２次元listを表としてprintする
    :param data:                                       表にしたいデータ、現状2jigennlistのみ
    :param headers:                                    列ごとのヘッダー、１次元list
    :param table_mode:                'adapt', 'equal' 表の幅を列ごとに変える、すべて同じ
    :param format_alignment:          '<', '^', '>'    文字の右寄せ、中央、左寄せ
    :param format_min_w:                               テキストの最小幅、数値
    :param format_significant_digits:                  小数の有効桁
    :return:
    """
    temp_format = '{:' + format_alignment + str(format_min_w) + '.' + str(
        format_significant_digits) + 'f}'  # '{:<12.5f}'
    # temp_format_header = '{:' + format_alignment + str(format_min_w) + '}'

    # dataを必ず充填済みの２次元listにする
    table_hw = np.shape(data)

    # header不足があれば追加
    lh = len(headers)
    if len(headers) < table_hw[1]:
        for i in np.arange(table_hw[1] - lh):
            headers.append('Col ' + str(i + lh))

    # 表の横幅取得
    # max_width_list = np.array([[len(temp_format.format(x)) for x in y] for y in data])
    width_list = []
    for y in data + [headers]:
        width_list.append([])
        for i, x in enumerate(y):
            if type(x) is str:
                now_pf = '{:' + format_alignment + str(format_min_w) + '}'
            else:
                now_pf = '{:' + format_alignment + str(format_min_w) + '.' + str(format_significant_digits) + 'f}'
            width_list[-1].append(len(now_pf.format(x)))
    width_list = np.array(width_list)

    if table_mode == 'equal':
        temp = np.max(width_list)
        max_width_list = [temp for i in np.arange(table_hw[1])]
    elif table_mode == 'adapt':
        max_width_list = [np.max(width_list[:, i]) for i in np.arange(table_hw[1])]

    # print
    print(end='│')
    for i, hd in enumerate(headers):
        now_pf = '{:' + format_alignment + str(max_width_list[i]) + '}'
        print(now_pf.format(hd), end='│')

    print(end='\n╞')
    for i, hd in enumerate(headers[:-1]):
        now_pf = '{:' + format_alignment + str(max_width_list[i]) + '}'
        print(now_pf.format('═' * max_width_list[i]), end='╪')
    now_pf = '{:' + format_alignment + str(max_width_list[-1]) + '}'
    print(now_pf.format('═' * max_width_list[-1]), end='╡')

    for y in data:
        print(end='\n│')
        for i, x in enumerate(y):
            if type(x) is str:
                now_pf = '{:' + format_alignment + str(max_width_list[i]) + '}'
            else:
                now_pf = '{:' + format_alignment + str(max_width_list[i]) + '.' + str(format_significant_digits) + 'f}'
            print(now_pf.format(x), end='│')

    print("")
    pass


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                      
                                tttt                 iiii       lllllll 
                             ttt:::t                i::::i      l:::::l 
                             t:::::t                 iiii       l:::::l 
                             t:::::t                            l:::::l 
uuuuuu    uuuuuu       ttttttt:::::ttttttt         iiiiiii       l::::l 
u::::u    u::::u       t:::::::::::::::::t         i:::::i       l::::l 
u::::u    u::::u       t:::::::::::::::::t          i::::i       l::::l 
u::::u    u::::u       tttttt:::::::tttttt          i::::i       l::::l 
u::::u    u::::u             t:::::t                i::::i       l::::l 
u::::u    u::::u             t:::::t                i::::i       l::::l 
u::::u    u::::u             t:::::t                i::::i       l::::l 
u:::::uuuu:::::u             t:::::t    tttttt      i::::i       l::::l 
u:::::::::::::::uu           t::::::tttt:::::t     i::::::i     l::::::l
 u:::::::::::::::u           tt::::::::::::::t     i::::::i     l::::::l
  uu::::::::uu:::u             tt:::::::::::tt     i::::::i     l::::::l
    uuuuuuuu  uuuu               ttttttttttt       iiiiiiii     llllllll                                       
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def clim(cmin, cmax):
    for im in plt.gca().get_images():
        im.set_clim(cmin, cmax)
    plt.gca().figure.canvas.draw()
    pass


def xlim(xmin, xmax):
    plt.gca().set_xlim(xmin, xmax)
    plt.gca().figure.canvas.draw()
    pass


def ylim(ymin, ymax):
    plt.gca().set_ylim(ymin, ymax)
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


def tolist_only1axis(tgt_array, axis):
    axis_temp = np.arange(np.ndim(tgt_array))
    return [i for i in np.transpose(tgt_array, tuple([axis] + (axis_temp[axis_temp != axis]).tolist()))]


def list1toSQ2(tgt_list):
    sub_x = np.ceil(np.sqrt(len(tgt_list))).astype(int)
    sub_y = np.ceil(len(tgt_list) / sub_x).astype(int)
    idx_l = np.arange(0, len(tgt_list), sub_x).tolist() + [len(tgt_list)]
    return [tgt_list[idx_l[h]:idx_l[h + 1]] for h in np.arange(sub_y)]


def imq_inEASY(tgt_imgs, axis):
    if isinstance(tgt_imgs, np.ndarray):
        out_imgs = list1toSQ2(tolist_only1axis(tgt_imgs, axis))
    else:
        out_imgs = list1toSQ2(tgt_imgs)
    return out_imgs


def cmap_out_range_color(cmap_name='viridis', over_color='white', under_color='black', bad_color='red'):
    cm = pylab.cm.get_cmap(cmap_name)
    colors = cm.colors
    out_cmap = ListedColormap(colors, name='custom', N=255)
    out_cmap.set_over(over_color)
    out_cmap.set_under(under_color)
    out_cmap.set_bad(bad_color)
    return out_cmap


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HHHHHHHHH     HHHHHHHHH          OOOOOOOOO          TTTTTTTTTTTTTTTTTTTTTTT     KKKKKKKKK    KKKKKKK     EEEEEEEEEEEEEEEEEEEEEE     YYYYYYY       YYYYYYY
H:::::::H     H:::::::H        OO:::::::::OO        T:::::::::::::::::::::T     K:::::::K    K:::::K     E::::::::::::::::::::E     Y:::::Y       Y:::::Y
H:::::::H     H:::::::H      OO:::::::::::::OO      T:::::::::::::::::::::T     K:::::::K    K:::::K     E::::::::::::::::::::E     Y:::::Y       Y:::::Y
HH::::::H     H::::::HH     O:::::::OOO:::::::O     T:::::TT:::::::TT:::::T     K:::::::K   K::::::K     EE::::::EEEEEEEEE::::E     Y::::::Y     Y::::::Y
  H:::::H     H:::::H       O::::::O   O::::::O     TTTTTT  T:::::T  TTTTTT     KK::::::K  K:::::KKK       E:::::E       EEEEEE     YYY:::::Y   Y:::::YYY
  H:::::H     H:::::H       O:::::O     O:::::O             T:::::T               K:::::K K:::::K          E:::::E                     Y:::::Y Y:::::Y   
  H::::::HHHHH::::::H       O:::::O     O:::::O             T:::::T               K::::::K:::::K           E::::::EEEEEEEEEE            Y:::::Y:::::Y    
  H:::::::::::::::::H       O:::::O     O:::::O             T:::::T               K:::::::::::K            E:::::::::::::::E             Y:::::::::Y     
  H:::::::::::::::::H       O:::::O     O:::::O             T:::::T               K:::::::::::K            E:::::::::::::::E              Y:::::::Y      
  H::::::HHHHH::::::H       O:::::O     O:::::O             T:::::T               K::::::K:::::K           E::::::EEEEEEEEEE               Y:::::Y       
  H:::::H     H:::::H       O:::::O     O:::::O             T:::::T               K:::::K K:::::K          E:::::E                         Y:::::Y       
  H:::::H     H:::::H       O::::::O   O::::::O             T:::::T             KK::::::K  K:::::KKK       E:::::E       EEEEEE            Y:::::Y       
HH::::::H     H::::::HH     O:::::::OOO:::::::O           TT:::::::TT           K:::::::K   K::::::K     EE::::::EEEEEEEE:::::E            Y:::::Y       
H:::::::H     H:::::::H      OO:::::::::::::OO            T:::::::::T           K:::::::K    K:::::K     E::::::::::::::::::::E         YYYY:::::YYYY    
H:::::::H     H:::::::H        OO:::::::::OO              T:::::::::T           K:::::::K    K:::::K     E::::::::::::::::::::E         Y:::::::::::Y    
HHHHHHHHH     HHHHHHHHH          OOOOOOOOO                TTTTTTTTTTT           KKKKKKKKK    KKKKKKK     EEEEEEEEEEEEEEEEEEEEEE         YYYYYYYYYYYYY                                                                                                                                                             
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def q_hotkey__dummy(fig, event, state):
    pass


def q_hotkey__png_save(fig, event, state):
    fig.savefig(datetime.datetime.now().strftime('q-%Y_%m_%d_%H_%M_%S') + '.png', bbox_inches='tight')
    pass


def q_hotkey__reset(fig, event, state):
    for axe in state.axes_list:
        axe.set_xlim(state.initial_lims[0][0], state.initial_lims[0][1])
        axe.set_ylim(state.initial_lims[1][0], state.initial_lims[1][1])
    pass


def q_hotkey__roiset(fig, event, state):
    if state.mouse_mode == 'ROI':
        pos = (np.array(state.rect_list[state.current_axes_index].extents) + 0.5).astype(int)
        if pos[3] - pos[2] == 0 and pos[1] - pos[0] == 0:
            # [roi[event.key].set(xy=(-0.5, -0.5), width=0, height=0) for roi in state.roi_patch_list]
            for roi in state.roi_patch_list:
                roi[event.key].set(xy=(-0.5, -0.5), width=0, height=0)
            # [roi_text[event.key].set(x=-0.5, y=-0.5, alpha=0) for roi_text in state.roi_text_list]
            for roi_text in state.roi_text_list:
                roi_text[event.key].set(x=-0.5, y=-0.5, alpha=0)
        else:
            pos = pos + np.array([-0.5, 0.5, -0.5, 0.5])
            # [roi[event.key].set(xy=(pos[0], pos[2]), height=pos[3] - pos[2], width=pos[1] - pos[0]) for roi in state.roi_patch_list]
            # [roi_text[event.key].set(x=pos[0], y=pos[2], alpha=1) for roi_text in state.roi_text_list]
            for roi in state.roi_patch_list:
                roi[event.key].set(xy=(pos[0], pos[2]), height=pos[3] - pos[2], width=pos[1] - pos[0])
            for roi_text in state.roi_text_list:
                roi_text[event.key].set(x=pos[0], y=pos[2], alpha=1)
    pass


def q_hotkey__mousemode_Normal(fig, event, state):
    state.mouse_mode = 'Normal'


def q_hotkey__mousemode_ROI(fig, event, state):
    state.mouse_mode = 'ROI'


def q_hotkey__roistats(fig, event, state):
    active_roi_keys = [rk for rk in state.roi_patch_list[0].keys() if state.roi_patch_list[0][rk].get_height() > 0]

    stats_list = np.zeros((len(state.axes_list), len(active_roi_keys))).tolist()
    for axe_num in np.arange(len(state.axes_list)):
        axes_img = state.axes_list[axe_num].images[0].get_array().data
        for i, rk in enumerate(active_roi_keys):
            tgt_roi = state.roi_patch_list[axe_num][rk]
            roi_x, roi_y = np.array(tgt_roi.get_xy()) + 0.5
            roi_xx, roi_yy = roi_x + tgt_roi.get_width(), roi_y + tgt_roi.get_height()
            roi_y, roi_yy, roi_x, roi_xx = int(roi_y), int(roi_yy), int(roi_x), int(roi_xx)
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y, 0, hw[0]), np.clip(roi_yy, 0, hw[0]), np.clip(roi_x, 0,
                                                                                                        hw[1]), np.clip(
                roi_xx, 0, hw[1])
            roi_img = axes_img[roi_y:roi_yy, roi_x:roi_xx, ...]

            stats_list[axe_num][i] = {'roi_x': roi_x, 'roi_xx': roi_xx,
                                      'roi_y': roi_y, 'roi_yy': roi_yy,
                                      'mean': np.nanmean(roi_img, axis=(0, 1)),
                                      'std': np.nanstd(roi_img, axis=(0, 1)),
                                      'min': np.nanmin(roi_img, axis=(0, 1)),
                                      'max': np.nanmax(roi_img, axis=(0, 1)),
                                      'med': np.nanmedian(roi_img, axis=(0, 1)), }

    print_header = [''] + ['img' + str(axe_num) for axe_num in np.arange(len(state.axes_list))]
    print_data = []
    for i, rk in enumerate(active_roi_keys):
        print('ROI <' + rk + '>',
              'pos=[' + str(stats_list[0][i]['roi_y']) + ':' + str(stats_list[0][i]['roi_yy']) + ', ' + str(
                  stats_list[0][i]['roi_x']) + ':' + str(stats_list[0][i]['roi_xx']) + ']')
        print_data = [['mean'],
                      ['std'],
                      ['min'],
                      ['max'],
                      ['med'], ]
        for axe_num in np.arange(len(state.axes_list)):
            print_data[0].append(stats_list[axe_num][i]['mean'])
            print_data[1].append(stats_list[axe_num][i]['std'])
            print_data[2].append(stats_list[axe_num][i]['min'])
            print_data[3].append(stats_list[axe_num][i]['max'])
            print_data[4].append(stats_list[axe_num][i]['med'])

        table_print(data=print_data, headers=print_header)

    pass


# clim
#
def q_hotkey_util__climMANUAL(fig, plusminus, gain):
    c_axe = fig.gca()
    im = c_axe.images[0]  # 最初のimshow
    caxis_min, caxis_max = im.get_clim()
    diff = caxis_max - caxis_min
    min_change = plusminus[0] * diff * gain
    max_change = plusminus[1] * diff * gain
    im.set_clim(caxis_min + min_change, caxis_max + max_change)
    pass


def q_hotkey__climMANUAL_top_down(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[0, -1], gain=0.025)


def q_hotkey__climMANUAL_btm_up(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[1, 0], gain=0.025)


def q_hotkey__climMANUAL_top_up(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[0, 1], gain=0.025)


def q_hotkey__climMANUAL_btm_down(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[-1, 0], gain=0.025)


def q_hotkey__climMANUAL_slide_up(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[1, 1], gain=0.025)


def q_hotkey__climMANUAL_slide_down(fig, event, state):
    q_hotkey_util__climMANUAL(fig, plusminus=[-1, -1], gain=0.025)


def q_hotkey__climAUTO(fig, event, state):
    c_axe = fig.gca()
    now_lim_x = np.clip((np.array(c_axe.get_xlim()) + 0.5).astype(int), 0, None)
    now_lim_y = np.clip((np.array(c_axe.get_ylim()) + 0.5).astype(int), 0, None)
    temp = c_axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
    c_axe.images[0].set_clim(
        (np.nanmin(temp[(temp != -np.inf) * (temp != np.inf)]), np.nanmax(temp[(temp != -np.inf) * (temp != np.inf)])))
    pass


def q_hotkey__climWHOLE(fig, event, state):
    now_lim_x = np.clip((np.array(state.axes_list[0].get_xlim()) + 0.5).astype(int), 0, None)
    now_lim_y = np.clip((np.array(state.axes_list[0].get_ylim()) + 0.5).astype(int), 0, None)
    whole_max = np.nanmax(np.array(
        [np.nanmax(axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for axe in
         state.axes_list]))
    whole_min = np.nanmin(np.array(
        [np.nanmin(axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for axe in
         state.axes_list]))
    for axe in state.axes_list:
        axe.images[0].set_clim(whole_min, whole_max)
    pass


def q_hotkey__climEACH(fig, event, state):
    now_lim_x = np.clip((np.array(state.axes_list[0].get_xlim()) + 0.5).astype(int), 0, None)
    now_lim_y = np.clip((np.array(state.axes_list[0].get_ylim()) + 0.5).astype(int), 0, None)
    for axe in state.axes_list:
        temp = axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
        axe.images[0].set_clim((np.nanmin(temp[(temp != -np.inf) * (temp != np.inf)]),
                                np.nanmax(temp[(temp != -np.inf) * (temp != np.inf)])))
    pass


def q_hotkey__climSYNC(fig, event, state):
    c_axe = fig.gca()
    sync_clim = c_axe.images[0].get_clim()
    for axe in state.axes_list:
        axe.images[0].set_clim(sync_clim[0], sync_clim[1])
    pass


def q_hotkey_util__climSYNCudlr(fig, event, state, direction):
    c_axe = fig.gca()
    sync_clim = c_axe.images[0].get_clim()
    sps = c_axe.get_subplotspec()
    subplot_x, subplot_y = sps.colspan[0], sps.rowspan[0]

    x_adj = {'up': 0, 'down': 0, 'left': -1, 'right': 1}
    y_adj = {'up': -1, 'down': 1, 'left': 0, 'right': 0}
    sync_tgt_x = subplot_x + x_adj[direction]
    sync_tgt_y = subplot_y + y_adj[direction]

    for axe in state.axes_list:
        sps = axe.get_subplotspec()
        if sps.colspan[0] == sync_tgt_x and sps.rowspan[0] == sync_tgt_y:
            axe.images[0].set_clim(sync_clim[0], sync_clim[1])
    pass


def q_hotkey__climSYNCup(fig, event, state):
    q_hotkey_util__climSYNCudlr(fig, event, state, 'up')
    pass


def q_hotkey__climSYNCdown(fig, event, state):
    q_hotkey_util__climSYNCudlr(fig, event, state, 'down')
    pass


def q_hotkey__climSYNCleft(fig, event, state):
    q_hotkey_util__climSYNCudlr(fig, event, state, 'left')
    pass


def q_hotkey__climSYNCright(fig, event, state):
    q_hotkey_util__climSYNCudlr(fig, event, state, 'right')
    pass


# ############################################################################ lineprof
def q_hotkey_util__lineprof(fig, event, state, mode):
    img_xlim = state.axes_list[0].get_xlim()
    img_xlim = [int(img_xlim[0] + 0.5), int(img_xlim[1] + 0.5)]
    img_ylim = state.axes_list[0].get_ylim()
    img_ylim = [int(img_ylim[0] + 0.5), int(img_ylim[1] + 0.5)]

    if mode == 'H':
        mouse_y = int(event.ydata + 0.5)
        line_img_xpos = img_xlim
        line_img_ypos = [mouse_y + 1, mouse_y]
        line_plot_idx = 0
        ax_line_xy1 = (img_xlim[0], mouse_y)
        ax_line_xy2 = (img_xlim[1], mouse_y)
        ax_line_xy3 = (img_xlim[0], mouse_y)
        ax_line_xy4 = (img_xlim[1], mouse_y)
    else:  # elif mode=='V':
        mouse_x = int(event.xdata + 0.5)
        line_img_xpos = [mouse_x, mouse_x + 1]
        line_img_ypos = img_ylim
        line_plot_idx = 1
        ax_line_xy1 = (mouse_x, img_ylim[0])
        ax_line_xy2 = (mouse_x, img_ylim[1])
        ax_line_xy3 = (mouse_x, img_ylim[0])
        ax_line_xy4 = (mouse_x, img_ylim[1])

    ana_fig = plt.figure()
    plot_sub_ax = ana_fig.add_subplot(2, len(state.axes_list), (1, len(state.axes_list)), picker=True)

    for cn, axe in enumerate(state.axes_list):
        axes_img = axe.images[0].get_array().data
        hw = np.shape(axes_img)
        x_pos = [np.clip(line_img_xpos[0], 0, hw[1]), np.clip(line_img_xpos[1], 0, hw[1])]
        y_pos = [np.clip(line_img_ypos[1], 0, hw[0]), np.clip(line_img_ypos[0], 0, hw[0])]
        line_img, pos = axes_img[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1]], [x_pos, y_pos]

        ####################### plot
        m0 = int(np.mod(cn, 4) > 0)
        m1 = int(np.mod(cn, 4) > 1)
        m2 = int(np.mod(cn, 4) > 2)
        linestyle = (0, (5, m0, m0, m0, m1, m1, m2, m2))

        if np.ndim(line_img) == 2:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                             np.squeeze(line_img),
                             label=cn)
        elif np.ndim(line_img) == 3:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                             np.squeeze(line_img[:, :, 0]),
                             label=str(cn) + '(0ch)',
                             color=(0.75, np.clip(cn / len(state.axes_list) - 0.5, 0, 1),
                                    np.clip(-cn / len(state.axes_list) + 0.5, 0, 1)),
                             linestyle=linestyle)
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                             np.squeeze(line_img[:, :, 1]),
                             label=str(cn) + '(1ch)',
                             color=(np.clip(cn / len(state.axes_list) - 0.5, 0, 1), 0.75,
                                    np.clip(-cn / len(state.axes_list) + 0.5, 0, 1)),
                             linestyle=linestyle)
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                             np.squeeze(line_img[:, :, 2]),
                             label=str(cn) + '(2ch)',
                             color=(np.clip(cn / len(state.axes_list) - 0.5, 0, 1),
                                    np.clip(-cn / len(state.axes_list) + 0.5, 0, 1), 0.75),
                             linestyle=linestyle)

        ####################### image
        pos2 = [[np.clip(img_xlim[0], 0, hw[1]), np.clip(img_xlim[1], 0, hw[1])],
                [np.clip(img_ylim[1], 0, hw[0]), np.clip(img_ylim[0], 0, hw[0])]]
        part_img = axes_img[pos2[1][0]:pos2[1][1], pos2[0][0]:pos2[0][1]]
        img_sub_ax = ana_fig.add_subplot(2, len(state.axes_list), len(state.axes_list) + cn + 1, picker=True)
        img_sub_ax.axline(xy1=ax_line_xy1, xy2=ax_line_xy2, color='pink')
        img_sub_ax.axline(xy1=ax_line_xy3, xy2=ax_line_xy4, color='pink')
        img_sub_im = img_sub_ax.imshow(part_img, interpolation='nearest', cmap=axe.images[0].get_cmap(),interpolation_stage='data',
                                       extent=[pos2[0][0] - 0.5, pos2[0][1] - 0.5, pos2[1][1] - 0.5, pos2[1][0] - 0.5],
                                       aspect='equal')
        img_sub_im.set_clim(axe.images[0].get_clim())

    plot_sub_ax.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={"weight": "bold", "size": "large"})
    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
    ana_fig.show()
    pass


def q_hotkey__lineprofH(fig, event, state):
    if event.inaxes:
        q_hotkey_util__lineprof(fig, event, state, mode='H')


def q_hotkey__lineprofV(fig, event, state):
    if event.inaxes:
        q_hotkey_util__lineprof(fig, event, state, mode='V')


def q_hotkey_util__meanlineprof(fig, event, state, mode):
    active_roi_keys = [rk for rk in state.roi_patch_list[0].keys() if state.roi_patch_list[0][rk].get_height() > 0]

    img_xlim = state.axes_list[0].get_xlim()
    img_xlim = [int(img_xlim[0] + 0.5), int(img_xlim[1] + 0.5)]
    img_ylim = state.axes_list[0].get_ylim()
    img_ylim = [int(img_ylim[0] + 0.5), int(img_ylim[1] + 0.5)]

    for i, rk in enumerate(active_roi_keys):
        tgt_roi = state.roi_patch_list[0][rk]
        roi_x, roi_y = np.array(tgt_roi.get_xy()) + 0.5
        roi_xx, roi_yy = roi_x + tgt_roi.get_width(), roi_y + tgt_roi.get_height()
        roi_y, roi_yy, roi_x, roi_xx = int(roi_y), int(roi_yy), int(roi_x), int(roi_xx)

        if mode == 'roiH':
            line_img_xpos = img_xlim
            line_img_ypos = [roi_yy, roi_y]
            line_plot_idx = 0
            ax_line_xy1 = (img_xlim[0], roi_y - 0.5)
            ax_line_xy2 = (img_xlim[1], roi_y - 0.5)
            ax_line_xy3 = (img_xlim[0], roi_yy - 0.5)
            ax_line_xy4 = (img_xlim[1], roi_yy - 0.5)
        else:  # elif mode == 'roiV':
            line_img_xpos = [roi_x, roi_xx]
            line_img_ypos = img_ylim
            line_plot_idx = 1
            ax_line_xy1 = (roi_x - 0.5, img_ylim[0])
            ax_line_xy2 = (roi_x - 0.5, img_ylim[1])
            ax_line_xy3 = (roi_xx - 0.5, img_ylim[0])
            ax_line_xy4 = (roi_xx - 0.5, img_ylim[1])

        ana_fig = plt.figure()
        plot_sub_ax = ana_fig.add_subplot(2, len(state.axes_list), (1, len(state.axes_list)), picker=True)

        for cn, axe in enumerate(state.axes_list):
            axes_img = axe.images[0].get_array().data
            hw = np.shape(axes_img)
            x_pos = [np.clip(line_img_xpos[0], 0, hw[1]), np.clip(line_img_xpos[1], 0, hw[1])]
            y_pos = [np.clip(line_img_ypos[1], 0, hw[0]), np.clip(line_img_ypos[0], 0, hw[0])]
            line_img, pos = axes_img[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1]], [x_pos, y_pos]

            ####################### plot
            m0 = int(np.mod(cn, 4) > 0)
            m1 = int(np.mod(cn, 4) > 1)
            m2 = int(np.mod(cn, 4) > 2)
            linestyle = (0, (5, m0, m0, m0, m1, m1, m2, m2))

            if np.ndim(line_img) == 2:
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img, axis=line_plot_idx)),
                                 label=cn)
            elif np.ndim(line_img) == 3:
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 0], axis=line_plot_idx)),
                                 label=str(cn) + '(0ch)',
                                 color=(0.75, np.clip(cn / len(state.axes_list) - 0.5, 0, 1),
                                        np.clip(-cn / len(state.axes_list) + 0.5, 0, 1)),
                                 linestyle=linestyle)
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 1], axis=line_plot_idx)),
                                 label=str(cn) + '(1ch)',
                                 color=(np.clip(cn / len(state.axes_list) - 0.5, 0, 1), 0.75,
                                        np.clip(-cn / len(state.axes_list) + 0.5, 0, 1)),
                                 linestyle=linestyle)
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 2], axis=line_plot_idx)),
                                 label=str(cn) + '(2ch)',
                                 color=(np.clip(cn / len(state.axes_list) - 0.5, 0, 1),
                                        np.clip(-cn / len(state.axes_list) + 0.5, 0, 1), 0.75),
                                 linestyle=linestyle)

            ####################### image
            pos2 = [[np.clip(img_xlim[0], 0, hw[1]), np.clip(img_xlim[1], 0, hw[1])],
                    [np.clip(img_ylim[1], 0, hw[0]), np.clip(img_ylim[0], 0, hw[0])]]
            part_img = axes_img[pos2[1][0]:pos2[1][1], pos2[0][0]:pos2[0][1]]
            img_sub_ax = ana_fig.add_subplot(2, len(state.axes_list), len(state.axes_list) + cn + 1, picker=True)
            img_sub_ax.axline(xy1=ax_line_xy1, xy2=ax_line_xy2, color='pink')
            img_sub_ax.axline(xy1=ax_line_xy3, xy2=ax_line_xy4, color='pink')
            img_sub_im = img_sub_ax.imshow(part_img, interpolation='nearest', cmap=axe.images[0].get_cmap(),interpolation_stage='data',
                                           extent=[pos2[0][0] - 0.5, pos2[0][1] - 0.5, pos2[1][1] - 0.5,pos2[1][0] - 0.5],
                                           aspect='equal')
            img_sub_im.set_clim(axe.images[0].get_clim())

        plot_sub_ax.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={"weight": "bold", "size": "large"})
        ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
        ana_fig.show()


def q_hotkey__lineprofHmean(fig, event, state):
    q_hotkey_util__meanlineprof(fig, event, state, mode='roiH')


def q_hotkey__lineprofVmean(fig, event, state):
    q_hotkey_util__meanlineprof(fig, event, state, mode='roiV')


def q_hotkey__ROIhist(fig, event, state):
    active_roi_keys = [rk for rk in state.roi_patch_list[0].keys() if state.roi_patch_list[0][rk].get_height() > 0]

    subplot_h, subplot_w = state.axes_list[0].get_gridspec().nrows, state.axes_list[0].get_gridspec().ncols
    for i, rk in enumerate(active_roi_keys):
        tgt_roi = state.roi_patch_list[0][rk]
        roi_x, roi_y = np.array(tgt_roi.get_xy()) + 0.5
        roi_xx, roi_yy = roi_x + tgt_roi.get_width(), roi_y + tgt_roi.get_height()
        roi_y, roi_yy, roi_x, roi_xx = int(roi_y), int(roi_yy), int(roi_x), int(roi_xx)

        temp_input = [[] for i in range(subplot_h)]
        temp_inter = [[] for i in range(subplot_h)]
        temp_c = [[] for i in range(subplot_h)]
        temp_ec = [[] for i in range(subplot_h)]
        for j, axe in enumerate(state.axes_list):
            axes_img = axe.images[0].get_array().data
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y, 0, hw[0]), np.clip(roi_yy, 0, hw[0]), np.clip(roi_x, 0,
                                                                                                        hw[1]), np.clip(
                roi_xx, 0, hw[1])
            part_img = axes_img[roi_y:roi_yy, roi_x:roi_xx]

            sps = axe.get_subplotspec()
            subplot_x = sps.colspan[0]
            subplot_y = sps.rowspan[0]

            if np.ndim(axes_img) == 2:
                temp_input[subplot_y].append(part_img)
                temp_c[subplot_y].append(kutinawa_color[i])
                temp_ec[subplot_y].append(kutinawa_color[i])
                # temp_ec[subplot_y].append('black')

                if np.all(axes_img.astype(int).astype(float) == axes_img):
                    temp_inter[subplot_y].append(1)
                else:
                    temp_inter[subplot_y].append((axe.images[0].get_clim()[1] - axe.images[0].get_clim()[0]) / 256)

            elif np.ndim(axes_img) == 3:
                tc = ['#FF7171', '#9DDD15', '#57B8FF', ]
                for ch in np.arange(np.shape(axes_img)[2]):
                    temp_input[subplot_y].append(part_img[:, :, ch])
                    temp_inter[subplot_y].append((np.nanmax(axes_img) - np.nanmin(axes_img)) / 256)
                    temp_c[subplot_y].append(kutinawa_color[i])
                    temp_ec[subplot_y].append(tc[ch])

        histq(tgt_data_list=temp_input,
              interval_list=temp_inter,
              label_list=None,
              alpha_list=1,
              edgecolor_list=temp_ec,
              color_list=temp_c,
              histtype='bar',
              overlay=False)
    pass


def q_hotkey__layer(fig, event, state):
    temp_img = state.axes_list[min([int(event.key[1:]),len(state.axes_list)-1])].images[0].get_array()
    state.axes_list[0].images[0].set_data(temp_img)
    state.axes_list[0].images[0].set_extent((0, np.shape(temp_img)[1], np.shape(temp_img)[0], 0))
    pass

def q_hotkey__switch_cmap_linear_gamma(fig, event, state):
    tag = "_gamma"
    gamma = 1/2.1

    c_axe = fig.gca()
    im = c_axe.images[0]

    cmap = im.get_cmap()
    name = cmap.name

    if name.endswith(tag):
        print(name)
        print(name[:-len(tag)])
        orig_name = name[:-len(tag)]
        orig_cmap = matplotlib.colormaps.get_cmap(orig_name)
        im.set_cmap(orig_cmap)
        c_axe.set_title('')
        pass
    else:
        N = cmap.N
        x = np.linspace(0, 1, N)
        x_gamma = x ** gamma
        colors = cmap(x_gamma)
        gamma_cmap = ListedColormap(colors, name=name + tag)
        print(gamma_cmap.name)
        im.set_cmap(gamma_cmap)
        print(im.get_cmap())
        c_axe.set_title('2.1gamma')
        pass



"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                                                                                                                       
                                                                                         dddddddd                 dddddddd                                             
                                                                                         d::::::d                 d::::::d                                             
                                                                                         d::::::d                 d::::::d                                             
                                                                                         d::::::d                 d::::::d                                             
                                                                                         d:::::d                  d:::::d                                              
   qqqqqqqqq   qqqqq                                    aaaaaaaaaaaaa            ddddddddd:::::d          ddddddddd:::::d         ooooooooooo        nnnn  nnnnnnnn    
  q:::::::::qqq::::q                                    a::::::::::::a         dd::::::::::::::d        dd::::::::::::::d       oo:::::::::::oo      n:::nn::::::::nn  
 q:::::::::::::::::q                                    aaaaaaaaa:::::a       d::::::::::::::::d       d::::::::::::::::d      o:::::::::::::::o     n::::::::::::::nn 
q::::::qqqqq::::::qq                                             a::::a      d:::::::ddddd:::::d      d:::::::ddddd:::::d      o:::::ooooo:::::o     nn:::::::::::::::n
q:::::q     q:::::q                                       aaaaaaa:::::a      d::::::d    d:::::d      d::::::d    d:::::d      o::::o     o::::o       n:::::nnnn:::::n
q:::::q     q:::::q                                     aa::::::::::::a      d:::::d     d:::::d      d:::::d     d:::::d      o::::o     o::::o       n::::n    n::::n
q:::::q     q:::::q                                    a::::aaaa::::::a      d:::::d     d:::::d      d:::::d     d:::::d      o::::o     o::::o       n::::n    n::::n
q::::::q    q:::::q                                   a::::a    a:::::a      d:::::d     d:::::d      d:::::d     d:::::d      o::::o     o::::o       n::::n    n::::n
q:::::::qqqqq:::::q                                   a::::a    a:::::a      d::::::ddddd::::::dd     d::::::ddddd::::::dd     o:::::ooooo:::::o       n::::n    n::::n
 q::::::::::::::::q                                   a:::::aaaa::::::a       d:::::::::::::::::d      d:::::::::::::::::d     o:::::::::::::::o       n::::n    n::::n
  qq::::::::::::::q                                    a::::::::::aa:::a       d:::::::::ddd::::d       d:::::::::ddd::::d      oo:::::::::::oo        n::::n    n::::n
    qqqqqqqq::::::q                                     aaaaaaaaaa  aaaa        ddddddddd   ddddd        ddddddddd   ddddd        ooooooooooo          nnnnnn    nnnnnn
            q:::::q      ________________________                                                                                                                      
            q:::::q      _::::::::::::::::::::::_                                                                                                                      
           q:::::::q     ________________________                                                                                                                      
           q:::::::q                                                                                                                                                   
           q:::::::q                                                                                                                                                   
            qqqqqqqqq                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class QAddonState:
    """q_addon 全体で共有する状態"""

    def __init__(self):
        self.axes_list = []
        self.mouse_mode = "Normal"
        self.initial_lims = None
        self.current_axes_index = 0
        self.rect_list = []
        self.roi_patch_list = []
        self.roi_text_list = []
        self.key_press_xy = [0, 0]
        self.key_press_xydata = [0, 0]
        self.btn_press_xy = [0, 0]
        self.btn_press_xydata = [0, 0]
        self.btn_press_clim = [0, 0]
        self.shift_flag = False
        self.inaxes_flag = False
        self.cbar_list = []
        self.incbar_flag = False


def q_addon(fig, axes_list=None, keyboard_dict=None, imageq=False, cbar_list=None):
    """
    Matplotlib Figure に対してマウス・キーボード操作を拡張する
    """

    plt.interactive(False)
    keyboard_dict = keyboard_dict or {}
    state = QAddonState()
    state.axes_list = axes_list or []
    state.cbar_list = cbar_list or []

    # 軸範囲を取得、プロット領域の位置初期化
    xlims = np.array([axe.get_xlim() for axe in axes_list])
    ylims = np.array([axe.get_ylim() for axe in axes_list])
    y_inverted = int(ylims[0, 0] > ylims[0, 1])
    if y_inverted == 1:
        state.initial_lims = [[np.min(xlims), np.max(xlims)],
                              [np.max(ylims), np.min(ylims)]]
    else:
        state.initial_lims = [[np.min(xlims), np.max(xlims)],
                              [np.min(ylims), np.max(ylims)]]
    for axe in state.axes_list:
        axe.set_xlim(*state.initial_lims[0])
        axe.set_ylim(*state.initial_lims[1])

    # _initialize_rectangle_selectors(fig, axes_list, state)
    # 拡縮、ROIの設定時に使用する、マウスボタン押下中にしか表示されない四角
    # 1つのsubplotに1つずつ存在する
    def dummy_callback(eclick, erelease):
        pass

    for axe in state.axes_list:
        selector = RectangleSelector(axe,
                                     dummy_callback,
                                     useblit=True,
                                     button=[1],  # disable right & middle button
                                     minspanx=5,
                                     minspany=5,
                                     spancoords='pixels',
                                     interactive=True,
                                     state_modifier_keys={"square": 'ctrl'},
                                     props=dict(
                                         facecolor='pink',
                                         edgecolor='white',
                                         alpha=1,
                                         fill=False)
                                     )
        selector.set_visible(False)
        state.rect_list.append(selector)

    # _initialize_roi_objects(axes_list, state)
    # ROIを使用する場合の初期化
    # 各subplotに0~9,ctrl+0~9,alt+0~9分の30個のROIを見えない状態で描画
    # 全てにROI設置用関数を紐付け
    if imageq:
        cbar_index = {ax: i for i, ax in enumerate(cbar_list)}
        base_num_keys = [str(i + 1) for i in range(9)] + ['0']
        roi_keys = base_num_keys + [f"ctrl+{k}" for k in base_num_keys] + [f"alt+{k}" for k in base_num_keys]

        for axe in state.axes_list:
            roi_dict = {}
            roi_text_dict = {}

            for i_r, roi_key in enumerate(roi_keys):
                rect = patches.Rectangle(xy=(-0.5, -0.5), width=0, height=0, ec=kutinawa_color[i_r], fill=False)
                text = axe.text(-0.5, -0.5, s=roi_key, c=kutinawa_color[i_r], ha='right', va='top', alpha=0,
                                weight='bold')

                axe.add_patch(rect)
                roi_dict[roi_key] = rect
                roi_text_dict[roi_key] = text

            state.roi_patch_list.append(roi_dict)
            state.roi_text_list.append(roi_text_dict)

        # _register_keyboard_shortcuts
        for key in roi_keys:
            keyboard_dict[key] = q_hotkey__roiset

        keyboard_dict['n'] = q_hotkey__mousemode_Normal
        keyboard_dict['r'] = q_hotkey__mousemode_ROI
    else:
        cbar_index = None

    # _register_events(fig, axes_list, keyboard_dict, state)
    axes_index = {ax: i for i, ax in enumerate(axes_list)}

    def on_key_press(event):
        handler = keyboard_dict.get(event.key)
        if handler:
            handler(fig, event, state)

    def on_key_release(event):
        if not event.key in ['shift', 'alt', 'ctrl']:
            fig.canvas.draw_idle()

    def on_button_press(event):
        if event.inaxes in axes_index:
            state.btn_press_xy = [event.x, event.y]
            state.btn_press_xydata = [event.xdata, event.ydata]
            state.inaxes_flag = True

            # クリックした画像を着目画像(current axes)に指定
            state.current_axes_index = axes_index[event.inaxes]
            fig.sca(event.inaxes)
            for axe in state.axes_list:
                axe.spines['bottom'].set(color="black", linewidth=1)
            event.inaxes.spines['bottom'].set(color="#FF4500", linewidth=6)
            # ダブルクリックで最も大きい画像に合わせて表示領域リセット
            if (event.dblclick) and (event.button == 1):
                for axe in state.axes_list:
                    axe.set_xlim(state.initial_lims[0][0], state.initial_lims[0][1])
                    axe.set_ylim(state.initial_lims[1][0], state.initial_lims[1][1])

            if state.mouse_mode == 'Normal':
                if event.key == "shift":
                    for rect in state.rect_list:
                        rect.set_visible(True)
                else:
                    for rect in state.rect_list:
                        rect.set_visible(False)
            elif state.mouse_mode == 'ROI':
                for rect in state.rect_list:
                    rect.set_visible(False)
                state.rect_list[state.current_axes_index].set_visible(True)

        elif (cbar_index) and (event.inaxes in cbar_index):
            state.btn_press_xy = [event.x, event.y]
            state.btn_press_xydata = [event.xdata, event.ydata]
            state.incbar_flag = True

            state.current_axes_index = cbar_index[event.inaxes]
            fig.sca(state.axes_list[state.current_axes_index])
            for axe in state.axes_list:
                axe.spines['bottom'].set(color="black", linewidth=1)
            state.axes_list[state.current_axes_index].spines['bottom'].set(color="#FF4500", linewidth=6)
            state.btn_press_clim = state.axes_list[state.current_axes_index].images[0].get_clim()

    def on_button_release(event):
        if state.mouse_mode == 'Normal':
            if state.inaxes_flag:
                state.inaxes_flag = False
                for rect in state.rect_list:
                    rect.set_visible(False)

                ax_x_px, ax_y_px = int((axes_list[0].bbox.x1 - axes_list[0].bbox.x0)), int((axes_list[0].bbox.y1 - axes_list[0].bbox.y0))
                move_x, move_y = state.btn_press_xy[0] - event.x, state.btn_press_xy[1] - event.y
                lim_x, lim_y = axes_list[0].get_xlim(), axes_list[0].get_ylim()
                ax_img_pix_x, ax_img_pix_y = lim_x[1] - lim_x[0], lim_y[1] - lim_y[0]
                move_x_pix, move_y_pix = move_x / ax_x_px * ax_img_pix_x, move_y / ax_y_px * ax_img_pix_y

                if event.key == "shift":
                    x_lim = np.sort([state.btn_press_xydata[0], state.btn_press_xydata[0] - move_x_pix])
                    y_lim = np.sort([state.btn_press_xydata[1], state.btn_press_xydata[1] - move_y_pix])
                    for axe in axes_list:
                        axe.set_xlim(x_lim[0], x_lim[1])
                        axe.set_ylim(y_lim[int(bool(0 - y_inverted))], y_lim[int(bool(1 - y_inverted))])
                else:
                    for axe in axes_list:
                        axe.set_xlim(lim_x[0] + move_x_pix, lim_x[1] + move_x_pix)
                        axe.set_ylim(lim_y[0] + move_y_pix, lim_y[1] + move_y_pix)

                fig.canvas.draw_idle()

            elif state.incbar_flag:
                state.incbar_flag = False
                ax_y_px = int((cbar_list[state.current_axes_index].bbox.y1 - cbar_list[state.current_axes_index].bbox.y0))
                move_y = state.btn_press_xy[1] - event.y
                lim_y = cbar_list[state.current_axes_index].get_ylim()
                ax_img_pix_y = lim_y[1] - lim_y[0]
                move_y_pix = move_y / ax_y_px * ax_img_pix_y

                if event.key == "shift":
                    print(lim_y[0], lim_y[1], move_y_pix)
                    if (ax_img_pix_y/2+lim_y[0])<state.btn_press_xydata[1]:
                        axes_list[state.current_axes_index].images[0].set_clim(lim_y[0] , lim_y[1]-move_y_pix)
                    else:
                        axes_list[state.current_axes_index].images[0].set_clim(lim_y[0]-move_y_pix, lim_y[1])

                else:
                    axes_list[state.current_axes_index].images[0].set_clim(lim_y[0]+move_y_pix,lim_y[1]+move_y_pix)
                fig.canvas.draw_idle()


        elif state.mouse_mode == 'ROI':
            if state.inaxes_flag:
                state.inaxes_flag = False

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    fig.canvas.mpl_connect("button_release_event", on_button_release)

    return fig


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


DDDDDDDDDDDDD                                   tttt                                    SSSSSSSSSSSSSSS hhhhhhh                                                     iiii                                        
D::::::::::::DDD                             ttt:::t                                  SS:::::::::::::::Sh:::::h                                                    i::::i                                       
D:::::::::::::::DD                           t:::::t                                 S:::::SSSSSS::::::Sh:::::h                                                     iiii                                        
DDD:::::DDDDD:::::D                          t:::::t                                 S:::::S     SSSSSSSh:::::h                                                                                                 
  D:::::D    D:::::D   aaaaaaaaaaaaa   ttttttt:::::ttttttt      aaaaaaaaaaaaa        S:::::S             h::::h hhhhh         aaaaaaaaaaaaa   ppppp   ppppppppp   iiiiiii nnnn  nnnnnnnn       ggggggggg   ggggg
  D:::::D     D:::::D  a::::::::::::a  t:::::::::::::::::t      a::::::::::::a       S:::::S             h::::hh:::::hhh      a::::::::::::a  p::::ppp:::::::::p  i:::::i n:::nn::::::::nn    g:::::::::ggg::::g
  D:::::D     D:::::D  aaaaaaaaa:::::a t:::::::::::::::::t      aaaaaaaaa:::::a       S::::SSSS          h::::::::::::::hh    aaaaaaaaa:::::a p:::::::::::::::::p  i::::i n::::::::::::::nn  g:::::::::::::::::g
  D:::::D     D:::::D           a::::a tttttt:::::::tttttt               a::::a        SS::::::SSSSS     h:::::::hhh::::::h            a::::a pp::::::ppppp::::::p i::::i nn:::::::::::::::ng::::::ggggg::::::gg
  D:::::D     D:::::D    aaaaaaa:::::a       t:::::t              aaaaaaa:::::a          SSS::::::::SS   h::::::h   h::::::h    aaaaaaa:::::a  p:::::p     p:::::p i::::i   n:::::nnnn:::::ng:::::g     g:::::g 
  D:::::D     D:::::D  aa::::::::::::a       t:::::t            aa::::::::::::a             SSSSSS::::S  h:::::h     h:::::h  aa::::::::::::a  p:::::p     p:::::p i::::i   n::::n    n::::ng:::::g     g:::::g 
  D:::::D     D:::::D a::::aaaa::::::a       t:::::t           a::::aaaa::::::a                  S:::::S h:::::h     h:::::h a::::aaaa::::::a  p:::::p     p:::::p i::::i   n::::n    n::::ng:::::g     g:::::g 
  D:::::D    D:::::D a::::a    a:::::a       t:::::t    tttttta::::a    a:::::a                  S:::::S h:::::h     h:::::ha::::a    a:::::a  p:::::p    p::::::p i::::i   n::::n    n::::ng::::::g    g:::::g 
DDD:::::DDDDD:::::D  a::::a    a:::::a       t::::::tttt:::::ta::::a    a:::::a      SSSSSSS     S:::::S h:::::h     h:::::ha::::a    a:::::a  p:::::ppppp:::::::pi::::::i  n::::n    n::::ng:::::::ggggg:::::g 
D:::::::::::::::DD   a:::::aaaa::::::a       tt::::::::::::::ta:::::aaaa::::::a      S::::::SSSSSS:::::S h:::::h     h:::::ha:::::aaaa::::::a  p::::::::::::::::p i::::::i  n::::n    n::::n g::::::::::::::::g 
D::::::::::::DDD      a::::::::::aa:::a        tt:::::::::::tt a::::::::::aa:::a     S:::::::::::::::SS  h:::::h     h:::::h a::::::::::aa:::a p::::::::::::::pp  i::::::i  n::::n    n::::n  gg::::::::::::::g 
DDDDDDDDDDDDD          aaaaaaaaaa  aaaa          ttttttttttt    aaaaaaaaaa  aaaa      SSSSSSSSSSSSSSS    hhhhhhh     hhhhhhh  aaaaaaaaaa  aaaa p::::::pppppppp    iiiiiiii  nnnnnn    nnnnnn    gggggggg::::::g 
                                                                                                                                               p:::::p                                                  g:::::g 
                                                                                                                                               p:::::p                                      gggggg      g:::::g 
                                                                                                                                              p:::::::p                                     g:::::gg   gg:::::g 
                                                                                                                                              p:::::::p                                      g::::::ggg:::::::g 
                                                                                                                                              p:::::::p                                       gg:::::::::::::g  
                                                                                                                                              ppppppppp                                         ggg::::::ggg    
                                                                                                                                                                                                   gggggg                                                                                                                                                         
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def qutil_data_shaping_2dlist_main(input_data):
    """
    入力データを2次元リストにし、
    同じ構造を持つ0埋めのリストを返す。
    """
    # ケース1: 入力がリストでない場合
    if not isinstance(input_data, list):
        output_2dlisted_data = [[input_data]]
        output_shape_template = [[0]]
        return output_2dlisted_data, output_shape_template
    # ケース2: 入力が1次元リストの場合（要素がすべて非リスト）
    contains_list = any(isinstance(element, list) for element in input_data)  # input_data の中に「list 型の要素が1つでも含まれているか」を判定
    if not contains_list:
        output_2dlisted_data = [input_data]
        output_shape_template = [[0 for _ in input_data]]
        return output_2dlisted_data, output_shape_template
    # ケース3: 入力が2次元相当（リストと非リストが混在）
    output_2dlisted_data = []
    output_shape_template = []
    for element in input_data:
        if isinstance(element, list):  # 要素が list の場合
            output_2dlisted_data.append(element)
            output_shape_template.append([0 for _ in element])
        else:  # 要素が list ではない場合
            output_2dlisted_data.append([element])
            output_shape_template.append([0])
    return output_2dlisted_data, output_shape_template


def qutil_data_shaping_2dlist_sub(input_data, shape_template):
    """
    shape_template の構造に合わせて input_data を整形する関数。
    - 非リスト(スカラ、タプル、辞書、etc...)   : shape 全体にブロードキャスト
    - 1次元リスト                              : 行をまたいで shape に順番詰め
    - 2次元リスト                              : 行対応を維持して shape に切り詰め
    """

    ### ケース1: input_data が非リストの場合
    if not isinstance(input_data, list):
        output_data = []
        for row_shape in shape_template:
            output_data.append([input_data for _ in row_shape])
        return output_data

    ### ケース2: input_data がリストの場合
    # list の次元判定
    is_2d_input = any(isinstance(element, list) for element in input_data)
    is_2d_shape = any(isinstance(element, list) for element in shape_template)
    # ケース2-1: input_data が1次元リスト
    if not is_2d_input:
        # 要素数チェック
        input_count = len(input_data)
        shape_count = sum(len(row) for row in shape_template)
        # 要素数＋形状チェック
        if is_2d_shape:  # shape_templateが2次元
            print("warning:shape mismatch! ::: input_data -> 1d | shape_template -> 2d")
            if input_count == shape_count:
                print(
                    "warning:shape mismatch! ::: input_data's element number == shape_template's element number : Adjust the data shape and run")
        if input_count > shape_count:
            print(
                "warning:shape mismatch! ::: input_data's element number > shape_template's element number : Adjust the data shape and run, but input_data will be truncated.")
        elif input_count < shape_count:
            raise ValueError(
                " shape mismatch! ::: input_data's element number < shape_template's element number : Cannot adjust data shape, stopping.")

        # 出力データ成形
        reshaped_data = []
        index = 0
        # shape に従って順番に詰める
        for row_shape in shape_template:
            row_length = len(row_shape)
            reshaped_row = input_data[index:index + row_length]
            reshaped_data.append(reshaped_row)
            index += row_length

        return reshaped_data

    # ケース2-2: input_data が2次元リスト
    else:
        if len(shape_template) != len(input_data):
            raise ValueError(
                " shape mismatch! ::: input_data and shape_template are 2d-list, but the number of rows is different.")

        reshaped_data = []
        for row_index, row_shape in enumerate(shape_template):
            if len(shape_template[row_index]) < len(shape_template[row_index]):
                raise ValueError(
                    " shape mismatch! input_data and shape_template are 2d-list, but the number of elements in one of input_data's rows is less than that in shape_template.")
            elif len(shape_template[row_index]) > len(shape_template[row_index]):
                print(
                    "warning:shape mismatch! ::: input_data's element number > shape_template's element number : Adjust the data shape and run, but input_data will be truncated.")
            required_length = len(row_shape)
            reshaped_row = input_data[row_index][:required_length]
            reshaped_data.append(reshaped_row)
        return reshaped_data


def qutil_color_shaping_2dlist_sub(input_color, shape_template):
    output_color = []
    if input_color in matplotlib_colormap_list:
        cm = plt.cm.get_cmap(input_color)
        main_data_num = sum(len(v) for v in shape_template)
        i = 0
        for y in np.arange(len(shape_template)):
            temp = []
            for x in range(len(shape_template[y])):
                temp.append(cm(i / np.clip(main_data_num - 1, 1, None)))
                i = i + 1
            output_color.append(temp)
    elif input_color == 'kutinawa_color':
        i = 0
        for y in np.arange(len(shape_template)):
            temp = []
            for x in range(len(shape_template[y])):
                temp.append(kutinawa_color[i % len(kutinawa_color)])
                i = i + 1
            output_color.append(temp)
    else:
        output_color = qutil_data_shaping_2dlist_sub(input_color, shape_template)
        # if isinstance(input_color,list):
        #     output_color = input_color
        # else:
        #     for y in np.arange(len(shape_template)):
        #         output_color.append([input_color for x in range(len(shape_template[y]))])

    return output_color


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IIIIIIIIII                                                                                                      
I::::::::I                                                                                                      
I::::::::I                                                                                                      
II::::::II                                                                                                      
  I::::I          mmmmmmm    mmmmmmm          aaaaaaaaaaaaa           ggggggggg   ggggg         eeeeeeeeeeee    
  I::::I        mm:::::::m  m:::::::mm        a::::::::::::a         g:::::::::ggg::::g       ee::::::::::::ee  
  I::::I       m::::::::::mm::::::::::m       aaaaaaaaa:::::a       g:::::::::::::::::g      e::::::eeeee:::::ee
  I::::I       m::::::::::::::::::::::m                a::::a      g::::::ggggg::::::gg     e::::::e     e:::::e
  I::::I       m:::::mmm::::::mmm:::::m         aaaaaaa:::::a      g:::::g     g:::::g      e:::::::eeeee::::::e
  I::::I       m::::m   m::::m   m::::m       aa::::::::::::a      g:::::g     g:::::g      e:::::::::::::::::e 
  I::::I       m::::m   m::::m   m::::m      a::::aaaa::::::a      g:::::g     g:::::g      e::::::eeeeeeeeeee  
  I::::I       m::::m   m::::m   m::::m     a::::a    a:::::a      g::::::g    g:::::g      e:::::::e           
II::::::II     m::::m   m::::m   m::::m     a::::a    a:::::a      g:::::::ggggg:::::g      e::::::::e          
I::::::::I     m::::m   m::::m   m::::m     a:::::aaaa::::::a       g::::::::::::::::g       e::::::::eeeeeeee  
I::::::::I     m::::m   m::::m   m::::m      a::::::::::aa:::a       gg::::::::::::::g        ee:::::::::::::e  
IIIIIIIIII     mmmmmm   mmmmmm   mmmmmm       aaaaaaaaaa  aaaa         gggggggg::::::g          eeeeeeeeeeeeee  
                                                                               g:::::g                          
                                                                   gggggg      g:::::g                          
                                                                   g:::::gg   gg:::::g                          
                                                                    g::::::ggg:::::::g                          
                                                                     gg:::::::::::::g                           
                                                                       ggg::::::ggg                             
                                                                          gggggg                                                                                                                                                                               
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def imageq(tgt_img_list, caxis_list=(0, 0), cmap_list='viridis', disp_cbar=True, fig=None, mode='comp'):
    plt.interactive(False)
    # fig受け取っていなければ新規作成
    if not fig:
        fig = plt.figure()

    # tgt_shape_templateをお手本配列形状としてパラメータ類の形状成形、必ず2次元listの形状にする
    tgt_img_list, tgt_shape_template = qutil_data_shaping_2dlist_main(tgt_img_list)
    caxis_list = qutil_data_shaping_2dlist_sub(caxis_list, tgt_shape_template)
    cmap_list = qutil_data_shaping_2dlist_sub(cmap_list, tgt_shape_template)
    keyboard_dict = {'tab': q_hotkey__reset, 'n': q_hotkey__mousemode_Normal, 'r': q_hotkey__mousemode_ROI,
                     'p': q_hotkey__png_save,
                     'A': q_hotkey__climAUTO, 'W': q_hotkey__climWHOLE, 'E': q_hotkey__climEACH,
                     'S': q_hotkey__climSYNC,
                     '>': q_hotkey__climMANUAL_top_down, 'alt+>': q_hotkey__climMANUAL_top_up,
                     'up': q_hotkey__climSYNCup, 'down': q_hotkey__climSYNCdown,
                     'left': q_hotkey__climSYNCleft, 'right': q_hotkey__climSYNCright,
                     'i': q_hotkey__lineprofV, '-': q_hotkey__lineprofH, 'I': q_hotkey__lineprofVmean,
                     '=': q_hotkey__lineprofHmean,
                     '#': q_hotkey__roistats, 'H': q_hotkey__ROIhist,
                     'm':q_hotkey__switch_cmap_linear_gamma,
                     }
    if mode == 'layer':
        tgt_img_list_layer       = [[tgt_img_list[0][0]      ]]
        tgt_shape_template_layer = [[tgt_shape_template[0][0]]]
        caxis_list_layer         = [[caxis_list[0][0]        ]]
        cmap_list_layer          = [[cmap_list[0][0]         ]]
        for y_id in range(len(tgt_shape_template)):
            for x_id in range(len(tgt_shape_template[y_id])):
                tgt_img_list_layer[0].append(tgt_img_list[y_id][x_id])
                tgt_shape_template_layer[0].append(tgt_shape_template[y_id][x_id])
                caxis_list_layer[0].append(caxis_list[y_id][x_id])
                cmap_list_layer[0].append(cmap_list[y_id][x_id])
        tgt_img_list        = tgt_img_list_layer
        tgt_shape_template  = tgt_shape_template_layer
        caxis_list          = caxis_list_layer
        cmap_list           = cmap_list_layer

        keyboard_dict.update([('f'+str(i),q_hotkey__layer) for i in np.arange(1,13)])


    # 各imshow描画
    y_id_max = len(tgt_shape_template)
    x_id_max = np.max(np.array([len(i) for i in tgt_shape_template]))
    for y_id in range(len(tgt_shape_template)):
        for x_id in range(len(tgt_shape_template[y_id])):

            if mode=='layer':
                if y_id+x_id==0:
                    ax = fig.add_subplot(13, 12, (1,144), picker=True)
                else:
                    ax = fig.add_subplot(13, 12, 144+x_id, picker=True)
            else:
                ax = fig.add_subplot(y_id_max, x_id_max, x_id_max * y_id + x_id + 1, picker=True)

            # 描画画像指定
            tgt_img = tgt_img_list[y_id][x_id]
            # caxis指定がmin>=maxの場合、画素値の最小最大から自動でcaxis指定
            if caxis_list[y_id][x_id][0] >= caxis_list[y_id][x_id][1]:
                # inf、nanを除いたmin-max
                caxis_list[y_id][x_id] = (np.nanmin(tgt_img[(tgt_img != -np.inf) * (tgt_img != np.inf)]),
                                          np.nanmax(tgt_img[(tgt_img != -np.inf) * (tgt_img != np.inf)]))

            # 各subplot描画
            ims = []
            if np.ndim(tgt_img) == 2:  # 1ch画像の場合
                ims = ax.imshow(tgt_img.astype(float), interpolation='nearest', cmap=cmap_list[y_id][x_id],interpolation_stage='data',)
                ims.set_clim(caxis_list[y_id][x_id][0], caxis_list[y_id][x_id][1])

            elif np.ndim(tgt_img) == 3:  # 2ch以上画像の場合
                # 画像をcmin~cmaxのレンジで正規化するため、値域調整
                print(
                    "imq-Warning: The image was normalized to 0-1 and clipped in the cmin-cmax range for a 3-channel image.")
                ims = ax.imshow(np.clip((tgt_img.astype(float) - caxis_list[y_id][x_id][0]) / (
                            caxis_list[y_id][x_id][1] - caxis_list[y_id][x_id][0]), 0, 1),
                                interpolation='nearest', cmap=cmap_list[y_id][x_id],interpolation_stage='data')

            elif np.ndim(tgt_img) == 1 or np.ndim(tgt_img) >= 4:
                # 画像は表示できないのでエラー返して終了
                raise ValueError(" imageq can only draw 2 or 3dimensional")

            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.tick_params(bottom=False, left=False, right=False, top=False)

            if disp_cbar:
                divider = make_axes_locatable(ax)
                ax_cbar = divider.new_horizontal(size="5%", pad=0.075)
                fig.add_axes(ax_cbar)
                fig.colorbar(ims, cax=ax_cbar)
                pass

    axes_list = fig.get_axes()[0::2]
    cbar_list = fig.get_axes()[1::2]

    fig = q_addon(fig, axes_list,keyboard_dict=keyboard_dict,imageq=True, cbar_list=cbar_list)

    ###############################
    # status barの表示変更
    def format_coord(x, y):
        int_x = int(x + 0.5)
        int_y = int(y + 0.5)
        return_str = 'x=' + str(int_x) + ', y=' + str(int_y) + ' |  '
        for k, axe in enumerate(axes_list):
            now_img = axe.images[0].get_array().data
            if 0 <= int_x < np.shape(now_img)[1] and 0 <= int_y < np.shape(now_img)[0]:
                now_img_val = now_img[int_y, int_x]
                if np.sum(np.isnan(now_img_val)) or np.sum(np.isinf(now_img_val)):
                    return_str = return_str + str(k) + ': ###' + '  '
                else:
                    if np.ndim(now_img_val) == 0:
                        return_str = return_str + str(k) + ': ' + '{:.3f}'.format(now_img_val) + '  '
                    else:
                        return_str = return_str + str(k) + ': <' + '{:.3f}'.format(
                            now_img_val[0]) + ', ' + '{:.3f}'.format(now_img_val[1]) + ', ' + '{:.3f}'.format(
                            now_img_val[2]) + '>  '
            else:
                return_str = return_str + str(k) + ': ###' + '  '
        # 対処には、https://stackoverflow.com/questions/47082466/matplotlib-imshow-formatting-from-cursor-position
        # のような実装が必要になり、別の関数＋matplotlibの関数を叩くが必要ありめんどくさい
        return return_str

    for axe in axes_list:
        axe.format_coord = format_coord

    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
    fig.show()

    return fig


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                                     
HHHHHHHHH     HHHHHHHHH       iiii                                      tttt          
H:::::::H     H:::::::H      i::::i                                  ttt:::t          
H:::::::H     H:::::::H       iiii                                   t:::::t          
HH::::::H     H::::::HH                                              t:::::t          
  H:::::H     H:::::H       iiiiiii          ssssssssss        ttttttt:::::ttttttt    
  H:::::H     H:::::H       i:::::i        ss::::::::::s       t:::::::::::::::::t    
  H::::::HHHHH::::::H        i::::i      ss:::::::::::::s      t:::::::::::::::::t    
  H:::::::::::::::::H        i::::i      s::::::ssss:::::s     tttttt:::::::tttttt    
  H:::::::::::::::::H        i::::i       s:::::s  ssssss            t:::::t          
  H::::::HHHHH::::::H        i::::i         s::::::s                 t:::::t          
  H:::::H     H:::::H        i::::i            s::::::s              t:::::t          
  H:::::H     H:::::H        i::::i      ssssss   s:::::s            t:::::t    tttttt
HH::::::H     H::::::HH     i::::::i     s:::::ssss::::::s           t::::::tttt:::::t
H:::::::H     H:::::::H     i::::::i     s::::::::::::::s            tt::::::::::::::t
H:::::::H     H:::::::H     i::::::i      s:::::::::::ss               tt:::::::::::tt
HHHHHHHHH     HHHHHHHHH     iiiiiiii       sssssssssss                   ttttttttttt                                                                    
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def histq(tgt_data_list,
          interval_list=None,
          label_list=None,
          alpha_list=0.75,
          edgecolor_list='kutinawa_color',
          color_list='kutinawa_color',
          histtype='bar',
          overlay=False
          ):
    ############################### 準備
    plt.interactive(False)
    fig = plt.figure()

    ############################### 必ず2次元listの形状にする
    tgt_data_list, tgt_shape_template = qutil_data_shaping_2dlist_main(tgt_data_list)
    if not interval_list:
        interval_list = []
        for row in range(len(tgt_data_list)):
            interval_list.append([])
            for col in range(len(tgt_data_list[row])):
                temp_tgt = tgt_data_list[row][col]
                data_range = np.nanmax(temp_tgt[(temp_tgt != -np.inf) * (temp_tgt != np.inf)]) - np.nanmin(
                    temp_tgt[(temp_tgt != -np.inf) * (temp_tgt != np.inf)])
                interval_list[-1].append(data_range / np.sqrt(temp_tgt.size))
    interval_list = qutil_data_shaping_2dlist_sub(input_data=interval_list,
                                                  shape_template=tgt_shape_template)
    # if not label_list:
    #     label_list = [str(i) for i in range(sum(len(row) for row in tgt_data_list))]
    if not label_list:
        label_list = []
        i = 0
        for row in range(len(tgt_data_list)):
            label_list.append([])
            for col in range(len(tgt_data_list[row])):
                label_list[-1].append(str(i))
                i = i + 1
    label_list = qutil_data_shaping_2dlist_sub(input_data=label_list,
                                               shape_template=tgt_shape_template)
    alpha_list = qutil_data_shaping_2dlist_sub(input_data=alpha_list,
                                               shape_template=tgt_shape_template)
    edgecolor_list = qutil_color_shaping_2dlist_sub(input_color=edgecolor_list,
                                                    shape_template=tgt_shape_template)
    color_list = qutil_color_shaping_2dlist_sub(input_color=color_list,
                                                shape_template=tgt_shape_template)

    ############################### 各描画
    y_id_max = len(tgt_data_list)
    x_id_max = np.max(np.array([len(i) for i in tgt_data_list]))
    if overlay:
        ax_id = 1
        ax = fig.add_subplot(1, 1, 1, picker=True)
        y_id_max = 1
        x_id_max = 1

    x_min = []
    x_max = []
    y_max = []
    for y_id, temp_list in enumerate(tgt_data_list):
        for x_id, target_data in enumerate(temp_list):
            if not overlay:
                ax_id = x_id_max * y_id + x_id + 1
                ax = fig.add_subplot(y_id_max, x_id_max, ax_id, picker=True)

            hist_bins = np.arange(np.nanmin(target_data), np.nanmax(target_data) + interval_list[y_id][x_id] * 2,interval_list[y_id][x_id])

            hi = ax.hist(np.squeeze(np.reshape(target_data, (1, -1))),
                         bins=hist_bins,
                         label=label_list[y_id][x_id],
                         color=color_list[y_id][x_id],
                         edgecolor=edgecolor_list[y_id][x_id],
                         alpha=alpha_list[y_id][x_id],
                         histtype=histtype)

            x_min.append(np.nanmin(target_data))
            x_max.append(np.nanmax(target_data))
            y_max.append(np.max(hi[0]))

    x_min = np.min(np.array(x_min))
    x_max = np.max(np.array(x_max))
    y_max = np.max(np.array(y_max))
    x_spc = (x_max - x_min) * 0.0495

    ############################### キーボードショートカット追加
    axes_list = fig.get_axes()
    fig = q_addon(fig, axes_list, keyboard_dict={'tab': q_hotkey__reset,
                                                 'p': q_hotkey__png_save, })

    ############################### 表示
    for axe in axes_list:
        axe.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={"weight": "bold", "size": "large"})
        axe.set_xlim(x_min - x_spc, x_max + x_spc)
        axe.set_ylim(0, y_max * 1.05)

    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
    fig.show()

    return fig


"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PPPPPPPPPPPPPPPPP        lllllll                                     tttt          
P::::::::::::::::P       l:::::l                                  ttt:::t          
P::::::PPPPPP:::::P      l:::::l                                  t:::::t          
PP:::::P     P:::::P     l:::::l                                  t:::::t          
  P::::P     P:::::P      l::::l         ooooooooooo        ttttttt:::::ttttttt    
  P::::P     P:::::P      l::::l       oo:::::::::::oo      t:::::::::::::::::t    
  P::::PPPPPP:::::P       l::::l      o:::::::::::::::o     t:::::::::::::::::t    
  P:::::::::::::PP        l::::l      o:::::ooooo:::::o     tttttt:::::::tttttt    
  P::::PPPPPPPPP          l::::l      o::::o     o::::o           t:::::t          
  P::::P                  l::::l      o::::o     o::::o           t:::::t          
  P::::P                  l::::l      o::::o     o::::o           t:::::t          
  P::::P                  l::::l      o::::o     o::::o           t:::::t    tttttt
PP::::::PP               l::::::l     o:::::ooooo:::::o           t::::::tttt:::::t
P::::::::P               l::::::l     o:::::::::::::::o           tt::::::::::::::t
P::::::::P               l::::::l      oo:::::::::::oo              tt:::::::::::tt
PPPPPPPPPP               llllllll        ooooooooooo                  ttttttttttt                                          
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

def plotq(tgt_list_x,
          tgt_list_y=None,
          marker_list='None',
          markersize_list=7,
          linestyle_list='-',
          linewidth_list=2,
          color_list='kutinawa_color',
          alpha_list=1.0,
          label_list=None,
          overlay=True,
          fig=None,
          ):
    ############################### 準備
    plt.interactive(False)
    if not fig:
        fig = plt.figure()

    ############################### 必ず2次元listの形状にする
    if tgt_list_y:
        tgt_list_x, tgt_shape_template = qutil_data_shaping_2dlist_main(tgt_list_x)
        tgt_list_y = qutil_data_shaping_2dlist_sub(tgt_list_y, tgt_shape_template)
    else:
        tgt_list_y, tgt_shape_template = qutil_data_shaping_2dlist_main(tgt_list_x)
        tgt_list_x = []
        for row in range(len(tgt_list_y)):
            tgt_list_x.append([])
            for col in range(len(tgt_list_y[row])):
                tgt_list_x[-1].append(np.arange(len(tgt_list_y[row][col])))

    marker_list = qutil_data_shaping_2dlist_sub(input_data=marker_list,
                                                shape_template=tgt_shape_template)
    markersize_list = qutil_data_shaping_2dlist_sub(input_data=markersize_list,
                                                    shape_template=tgt_shape_template)
    linestyle_list = qutil_data_shaping_2dlist_sub(input_data=linestyle_list,
                                                   shape_template=tgt_shape_template)
    linewidth_list = qutil_data_shaping_2dlist_sub(input_data=linewidth_list,
                                                   shape_template=tgt_shape_template)
    alpha_list = qutil_data_shaping_2dlist_sub(input_data=alpha_list,
                                               shape_template=tgt_shape_template)
    if not label_list:
        label_list = []
        i = 0
        for row in range(len(tgt_list_y)):
            label_list.append([])
            for col in range(len(tgt_list_y[row])):
                label_list[-1].append(str(i))
                i = i + 1
    label_list = qutil_data_shaping_2dlist_sub(input_data=label_list,
                                               shape_template=tgt_shape_template)
    color_list = qutil_color_shaping_2dlist_sub(input_color=color_list,
                                                shape_template=tgt_shape_template)

    ############################### 各描画
    y_id_max = len(tgt_shape_template)
    x_id_max = np.max(np.array([len(i) for i in tgt_shape_template]))
    xxx = [np.inf, -np.inf]
    yyy = [np.inf, -np.inf]
    if overlay:
        ax_id = 1
        ax = fig.add_subplot(1, 1, 1, picker=True)
        y_id_max = 1
        x_id_max = 1

    for y_id, temp_list in enumerate(tgt_shape_template):
        for x_id, temp2_data in enumerate(temp_list):
            if not overlay:
                ax_id = x_id_max * y_id + x_id + 1
                ax = fig.add_subplot(y_id_max, x_id_max, ax_id, picker=True)

            ax.plot(np.squeeze(np.reshape(tgt_list_x[y_id][x_id], (1, -1))),
                    np.squeeze(np.reshape(tgt_list_y[y_id][x_id], (1, -1))),
                    label=label_list[y_id][x_id],
                    color=color_list[y_id][x_id],
                    alpha=alpha_list[y_id][x_id],
                    marker=marker_list[y_id][x_id],
                    markersize=markersize_list[y_id][x_id],
                    linestyle=linestyle_list[y_id][x_id],
                    linewidth=linewidth_list[y_id][x_id],
                    )

            xxx[0] = np.minimum(xxx[0], np.nanmin(tgt_list_x[y_id][x_id].astype(float)))
            xxx[1] = np.maximum(xxx[1], np.nanmax(tgt_list_x[y_id][x_id].astype(float)))
            yyy[0] = np.minimum(yyy[0], np.nanmin(tgt_list_y[y_id][x_id].astype(float)))
            yyy[1] = np.maximum(yyy[1], np.nanmax(tgt_list_y[y_id][x_id].astype(float)))
            x_spc = (xxx[1] - xxx[0]) * 0.0495
            y_spc = (yyy[1] - yyy[0]) * 0.0495
    ############################### キーボードショートカット追加
    axes_list = fig.get_axes()
    fig = q_addon(fig, axes_list, keyboard_dict={'tab': q_hotkey__reset,
                                                 'p': q_hotkey__png_save, })

    ############################### 表示
    for axe in axes_list:
        axe.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={"weight": "bold", "size": "large"})
        axe.set_xlim(xxx[0] - x_spc, xxx[1] + x_spc)
        axe.set_ylim(yyy[0] - y_spc, yyy[1] + y_spc)

    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
    fig.show()

    return fig

# plotq([np.random.rand(100),np.random.rand(100)],color_list=['m','c'])
# histq([np.random.rand(100),np.random.rand(100)],color_list=['m','c'])
# histq([np.random.rand(100),np.random.rand(100)],)










