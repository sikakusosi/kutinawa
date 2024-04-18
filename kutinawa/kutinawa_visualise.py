
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

from .kutinawa_depot import weighted_least_squares

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

kutinawa_color = ['#FF7171','#57B8FF','#9DDD15','#FF8D44','#7096F8','#51B883','#FFC700','#BB87FF','#2BC8E4','#F661F6',
                  '#EC0000','#0066BE','#618E00','#C74700','#0031D8','#197A48','#A58000','#5C10BE','#008299','#AA00AA',
                  '#FFDADA','#DCF0FF','#D0F5A2','#FFDFCA','#D9E6FF','#C2E5D1','#FFF0B3','#ECDDFF','#C8F8FF','#FFD0FF',]

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
def table_print(data,headers=[],table_mode='adapt',format_alignment='<',format_min_w=12,format_significant_digits=5):
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
    temp_format = '{:' + format_alignment + str(format_min_w) + '.' + str(format_significant_digits) + 'f}' #'{:<12.5f}'
    # temp_format_header = '{:' + format_alignment + str(format_min_w) + '}'

    #dataを必ず充填済みの２次元listにする
    table_hw = np.shape(data)

    # header不足があれば追加
    lh = len(headers)
    if len(headers)<table_hw[1]:
        for i in np.arange(table_hw[1]-lh):
            headers.append('Col '+str(i+lh))

    # 表の横幅取得
    # max_width_list = np.array([[len(temp_format.format(x)) for x in y] for y in data])
    width_list = []
    for y in data+[headers]:
        width_list.append([])
        for i,x in enumerate(y):
            if type(x) is str:
                now_pf = '{:' + format_alignment + str(format_min_w) + '}'
            else:
                now_pf = '{:' + format_alignment + str(format_min_w) + '.' + str(format_significant_digits) + 'f}'
            width_list[-1].append( len(now_pf.format(x)) )
    width_list = np.array(width_list)

    if table_mode=='equal':
        temp = np.max(width_list)
        max_width_list = [temp for i in np.arange(table_hw[1])]
    elif table_mode=='adapt':
        max_width_list = [np.max(width_list[:,i]) for i in np.arange(table_hw[1])]

    # print
    print(end='│')
    for i,hd in enumerate(headers):
        now_pf = '{:' + format_alignment + str(max_width_list[i]) + '}'
        print(now_pf.format(hd), end='│')

    print(end='\n╞')
    for i,hd in enumerate(headers[:-1]):
        now_pf = '{:' + format_alignment + str(max_width_list[i]) + '}'
        print(now_pf.format('═' * max_width_list[i]), end='╪')
    now_pf = '{:' + format_alignment + str(max_width_list[-1]) + '}'
    print(now_pf.format('═' * max_width_list[-1]), end='╡')

    for y in data:
        print(end='\n│')
        for i,x in enumerate(y):
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

def tolist_only1axis(tgt_array, axis):
    axis_temp = np.arange(np.ndim(tgt_array))
    return [i for i in np.transpose(tgt_array, tuple([axis] + (axis_temp[axis_temp != axis]).tolist()))]

def list1toSQ2(tgt_list):
    sub_x = np.ceil(np.sqrt(len(tgt_list))).astype(int)
    sub_y = np.ceil(len(tgt_list) / sub_x).astype(int)
    idx_l = np.arange(0,len(tgt_list),sub_x).tolist()+[len(tgt_list)]
    return [tgt_list[idx_l[h]:idx_l[h+1]] for h in np.arange(sub_y)]

def imq_inEASY(tgt_imgs,axis):
    if isinstance(tgt_imgs, np.ndarray):
        out_imgs = list1toSQ2(tolist_only1axis(tgt_imgs, axis))
    else:
        out_imgs = list1toSQ2(tgt_imgs)
    return out_imgs

def cmap_out_range_color(cmap_name='viridis',over_color='white',under_color='black',bad_color='red'):
    cm = pylab.cm.get_cmap(cmap_name)
    colors = cm.colors
    out_cmap = ListedColormap(colors,name='custom',N=255)
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

############################################################################ clim
def q_hotkey__climAUTO(fig, event, hotkey_use):
    c_axe = fig.gca()
    now_lim_x = np.clip((np.array(c_axe.get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(c_axe.get_ylim()) + 0.5).astype(int),0,None)
    temp = c_axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
    c_axe.images[0].set_clim((np.nanmin(temp[(temp!=-np.inf)*(temp!=np.inf)]), np.nanmax(temp[(temp!=-np.inf)*(temp!=np.inf)])))
    pass

def q_hotkey__climWHOLE(fig, event, hotkey_use):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    now_lim_x = np.clip((np.array(axes_list[0].get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(axes_list[0].get_ylim()) + 0.5).astype(int),0,None)
    whole_max = np.nanmax(np.array([np.nanmax(axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for axe in axes_list]))
    whole_min = np.nanmin(np.array([np.nanmin(axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]) for axe in axes_list]))
    for axe in axes_list:
        axe.images[0].set_clim(whole_min,whole_max)
    pass

def q_hotkey__climEACH(fig, event, hotkey_use):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    now_lim_x = np.clip((np.array(axes_list[0].get_xlim()) + 0.5).astype(int),0,None)
    now_lim_y = np.clip((np.array(axes_list[0].get_ylim()) + 0.5).astype(int),0,None)
    for axe in axes_list:
        temp = axe.images[0].get_array().data[now_lim_y[1]:now_lim_y[0], now_lim_x[0]:now_lim_x[1]]
        axe.images[0].set_clim((np.nanmin(temp[(temp!=-np.inf)*(temp!=np.inf)]), np.nanmax(temp[(temp!=-np.inf)*(temp!=np.inf)])))
    pass

def q_hotkey__climSYNC(fig, event, hotkey_use):
    c_axe = fig.gca()
    sync_clim = c_axe.images[0].get_clim()
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    for axe in axes_list:
        axe.images[0].set_clim(sync_clim[0],sync_clim[1])
    pass

def q_hotkey_util__climSYNCudlr(fig, event, hotkey_use, direction):
    c_axe = fig.gca()
    sync_clim = c_axe.images[0].get_clim()
    sps = c_axe.get_subplotspec()
    subplot_x, subplot_y = sps.colspan[0], sps.rowspan[0]

    x_adj = {'up':0,'down':0,'left':-1,'right':1}
    y_adj = {'up':-1,'down':1,'left':0,'right':0}
    sync_tgt_x = subplot_x+x_adj[direction]
    sync_tgt_y = subplot_y+y_adj[direction]

    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    for axe in axes_list:
        sps = axe.get_subplotspec()
        if sps.colspan[0]==sync_tgt_x and sps.rowspan[0]==sync_tgt_y:
            axe.images[0].set_clim(sync_clim[0], sync_clim[1])
    pass

def q_hotkey__climSYNCup(fig, event, hotkey_use):
    q_hotkey_util__climSYNCudlr(fig, event, hotkey_use, 'up')
    pass
def q_hotkey__climSYNCdown(fig, event, hotkey_use):
    q_hotkey_util__climSYNCudlr(fig, event, hotkey_use, 'down')
    pass
def q_hotkey__climSYNCleft(fig, event, hotkey_use):
    q_hotkey_util__climSYNCudlr(fig, event, hotkey_use, 'left')
    pass
def q_hotkey__climSYNCright(fig, event, hotkey_use):
    q_hotkey_util__climSYNCudlr(fig, event, hotkey_use, 'right')
    pass

# ############################################################################ roi
def q_hotkey__roistats(fig, event, hotkey_use):
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height()>0]
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe,matplotlib.axes._subplots.Subplot)]

    stats_list = np.zeros((len(axes_list),len(active_roi_keys))).tolist()
    for axe_num in np.arange(len(axes_list)):
        axes_img = axes_list[axe_num].images[0].get_array().data
        for i,rk in enumerate(active_roi_keys):
            tgt_roi = hotkey_use['roi_list'][axe_num][rk]
            roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
            roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
            roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y,0,hw[0]), np.clip(roi_yy,0,hw[0]), np.clip(roi_x,0,hw[1]), np.clip(roi_xx,0,hw[1])
            roi_img = axes_img[roi_y:roi_yy,roi_x:roi_xx,...]

            stats_list[axe_num][i] = {'roi_x':roi_x,'roi_xx':roi_xx,
                                      'roi_y':roi_y,'roi_yy':roi_yy,
                                      'mean':np.nanmean(roi_img,axis=(0,1)),
                                      'std' :np.nanstd(roi_img,axis=(0,1)),
                                      'min' :np.nanmin(roi_img,axis=(0,1)),
                                      'max' :np.nanmax(roi_img,axis=(0,1)),
                                      'med' :np.nanmedian(roi_img,axis=(0,1)),}


    print('======================== ROI stats ======================== ')
    for i,rk in enumerate(active_roi_keys):
        print('ROI key='+ rk
              +'    pos=['+str(stats_list[0][i]['roi_y'])+':'+str(stats_list[0][i]['roi_yy'])+', '+str(stats_list[0][i]['roi_x'])+':'+str(stats_list[0][i]['roi_xx'])+']'
              +'    pixel num=' + str(stats_list[0][i]['roi_yy']-stats_list[0][i]['roi_y'])+'*'+str(stats_list[0][i]['roi_xx']-stats_list[0][i]['roi_x']) + '=' +str( (stats_list[0][i]['roi_yy']-stats_list[0][i]['roi_y'])*(stats_list[0][i]['roi_xx']-stats_list[0][i]['roi_x']) )
              )
        print('img#'.ljust(5)      + '\t'
              + 'mean'.ljust(20)   + '\t'
              + 'std'.ljust(20)    + '\t'
              + 'min'.ljust(20)    + '\t'
              + 'max'.ljust(20)    + '\t'
              + 'median'.ljust(20) + '\t'
              )

        for axe_num in np.arange(len(axes_list)):
            if np.shape(stats_list[axe_num][i]['mean']): #チャンネルが存在する画像の場合
                print_array = np.concatenate([stats_list[axe_num][i]['mean'][:,np.newaxis],
                                              stats_list[axe_num][i]['std'][:,np.newaxis],
                                              stats_list[axe_num][i]['min'][:,np.newaxis],
                                              stats_list[axe_num][i]['max'][:,np.newaxis],
                                              stats_list[axe_num][i]['med'][:,np.newaxis],
                                              ],axis=1)

                print( str(axe_num).ljust(5) + '\t'
                       +'\n     \t'.join('\t'.join(str(x).ljust(20) for x in y) for y in print_array)
                      )

            else:
                print( str(axe_num).ljust(5) + '\t'
                      +str(stats_list[axe_num][i]['mean']).ljust(20)+'\t'
                      +str(stats_list[axe_num][i]['std' ]).ljust(20)+'\t'
                      +str(stats_list[axe_num][i]['min' ]).ljust(20)+'\t'
                      +str(stats_list[axe_num][i]['max' ]).ljust(20)+'\t'
                      +str(stats_list[axe_num][i]['med' ]).ljust(20)+'\t'
                      )
        print('')
    pass


def q_hotkey__roistats2(fig, event, hotkey_use):
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height()>0]
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe,matplotlib.axes._subplots.Subplot)]

    stats_list = np.zeros((len(axes_list),len(active_roi_keys))).tolist()
    for axe_num in np.arange(len(axes_list)):
        axes_img = axes_list[axe_num].images[0].get_array().data
        for i,rk in enumerate(active_roi_keys):
            tgt_roi = hotkey_use['roi_list'][axe_num][rk]
            roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
            roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
            roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y,0,hw[0]), np.clip(roi_yy,0,hw[0]), np.clip(roi_x,0,hw[1]), np.clip(roi_xx,0,hw[1])
            roi_img = axes_img[roi_y:roi_yy,roi_x:roi_xx,...]

            stats_list[axe_num][i] = {'roi_x':roi_x,'roi_xx':roi_xx,
                                      'roi_y':roi_y,'roi_yy':roi_yy,
                                      'mean':np.nanmean(roi_img,axis=(0,1)),
                                      'std' :np.nanstd(roi_img,axis=(0,1)),
                                      'min' :np.nanmin(roi_img,axis=(0,1)),
                                      'max' :np.nanmax(roi_img,axis=(0,1)),
                                      'med' :np.nanmedian(roi_img,axis=(0,1)),}


    print('======================== ROI stats ======================== ')
    headers = ['img No.','mean (imgN/img0)','std (imgN/img0)','min (imgN/img0)','max (imgN/img0)','median (imgN/img0)']
    for i,rk in enumerate(active_roi_keys):
        print('ROI key='+ rk
              +'    pos=['+str(stats_list[0][i]['roi_y'])+':'+str(stats_list[0][i]['roi_yy'])+', '+str(stats_list[0][i]['roi_x'])+':'+str(stats_list[0][i]['roi_xx'])+']'
              +'    pixel num=' + str(stats_list[0][i]['roi_yy']-stats_list[0][i]['roi_y'])+'*'+str(stats_list[0][i]['roi_xx']-stats_list[0][i]['roi_x']) + '=' +str( (stats_list[0][i]['roi_yy']-stats_list[0][i]['roi_y'])*(stats_list[0][i]['roi_xx']-stats_list[0][i]['roi_x']) )
              )

        print_list = []
        # print_list2= []
        for axe_num in np.arange(len(axes_list)):
            if np.shape(stats_list[axe_num][i]['mean']): #チャンネルが存在する画像の場合
                for ch in np.arange(np.shape(stats_list[axe_num][i]['mean'])):
                    print_list.append([str(axe_num)+' ('+str(ch)+'ch)',
                                       '{:<12.5f}'.format(stats_list[axe_num][i]['mean'][ch]) +' ('+'{:<.2f}'.format(stats_list[axe_num][i]['mean'][ch]/stats_list[0][i]['mean'][ch]*100)  + '%)',
                                       '{:<12.5f}'.format(stats_list[axe_num][i]['std'][ch])  +' ('+'{:<.2f}'.format(stats_list[axe_num][i]['std'][ch] /stats_list[0][i]['std'][ch] *100)  + '%)',
                                       '{:<12.5f}'.format(stats_list[axe_num][i]['min'][ch])  +' ('+'{:<.2f}'.format(stats_list[axe_num][i]['min'][ch] /stats_list[0][i]['min'][ch] *100)  + '%)',
                                       '{:<12.5f}'.format(stats_list[axe_num][i]['max'][ch])  +' ('+'{:<.2f}'.format(stats_list[axe_num][i]['max'][ch] /stats_list[0][i]['max'][ch] *100)  + '%)',
                                       '{:<12.5f}'.format(stats_list[axe_num][i]['med'][ch])  +' ('+'{:<.2f}'.format(stats_list[axe_num][i]['med'][ch] /stats_list[0][i]['med'][ch] *100)  + '%)',
                                       ])

            else:
                print_list.append([str(axe_num),
                                   '{:<12.5f}'.format(stats_list[axe_num][i]['mean']) + ' (' + '{:<.2f}'.format(stats_list[axe_num][i]['mean']/stats_list[0][i]['mean'] * 100) + '%)',
                                   '{:<12.5f}'.format(stats_list[axe_num][i]['std'])  + ' (' + '{:<.2f}'.format(stats_list[axe_num][i]['std'] /stats_list[0][i]['std'] * 100)  + '%)',
                                   '{:<12.5f}'.format(stats_list[axe_num][i]['min'])  + ' (' + '{:<.2f}'.format(stats_list[axe_num][i]['min'] /stats_list[0][i]['min'] * 100)  + '%)',
                                   '{:<12.5f}'.format(stats_list[axe_num][i]['max'])  + ' (' + '{:<.2f}'.format(stats_list[axe_num][i]['max'] /stats_list[0][i]['max'] * 100)  + '%)',
                                   '{:<12.5f}'.format(stats_list[axe_num][i]['med'])  + ' (' + '{:<.2f}'.format(stats_list[axe_num][i]['med'] /stats_list[0][i]['med'] * 100)  + '%)',
                                   ])

        table_print(print_list,headers)
        print('')
        # table_print(print_list2,headers)
        # print('')

    pass

def q_hotkey__roipixval(fig, event, hotkey_use):
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height() > 0]
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]

    subplot_h,subplot_w = axes_list[0].get_gridspec().nrows, axes_list[0].get_gridspec().ncols

    for i,rk in enumerate(active_roi_keys):
        tgt_roi = hotkey_use['roi_list'][0][rk]
        roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
        roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
        roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)

        # 表示範囲制限
        max_range = 25 # 表示最大範囲を25x25pixに制限
        roi_yy,roi_xx = np.clip(roi_yy,None,roi_y+max_range), np.clip(roi_xx,None,roi_x+max_range)

        ana_fig = plt.figure()
        for j,axe in enumerate(axes_list):
            axes_img = axe.images[0].get_array().data
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y,0,hw[0]), np.clip(roi_yy,0,hw[0]), np.clip(roi_x,0,hw[1]), np.clip(roi_xx,0,hw[1])
            part_img = axes_img[roi_y:roi_yy, roi_x:roi_xx]


            sps = axe.get_subplotspec()
            subplot_x = sps.colspan[0]
            subplot_y = sps.rowspan[0]

            img_sub_ax = ana_fig.add_subplot(subplot_h, subplot_w, subplot_y*subplot_w+subplot_x+1, picker = True)
            img_sub_im = img_sub_ax.imshow(part_img, interpolation='nearest', cmap=axe.images[0].get_cmap(),
                                           extent=[roi_x-0.5, roi_xx-0.5, roi_yy-0.5, roi_y-0.5 ],
                                           aspect='equal')
            img_sub_im.set_clim(axe.images[0].get_clim())

            if np.ndim(axes_img) == 2:
                v_a = {0: 'bottom', 1: 'top'}
                for (iy, ix), val in np.ndenumerate(part_img):
                    img_sub_ax.text(ix+roi_x, iy+roi_y, '{0:.3f}'.format(val), ha='center', va=v_a[ix%2], color='#F661F6', fontweight='bold', fontsize=9)
            elif np.ndim(axes_img) == 3:
                v_a = [-0.25,0,0.25]
                tc = ['#FF7171','#9DDD15','#57B8FF',]
                for c in np.arange(np.shape(part_img)[2]):
                    for (iy, ix), val in np.ndenumerate(part_img[:,:,c]):
                        img_sub_ax.text(ix+roi_x, iy+roi_y+v_a[c], '{0:.3f}'.format(val), ha='center', va='center', color=tc[c], fontweight='bold', fontsize=8)

        ana_fig = q_addon(ana_fig)
        ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
        ana_fig.show()

    pass


# def q_hotkey__roiscatter(fig, event, hotkey_use):
#

def q_hotkey__noise_analyze(fig, event, hotkey_use):
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height() > 0]
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]

    for j, axe in enumerate(axes_list):
        mean_list = []
        var_list = []
        for i, rk in enumerate(active_roi_keys):
            tgt_roi = hotkey_use['roi_list'][0][rk]
            roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
            roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
            roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)

            axes_img = axe.images[0].get_array().data
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y, 0, hw[0]), np.clip(roi_yy, 0, hw[0]), np.clip(roi_x, 0, hw[1]), np.clip(roi_xx, 0, hw[1])
            part_img = axes_img[roi_y:roi_yy, roi_x:roi_xx]

            mean_list.append(np.nanmean(part_img))
            var_list.append(np.nanvar(part_img))

        grad, intercept = weighted_least_squares(in_x=np.array(mean_list), in_y=np.array(var_list), weight=1/np.array(mean_list))
        plotq([np.array(mean_list),np.array([0,np.max(mean_list)])], [np.array(var_list),np.array([intercept,np.max(mean_list)*grad+intercept])],
              linewidth_list=[0,3],marker_list=['o',''])
        print('grad='+str(grad), 'intercept='+str(intercept))
    pass

def q_hotkey__colorchecker(fig, event, hotkey_use):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    for j, axe in enumerate(axes_list):
        tgt_roi = hotkey_use['roi_list'][j]['alt+0']
        roi_x, roi_y = np.array(tgt_roi.get_xy()) + 0.5
        roi_xx, roi_yy = roi_x + tgt_roi.get_width(), roi_y + tgt_roi.get_height()
        roi_y, roi_yy, roi_x, roi_xx = int(roi_y), int(roi_yy), int(roi_x), int(roi_xx)

        rk = ['1', '2', '3', '4', '5', '6',
              '7', '8', '9', '0','ctrl+1', 'ctrl+2',
              'ctrl+3', 'ctrl+4', 'ctrl+5', 'ctrl+6', 'ctrl+7', 'ctrl+8',
              'ctrl+9', 'ctrl+0', 'alt+1', 'alt+2', 'alt+3', 'alt+4']
        xn = (roi_xx-roi_x)/(40*6+6*5)
        yn = (roi_yy-roi_y)/(40*4+6*3)
        print(xn,yn)
        for h in np.arange(4):
            for w in np.arange(6):
                # [roi[event.key].set(xy=(-0.5, -0.5), width=0, height=0) for roi in hotkey_use['roi_list']]
                # [roi_text[event.key].set(x=-0.5, y=-0.5, alpha=0) for roi_text in hotkey_use['roi_text_list']]
                hotkey_use['roi_list'][j][rk[h*6+w]].set(xy=(roi_x+xn*w*46+xn*0.2*40, roi_y+yn*h*46+yn*0.2*40), width=40*xn-xn*0.4*40, height=40*yn-yn*0.4*40)
                hotkey_use['roi_text_list'][j][rk[h*6+w]].set(x=roi_x+xn*w*46+xn*0.2*40, y=roi_y+yn*h*46+yn*0.2*40, alpha=1)


# ############################################################################ hist
def q_hotkey_util__hist(fig, event, hotkey_use):
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height() > 0]
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]

    subplot_h,subplot_w = axes_list[0].get_gridspec().nrows, axes_list[0].get_gridspec().ncols
    for i,rk in enumerate(active_roi_keys):
        tgt_roi = hotkey_use['roi_list'][0][rk]
        roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
        roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
        roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)

        temp_input = [[] for i in range(subplot_h)]
        temp_inter = [[] for i in range(subplot_h)]
        temp_c     = [[] for i in range(subplot_h)]
        temp_ec    = [[] for i in range(subplot_h)]
        for j,axe in enumerate(axes_list):
            axes_img = axe.images[0].get_array().data
            hw = np.shape(axes_img)
            roi_y, roi_yy, roi_x, roi_xx = np.clip(roi_y,0,hw[0]), np.clip(roi_yy,0,hw[0]), np.clip(roi_x,0,hw[1]), np.clip(roi_xx,0,hw[1])
            part_img = axes_img[roi_y:roi_yy, roi_x:roi_xx]

            sps = axe.get_subplotspec()
            subplot_x = sps.colspan[0]
            subplot_y = sps.rowspan[0]

            if np.ndim(axes_img)==2:
                temp_input[subplot_y].append(part_img)
                temp_c[subplot_y].append(kutinawa_color[i])
                temp_ec[subplot_y].append('black')

                if np.all(axes_img.astype(int).astype(float)==axes_img):
                    temp_inter[subplot_y].append(1)
                else:
                    temp_inter[subplot_y].append((axe.images[0].get_clim()[1] - axe.images[0].get_clim()[0]) / 256)

            elif np.ndim(axes_img)==3:
                tc = ['#FF7171', '#9DDD15', '#57B8FF', ]
                for ch in np.arange(np.shape(axes_img)[2]):
                    temp_input[subplot_y].append(part_img[:,:,ch])
                    temp_inter[subplot_y].append((np.nanmax(axes_img)-np.nanmin(axes_img))/256)
                    temp_c[subplot_y].append(kutinawa_color[i])
                    temp_ec[subplot_y].append(tc[ch])

        histq(input_list = temp_input,
              interval_list=temp_inter,
              label_list=None,
              alpha_list = 1,
              edgecolor_list = temp_ec,
              color_list = temp_c,
              histtype='bar',
              overlay=False)
    pass

# ############################################################################ lineprof
def q_hotkey_util__lineprof(fig, event, hotkey_use, mode):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    img_xlim = axes_list[0].get_xlim()
    img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
    img_ylim = axes_list[0].get_ylim()
    img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

    if mode=='H':
        mouse_y = int(event.ydata + 0.5)
        line_img_xpos = img_xlim
        line_img_ypos = [mouse_y+1, mouse_y]
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

    ana_fig = plt.figure()
    plot_sub_ax = ana_fig.add_subplot(2,len(axes_list),(1,len(axes_list)),picker=True)

    for cn,axe in enumerate(axes_list):
        axes_img = axe.images[0].get_array().data
        hw = np.shape(axes_img)
        x_pos = [np.clip(line_img_xpos[0], 0, hw[1]), np.clip(line_img_xpos[1], 0, hw[1])]
        y_pos = [np.clip(line_img_ypos[1], 0, hw[0]), np.clip(line_img_ypos[0], 0, hw[0])]
        line_img,pos = axes_img[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1]], [x_pos, y_pos]

        ####################### plot
        m0 = int(np.mod(cn,4)>0)
        m1 = int(np.mod(cn,4)>1)
        m2 = int(np.mod(cn,4)>2)
        linestyle = (0, (5,m0,m0,m0,m1,m1,m2,m2))

        if np.ndim(line_img)==2:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(line_img),
                             label=cn)
        elif np.ndim(line_img)==3:
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(line_img[:,:,0]),
                             label=str(cn)+'(0ch)',
                             color=(0.75, np.clip(cn/len(axes_list)-0.5,0,1), np.clip(-cn/len(axes_list)+0.5,0,1)),
                             linestyle=linestyle )
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(line_img[:,:,1]),
                             label=str(cn)+'(1ch)',
                             color=(np.clip(cn/len(axes_list)-0.5,0,1), 0.75, np.clip(-cn/len(axes_list)+0.5,0,1)),
                             linestyle=linestyle )
            plot_sub_ax.plot(np.arange(pos[line_plot_idx][0],pos[line_plot_idx][1]),
                             np.squeeze(line_img[:,:,2]),
                             label=str(cn)+'(2ch)',
                             color=(np.clip(cn/len(axes_list)-0.5,0,1), np.clip(-cn/len(axes_list)+0.5,0,1), 0.75),
                             linestyle=linestyle )

        ####################### image
        pos2 = [[np.clip(img_xlim[0], 0, hw[1]),np.clip(img_xlim[1], 0, hw[1])], [np.clip(img_ylim[1], 0, hw[0]), np.clip(img_ylim[0], 0, hw[0])]]
        part_img   = axes_img[pos2[1][0]:pos2[1][1], pos2[0][0]:pos2[0][1]]
        img_sub_ax = ana_fig.add_subplot(2,len(axes_list),len(axes_list)+cn+1,picker=True)
        img_sub_ax.axline(xy1=ax_line_xy1,xy2=ax_line_xy2,color='pink')
        img_sub_ax.axline(xy1=ax_line_xy3,xy2=ax_line_xy4,color='pink')
        img_sub_im = img_sub_ax.imshow(part_img,interpolation='nearest', cmap=axe.images[0].get_cmap(),
                                       extent=[pos2[0][0]-0.5,pos2[0][1]-0.5,pos2[1][1]-0.5,pos2[1][0]-0.5],
                                       aspect='equal')
        img_sub_im.set_clim(axe.images[0].get_clim())

    plot_sub_ax.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
    ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    ana_fig.show()
    pass


def q_hotkey__lineprofH(fig, event, hotkey_use):
    if event.inaxes:
        q_hotkey_util__lineprof(fig, event, hotkey_use, mode='H')

def q_hotkey__lineprofV(fig, event, hotkey_use):
    if event.inaxes:
        q_hotkey_util__lineprof(fig, event, hotkey_use, mode='V')


def q_hotkey_util__meanlineprof(fig, event, hotkey_use, mode):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    active_roi_keys = [rk for rk in hotkey_use['roi_list'][0].keys() if hotkey_use['roi_list'][0][rk].get_height()>0]

    img_xlim = axes_list[0].get_xlim()
    img_xlim = [int(img_xlim[0]+ 0.5), int(img_xlim[1]+ 0.5)]
    img_ylim = axes_list[0].get_ylim()
    img_ylim = [int(img_ylim[0]+ 0.5), int(img_ylim[1]+ 0.5)]

    for i,rk in enumerate(active_roi_keys):
        tgt_roi = hotkey_use['roi_list'][0][rk]
        roi_x,roi_y = np.array(tgt_roi.get_xy())+0.5
        roi_xx,roi_yy = roi_x+tgt_roi.get_width(), roi_y+tgt_roi.get_height()
        roi_y,roi_yy,roi_x,roi_xx = int(roi_y),int(roi_yy),int(roi_x),int(roi_xx)

        if mode == 'roiH':
            line_img_xpos = img_xlim
            line_img_ypos = [roi_yy, roi_y]
            line_plot_idx = 0
            ax_line_xy1 = (img_xlim[0], roi_y-0.5)
            ax_line_xy2 = (img_xlim[1], roi_y-0.5)
            ax_line_xy3 = (img_xlim[0], roi_yy-0.5)
            ax_line_xy4 = (img_xlim[1], roi_yy-0.5)
        elif mode == 'roiV':
            line_img_xpos = [roi_x, roi_xx]
            line_img_ypos = img_ylim
            line_plot_idx = 1
            ax_line_xy1 = (roi_x-0.5, img_ylim[0])
            ax_line_xy2 = (roi_x-0.5, img_ylim[1])
            ax_line_xy3 = (roi_xx-0.5, img_ylim[0])
            ax_line_xy4 = (roi_xx-0.5, img_ylim[1])

        ana_fig = plt.figure()
        plot_sub_ax = ana_fig.add_subplot(2,len(axes_list),(1,len(axes_list)),picker=True)

        for cn,axe in enumerate(axes_list):
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
                                 np.squeeze(np.mean(line_img,axis=line_plot_idx)),
                                 label=cn)
            elif np.ndim(line_img) == 3:
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 0],axis=line_plot_idx)),
                                 label=str(cn) + '(0ch)',
                                 color=(0.75, np.clip(cn / len(axes_list) - 0.5, 0, 1), np.clip(-cn / len(axes_list) + 0.5, 0, 1)),
                                 linestyle=linestyle)
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 1],axis=line_plot_idx)),
                                 label=str(cn) + '(1ch)',
                                 color=(np.clip(cn / len(axes_list) - 0.5, 0, 1), 0.75, np.clip(-cn / len(axes_list) + 0.5, 0, 1)),
                                 linestyle=linestyle)
                plot_sub_ax.plot(np.arange(pos[line_plot_idx][0], pos[line_plot_idx][1]),
                                 np.squeeze(np.mean(line_img[:, :, 2],axis=line_plot_idx)),
                                 label=str(cn) + '(2ch)',
                                 color=(np.clip(cn / len(axes_list) - 0.5, 0, 1), np.clip(-cn / len(axes_list) + 0.5, 0, 1), 0.75),
                                 linestyle=linestyle)

            ####################### image
            pos2 = [[np.clip(img_xlim[0], 0, hw[1]), np.clip(img_xlim[1], 0, hw[1])], [np.clip(img_ylim[1], 0, hw[0]), np.clip(img_ylim[0], 0, hw[0])]]
            part_img = axes_img[pos2[1][0]:pos2[1][1], pos2[0][0]:pos2[0][1]]
            img_sub_ax = ana_fig.add_subplot(2, len(axes_list), len(axes_list) + cn + 1, picker=True)
            img_sub_ax.axline(xy1=ax_line_xy1, xy2=ax_line_xy2, color='pink')
            img_sub_ax.axline(xy1=ax_line_xy3, xy2=ax_line_xy4, color='pink')
            img_sub_im = img_sub_ax.imshow(part_img, interpolation='nearest', cmap=axe.images[0].get_cmap(),
                                           extent=[pos2[0][0] - 0.5, pos2[0][1] - 0.5, pos2[1][1] - 0.5, pos2[1][0] - 0.5],
                                           aspect='equal')
            img_sub_im.set_clim(axe.images[0].get_clim())

        plot_sub_ax.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={"weight": "bold", "size": "large"})
        ana_fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)  # 表示範囲調整
        ana_fig.show()


def q_hotkey__lineprofHmean(fig, event, hotkey_use):
    q_hotkey_util__meanlineprof(fig, event, hotkey_use,mode='roiH')

def q_hotkey__lineprofVmean(fig, event, hotkey_use):
    q_hotkey_util__meanlineprof(fig, event, hotkey_use,mode='roiV')

# def q_hotkey__posprint(event,config_list):
#     if config_list[0].roi:# 画像
#         view_pos = [config_list[0].get_xlim_pos_for_img(),config_list[0].get_ylim_pos_for_img()]
#         mouse_pos = [int(event.xdata + 0.5),int(event.ydata + 0.5)]
#         roi_pos = config_list[0].get_roi_pos()
#         roi_pos = [roi_pos[0:2],roi_pos[2:]]
#         print(  ('view[[xs,xe],[ys,ye]]:'+str(view_pos)).ljust(50)+'\t'
#                 +('mouse[x,y]:'+str(mouse_pos)).ljust(40)+'\t'
#                 +('roi[[xs,xe],[ys,ye]]:'+str(roi_pos)).ljust(50)+'\t')
#     else:
#         view_pos = [config_list[0].get_xlim_pos(),config_list[0].get_ylim_pos()]
#         mouse_pos = [event.xdata ,event.ydata]
#         print(  ('view[[xs,xe],[ys,ye]]:'+str(view_pos)).ljust(120)+'\t'
#                 +('mouse[x,y]:'+str(mouse_pos)).ljust(80)+'\t')
#     pass


def q_hotkey__zoomplot(fig, event, hotkey_use):
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe,matplotlib.axes._subplots.Subplot)]
    yud_mode = int(axes_list[0].get_ylim()[0] > axes_list[0].get_ylim()[1])

    for axe in axes_list:
        axes_img = axe.images[0].get_array().data

        for rk in np.arange(10):
            rk = str(rk)
            crk = 'ctrl+' + rk
            if hotkey_use['roi_list'][0][rk].get_height() > 0 and hotkey_use['roi_list'][0][crk].get_height() > 0:
                target_area = [hotkey_use['roi_list'][0][rk].get_x(), hotkey_use['roi_list'][0][rk].get_x()+hotkey_use['roi_list'][0][rk].get_width(),
                               hotkey_use['roi_list'][0][rk].get_y(), hotkey_use['roi_list'][0][rk].get_y()+hotkey_use['roi_list'][0][rk].get_height(),]

                ylim = [axe.get_ylim()[np.abs(0-yud_mode)], axe.get_ylim()[np.abs(1-yud_mode)]]
                xlim = axe.get_xlim()
                y_range = ylim[1]-ylim[0]
                x_range = xlim[1]-xlim[0]
                if yud_mode==1:
                    plot_area = [(hotkey_use['roi_list'][0][crk].get_x()-xlim[0])/x_range,1-(hotkey_use['roi_list'][0][crk].get_y()-ylim[0])/y_range-hotkey_use['roi_list'][0][crk].get_height()/y_range,
                                 hotkey_use['roi_list'][0][crk].get_width()/x_range, hotkey_use['roi_list'][0][crk].get_height()/y_range,]
                else:
                    plot_area = [(hotkey_use['roi_list'][0][crk].get_x()-xlim[0])/x_range,(hotkey_use['roi_list'][0][crk].get_y()-ylim[0])/y_range,
                                 hotkey_use['roi_list'][0][crk].get_width()/x_range, hotkey_use['roi_list'][0][crk].get_height()/y_range,]


                axins = axe.inset_axes(plot_area)
                axins.imshow(axes_img,origin='upper')
                axins.set_xlim(target_area[0], target_area[1])
                axins.set_ylim(target_area[2+np.abs(0-yud_mode)], target_area[2+np.abs(1-yud_mode)])
                axins.set_ylim(target_area[2], target_area[3])

                axins.set_xticklabels('')
                axins.set_yticklabels('')
                axe.indicate_inset_zoom(axins)

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
def q_addon(fig,keyboard_dict=None,roi_coordinate='int'):

    plt.interactive(False)

    # hotkey
    def q_hotkey__dummy(fig, event, hotkey_use):
        pass

    def q_hotkey__png_save(fig, event, hotkey_use):
        fig.savefig(datetime.datetime.now().strftime('q-%Y_%m_%d_%H_%M_%S') + '.png', bbox_inches='tight')
        pass

    def q_hotkey__reset(fig, event, hotkey_use):
        axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
        [axe.set_xlim(hotkey_use['init_xy_pos'][0][0], hotkey_use['init_xy_pos'][0][1]) for axe in axes_list]
        [axe.set_ylim(hotkey_use['init_xy_pos'][1][0], hotkey_use['init_xy_pos'][1][1]) for axe in axes_list]
        pass

    def q_hotkey__roiset_int(fig, event, hotkey_use):
        if hotkey_use['mouse_mode'] == 'ROI':
            pos = (np.array(hotkey_use['ca_rect'].extents) + 0.5).astype(int)
            if pos[3] - pos[2] == 0 and pos[1] - pos[0] == 0:
                [roi[event.key].set(xy=(-0.5, -0.5), width=0, height=0) for roi in hotkey_use['roi_list']]
                [roi_text[event.key].set(x=-0.5, y=-0.5, alpha=0) for roi_text in hotkey_use['roi_text_list']]
            else:
                pos = pos + np.array([-0.5, 0.5, -0.5, 0.5])
                [roi[event.key].set(xy=(pos[0], pos[2]), height=pos[3] - pos[2], width=pos[1] - pos[0]) for roi in hotkey_use['roi_list']]
                [roi_text[event.key].set(x=pos[0], y=pos[2], alpha=1) for roi_text in hotkey_use['roi_text_list']]
        pass

    def q_hotkey__roiset_float(fig, event, hotkey_use):
        if hotkey_use['mouse_mode'] == 'ROI':
            pos = np.array(hotkey_use['ca_rect'].extents)
            if pos[3] - pos[2] == 0 and pos[1] - pos[0] == 0:
                [roi[event.key].set(xy=(-0.5, -0.5), width=0, height=0) for roi in hotkey_use['roi_list']]
                [roi_text[event.key].set(x=-0.5, y=-0.5, alpha=0) for roi_text in hotkey_use['roi_text_list']]
            else:
                [roi[event.key].set(xy=(pos[0], pos[2]), height=pos[3] - pos[2], width=pos[1] - pos[0]) for roi in hotkey_use['roi_list']]
                [roi_text[event.key].set(x=pos[0], y=pos[2], alpha=1) for roi_text in hotkey_use['roi_text_list']]
        pass

    if keyboard_dict==None:
        keyboard_dict=dict()

    if roi_coordinate=='int':
        keyboard_dict.update([('P',q_hotkey__png_save),
                              ('tab',q_hotkey__reset),
                              ('1',q_hotkey__roiset_int),('2', q_hotkey__roiset_int),('3', q_hotkey__roiset_int),('4', q_hotkey__roiset_int),('5', q_hotkey__roiset_int),('6', q_hotkey__roiset_int),('7', q_hotkey__roiset_int),('8', q_hotkey__roiset_int),('9', q_hotkey__roiset_int),('0', q_hotkey__roiset_int),
                              ('ctrl+1', q_hotkey__roiset_int),('ctrl+2', q_hotkey__roiset_int),('ctrl+3', q_hotkey__roiset_int),('ctrl+4', q_hotkey__roiset_int),('ctrl+5', q_hotkey__roiset_int),('ctrl+6', q_hotkey__roiset_int),('ctrl+7', q_hotkey__roiset_int),('ctrl+8', q_hotkey__roiset_int),('ctrl+9', q_hotkey__roiset_int),('ctrl+0', q_hotkey__roiset_int),
                              ('alt+1', q_hotkey__roiset_int),('alt+2', q_hotkey__roiset_int),('alt+3', q_hotkey__roiset_int),('alt+4', q_hotkey__roiset_int),('alt+5', q_hotkey__roiset_int),('alt+6', q_hotkey__roiset_int),('alt+7', q_hotkey__roiset_int),('alt+8', q_hotkey__roiset_int),('alt+9', q_hotkey__roiset_int),('alt+0', q_hotkey__roiset_int),
                              ])
    elif roi_coordinate=='float':
        keyboard_dict.update([('P',q_hotkey__png_save),
                              ('tab',q_hotkey__reset),
                              ('1',q_hotkey__roiset_float),('2', q_hotkey__roiset_float),('3', q_hotkey__roiset_float),('4', q_hotkey__roiset_float),('5', q_hotkey__roiset_float),('6', q_hotkey__roiset_float),('7', q_hotkey__roiset_float),('8', q_hotkey__roiset_float),('9', q_hotkey__roiset_float),('0', q_hotkey__roiset_float),
                              ('ctrl+1', q_hotkey__roiset_float),('ctrl+2', q_hotkey__roiset_float),('ctrl+3', q_hotkey__roiset_float),('ctrl+4', q_hotkey__roiset_float),('ctrl+5', q_hotkey__roiset_float),('ctrl+6', q_hotkey__roiset_float),('ctrl+7', q_hotkey__roiset_float),('ctrl+8', q_hotkey__roiset_float),('ctrl+9', q_hotkey__roiset_float),('ctrl+0', q_hotkey__roiset_float),
                              ('alt+1', q_hotkey__roiset_float),('alt+2', q_hotkey__roiset_float),('alt+3', q_hotkey__roiset_float),('alt+4', q_hotkey__roiset_float),('alt+5', q_hotkey__roiset_float),('alt+6', q_hotkey__roiset_float),('alt+7', q_hotkey__roiset_float),('alt+8', q_hotkey__roiset_float),('alt+9', q_hotkey__roiset_float),('alt+0', q_hotkey__roiset_float),
                              ])

    def select_callback(eclick, erelease):
        pass

    event_dict = {'btn_prs_x': 0, 'btn_prs_y': 0,
                  'btn_prs_xdata': 0, 'btn_prs_ydata': 0,
                  'key_prs_xdata': 0, 'key_prs_ydata': 0,
                  'btn_prs_inaxes_flag': False,
                  'btn_prs_colorbar_flag': False,
                  'sca_num': 0,'ca_change':True
                  }
    hotkey_use = {'init_xy_pos':[[]],
                  'mouse_mode':'',
                  'ca_rect':None,
                  'roi_list':None,
                  'roi_text_list': None
                  }

    #マウスモード初期化
    mouse_mode_dict = {'n': 'Normal', 'r': 'ROI'}
    mouse_mode_color_dict = {'Normal': 'white', 'ROI': 'red'}
    hotkey_use['mouse_mode'] = mouse_mode_dict['n']

    # axesリスト取得
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
    axes_dict = dict( zip(axes_list, np.arange(len(axes_list)).tolist() ) )
    cbar_list = [axe for axe in fig.get_axes() if not isinstance(axe, matplotlib.axes._subplots.Subplot)]
    cbar_dict = dict( zip(cbar_list, np.arange(len(cbar_list)).tolist() ) )

    # 軸範囲を取得、プロット領域の位置初期化
    yud_mode = int(axes_list[0].get_ylim()[0] > axes_list[0].get_ylim()[1])
    xlims = np.array([axe.get_xlim() for axe in axes_list])
    ylims = np.array([axe.get_ylim() for axe in axes_list])
    if yud_mode==0:
        hotkey_use['init_xy_pos'] = [[np.min(xlims), np.max(xlims)],
                                     [np.min(ylims), np.max(ylims)]]
    elif yud_mode==1:
        hotkey_use['init_xy_pos'] = [[np.min(xlims), np.max(xlims)],
                                     [np.max(ylims), np.min(ylims)]]
    [axe.set_xlim(hotkey_use['init_xy_pos'][0][0], hotkey_use['init_xy_pos'][0][1]) for axe in axes_list]
    [axe.set_ylim(hotkey_use['init_xy_pos'][1][0], hotkey_use['init_xy_pos'][1][1]) for axe in axes_list]

    # rect初期化
    rect_list = [RectangleSelector(axe, select_callback,
                                   useblit=True,
                                   button=[1],  # disable right & middle button
                                   minspanx=5, minspany=5,
                                   spancoords='pixels',
                                   interactive=True,
                                   state_modifier_keys={"square":'ctrl'},
                                   props=dict(
                                       facecolor='pink',
                                       edgecolor='white',
                                       alpha=1,
                                       fill=False)
                                   ) for axe in axes_list]
    hotkey_use['ca_rect'] = rect_list[0]

    # ROI初期化
    rk = ['1','2','3','4','5','6','7','8','9','0',
          'ctrl+1','ctrl+2','ctrl+3','ctrl+4','ctrl+5','ctrl+6','ctrl+7','ctrl+8','ctrl+9','ctrl+0',
          'alt+1','alt+2','alt+3','alt+4','alt+5','alt+6','alt+7','alt+8','alt+9','alt+0']
    roi_list = [dict() for i in np.arange(len(axes_list))] # axes分のリストに、キーボード入力をkey・patches.Rectangleをvalにした辞書が入っている
    roi_text_list = [dict() for i in np.arange(len(axes_list))]
    for i_axe in np.arange(len(axes_list)):
        for i_r in np.arange(30):
            roi_list[i_axe][rk[i_r]] = patches.Rectangle(xy=(-0.5, -0.5), width=0, height=0, ec=kutinawa_color[i_r], fill=False)
            axes_list[i_axe].add_patch(roi_list[i_axe][rk[i_r]])
            roi_text_list[i_axe][rk[i_r]] = axes_list[i_axe].text(-0.5, -0.5, s=rk[i_r],c=kutinawa_color[i_r], ha='right', va='top', alpha=0, weight='bold')

    hotkey_use['roi_list'] = roi_list
    hotkey_use['roi_text_list'] = roi_text_list

    def mouse_key_event(fig):
        def button_press(event):
            event_dict['btn_prs_shift_flag']  =(event.key=="shift")
            event_dict['btn_prs_inaxes_flag'] = isinstance(event.inaxes, matplotlib.axes._subplots.Subplot)
            event_dict['btn_prs_colorbar_flag'] = event.inaxes if event.inaxes and (not event_dict['btn_prs_inaxes_flag']) else False  #ここの判定なんとかしたい
            event_dict['btn_prs_x'],    event_dict['btn_prs_y']     = event.x, event.y
            event_dict['btn_prs_xdata'],event_dict['btn_prs_ydata'] = event.xdata, event.ydata

            if event_dict['btn_prs_inaxes_flag']:
                if (event.dblclick) and (event.button == 1):
                    # ダブルクリックで最も大きい画像に合わせて表示領域リセット
                    [axe.set_xlim(hotkey_use['init_xy_pos'][0][0], hotkey_use['init_xy_pos'][0][1]) for axe in axes_list]
                    [axe.set_ylim(hotkey_use['init_xy_pos'][1][0], hotkey_use['init_xy_pos'][1][1]) for axe in axes_list]
                    pass
                else:
                    # クリックした画像を着目画像(current axes)に指定
                    [axe.spines['bottom'].set(color="black", linewidth=1) for axe in axes_list]
                    fig.sca(event.inaxes)
                    event.inaxes.spines['bottom'].set(color="#FF4500", linewidth=6)
                    event_dict['ca_change'] = event_dict['sca_num']!=axes_dict.get(event.inaxes, 0)
                    event_dict['sca_num'] = axes_dict.get(event.inaxes, 0)
                    hotkey_use['ca_rect'] = rect_list[event_dict['sca_num']]

                if hotkey_use['mouse_mode']=='Normal':
                    if event.key=="shift":
                        [rect.set_visible(True) for rect in rect_list]
                    else:
                        [rect.set_visible(False) for rect in rect_list]
                elif hotkey_use['mouse_mode']=='ROI':
                    [rect_list[i].set_visible(False) for i in set(np.arange(len(rect_list))).difference({event_dict['sca_num']})]
                    rect_list[event_dict['sca_num']].set_visible(True)

            pass

        def key_press(event):
            event_dict['key_prs_xdata']=event.xdata
            event_dict['key_prs_ydata']=event.ydata

            # nonlocal mouse_mode
            conv_mouse_mode = hotkey_use['mouse_mode']
            temp = mouse_mode_dict.get(event.key,None)
            if temp:
                if conv_mouse_mode==temp:
                    hotkey_use['mouse_mode'] = 'Normal'
                else:
                    hotkey_use['mouse_mode'] = temp
                [rs.artists[0].set_edgecolor(mouse_mode_color_dict[hotkey_use['mouse_mode']]) for rs in rect_list]
            else:
                hotkey_use['mouse_mode'] = conv_mouse_mode

            # rectの座標とかを知りたいときは、rect_list[0].extents
            keyboard_dict.get(event.key, q_hotkey__dummy)(fig, event, hotkey_use)
            pass

        def key_release(event):
            fig.canvas.manager.set_window_title('Figure'+str(fig.number)+' [q_addon : Mouse mode = "'+hotkey_use['mouse_mode']+'"]')
            if event.key in ['shift','alt','ctrl']:
                pass
            else:
                fig.canvas.draw()
            pass

        def button_release(event):
            if hotkey_use['mouse_mode']=='Normal':
                if event_dict['btn_prs_inaxes_flag']:
                    # rest非表示
                    [rect.set_visible(False) for rect in rect_list]

                    ax_x_px, ax_y_px = int((axes_list[0].bbox.x1 - axes_list[0].bbox.x0)), int((axes_list[0].bbox.y1 - axes_list[0].bbox.y0))
                    move_x, move_y = event_dict['btn_prs_x'] - event.x, event_dict['btn_prs_y'] - event.y
                    lim_x, lim_y = axes_list[0].get_xlim(), axes_list[0].get_ylim()
                    ax_img_pix_x, ax_img_pix_y = lim_x[1] - lim_x[0], lim_y[1] - lim_y[0]
                    move_x_pix, move_y_pix = move_x / ax_x_px * ax_img_pix_x, move_y / ax_y_px * ax_img_pix_y

                    if event.key == "shift":
                        x_lim = np.sort([event_dict['btn_prs_xdata'], event_dict['btn_prs_xdata'] - move_x_pix])
                        y_lim = np.sort([event_dict['btn_prs_ydata'], event_dict['btn_prs_ydata'] - move_y_pix])
                        [axe.set_xlim(x_lim[0], x_lim[1]) for axe in axes_list]
                        [axe.set_ylim(y_lim[int(bool(0 - yud_mode))], y_lim[int(bool(1 - yud_mode))]) for axe in axes_list]
                    else:
                        [axe.set_xlim(lim_x[0] + move_x_pix, lim_x[1] + move_x_pix) for axe in axes_list]
                        [axe.set_ylim(lim_y[0] + move_y_pix, lim_y[1] + move_y_pix) for axe in axes_list]


                elif event_dict['btn_prs_colorbar_flag']:
                    ax_y_px = int((event_dict['btn_prs_colorbar_flag'].bbox.y1 - event_dict['btn_prs_colorbar_flag'].bbox.y0))
                    move_y = event_dict['btn_prs_y'] - event.y
                    lim_y = event_dict['btn_prs_colorbar_flag'].get_ylim()
                    ax_img_pix_y = lim_y[1] - lim_y[0]
                    move_y_pix = move_y / ax_y_px * ax_img_pix_y

                    if event_dict['btn_prs_ydata']<(lim_y[1]-lim_y[0])*0.2+lim_y[0]:
                        axes_list[cbar_dict[event_dict['btn_prs_colorbar_flag']]].images[0].set_clim(lim_y[0] - move_y_pix, lim_y[1] )
                    elif (lim_y[1]-lim_y[0])*0.8+lim_y[0]<event_dict['btn_prs_ydata']:
                        axes_list[cbar_dict[event_dict['btn_prs_colorbar_flag']]].images[0].set_clim(lim_y[0], lim_y[1] - move_y_pix)
                    else:
                        axes_list[cbar_dict[event_dict['btn_prs_colorbar_flag']]].images[0].set_clim(lim_y[0] + move_y_pix, lim_y[1] + move_y_pix)

                fig.canvas.draw()

            elif hotkey_use['mouse_mode']=='ROI':
                if event_dict['ca_change']:
                    fig.canvas.draw()
                pass
            pass

        fig.canvas.mpl_connect('button_press_event',button_press)
        fig.canvas.mpl_connect('key_press_event',key_press)
        fig.canvas.mpl_connect('key_release_event',key_release)
        fig.canvas.mpl_connect('button_release_event',button_release)

        pass

    mouse_key_event(fig)
    fig.canvas.manager.set_window_title('Figure'+str(fig.number)+' [q_addon : Mouse mode = "'+hotkey_use['mouse_mode']+'"]')
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
    elif input_color=='kutinawa_color':
        i = 0
        for y in np.arange(len(main_list_for_shape)):
            temp=[]
            for x in range(len(main_list_for_shape[y])):
                temp.append(kutinawa_color[i%len(kutinawa_color)])
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
def imageq(target_img_list, caxis_list=(0, 0), cmap_list='viridis', disp_cbar=True, fig=None):
    plt.interactive(False)
    if not fig:
        fig = plt.figure()

    ############################## 必ず2次元listの形状にする
    target_img_list, target_img_list_for_shape = q_util_shaping_2dlist(target_img_list)
    caxis_list = q_util_shaping_2dlist_sub(caxis_list, target_img_list_for_shape)
    cmap_list = q_util_shaping_2dlist_sub(cmap_list, target_img_list_for_shape)

    ############################### 各imshow描画
    y_id_max = len(target_img_list)
    x_id_max = np.max(np.array([len(i) for i in target_img_list]))
    for y_id, temp_list in enumerate(target_img_list):
        for x_id, target_img in enumerate(temp_list):
            if caxis_list[y_id][x_id][1] - caxis_list[y_id][x_id][0] <= 0:
                caxis_list[y_id][x_id] = (np.nanmin(target_img[(target_img!=-np.inf)*(target_img!=np.inf)]), np.nanmax(target_img[(target_img!=-np.inf)*(target_img!=np.inf)]))

            ax = fig.add_subplot(y_id_max, x_id_max, x_id_max * y_id + x_id + 1, picker=True)
            if np.ndim(target_img) == 2:
                ims = ax.imshow(target_img.astype(float), interpolation='nearest', cmap=cmap_list[y_id][x_id])
                ims.set_clim(caxis_list[y_id][x_id][0], caxis_list[y_id][x_id][1])

            elif np.ndim(target_img) in [3, 4]:
                # 3,4ch画像をcmin~cmaxのレンジで正規化するため、値域調整
                print("imq-Warning: The image was normalized to 0-1 and clipped in the cmin-cmax range for a 3-channel image.")
                ims = ax.imshow(np.clip((target_img.astype(float) - caxis_list[y_id][x_id][0]) / (caxis_list[y_id][x_id][1] - caxis_list[y_id][x_id][0]), 0, 1),
                                interpolation='nearest', cmap=cmap_list[y_id][x_id])

            elif np.ndim(target_img) == 1 or np.ndim(target_img) > 4:
                # 4ch以上の画像は表示できないのでエラー返して終了
                print("imq-Warning: Cannot draw data other than 2-, 3-, and 4-dimensional.")
                return -1

            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            ax.tick_params(bottom=False, left=False, right=False, top=False)

            if disp_cbar:
                divider = make_axes_locatable(ax)
                ax_cbar = divider.new_horizontal(size="5%", pad=0.075)
                fig.add_axes(ax_cbar)
                fig.colorbar(ims, cax=ax_cbar)
                pass


    ############################### q機能追加
    fig = q_addon(fig, keyboard_dict={'$': q_hotkey__roistats,'%': q_hotkey__roistats2,
                                      'z': q_hotkey__zoomplot,
                                      'A':q_hotkey__climAUTO,'W':q_hotkey__climWHOLE,'E':q_hotkey__climEACH,'S':q_hotkey__climSYNC,
                                      'up':q_hotkey__climSYNCup,'down':q_hotkey__climSYNCdown,'left':q_hotkey__climSYNCleft,'right':q_hotkey__climSYNCright,
                                      '-':q_hotkey__lineprofH,'i':q_hotkey__lineprofV,'=':q_hotkey__lineprofHmean,'I':q_hotkey__lineprofVmean,
                                      'm':q_hotkey__roipixval,'h':q_hotkey_util__hist,
                                      'N':q_hotkey__noise_analyze,'c':q_hotkey__colorchecker
                                      })

    ###############################
    # status barの表示変更
    axes_list = [axe for axe in fig.get_axes() if isinstance(axe, matplotlib.axes._subplots.Subplot)]
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
                        return_str = return_str + str(k) + ': <' + '{:.3f}'.format(now_img_val[0]) + ', ' + '{:.3f}'.format(now_img_val[1]) + ', ' + '{:.3f}'.format(now_img_val[2]) + '>  '
            else:
                return_str = return_str + str(k) + ': ###' + '  '
        # 対処には、https://stackoverflow.com/questions/47082466/matplotlib-imshow-formatting-from-cursor-position
        # のような実装が必要になり、別の関数＋matplotlibの関数を叩くが必要ありめんどくさい
        return return_str

    for axe in axes_list:
        axe.format_coord = format_coord

    ############################### 表示
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


def histq(input_list,
          interval_list=1.0,
          label_list=None,
          alpha_list = 0.75,
          edgecolor_list = 'kutinawa_color',
          color_list = 'black',
          histtype='bar',
          overlay=False
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
    if overlay:
        ax_id = 1
        ax = fig.add_subplot(1,1,1,picker=True)
        y_id_max = 1
        x_id_max = 1

    x_min = []
    x_max = []
    y_max = []
    for y_id,temp_list in enumerate(input_list):
        for x_id,target_data in enumerate(temp_list):
            if not overlay:
                ax_id = x_id_max*y_id+x_id+1
                ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True)

            hist_bins = np.arange(np.nanmin(target_data), np.nanmax(target_data) + interval_list[y_id][x_id]*2, interval_list[y_id][x_id])
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
    x_spc = (x_max-x_min)*0.0495

    ############################### キーボードショートカット追加
    fig = q_addon(fig, keyboard_dict={'z': q_hotkey__zoomplot,
                                      })

    ############################### 表示
    axes_list = fig.get_axes()
    for axe in axes_list:
        axe.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
        axe.set_xlim(x_min-x_spc, x_max+x_spc)
        axe.set_ylim(0, y_max*1.05)

    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
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

def plotq(input_list_x,
          input_list_y=None,
          marker_list = 'None',
          markersize_list = 7,
          linestyle_list='-',
          linewidth_list=2,
          color_list = 'kutinawa_color',
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
                ax = fig.add_subplot(y_id_max,x_id_max,ax_id,picker=True)

            ax.plot(np.squeeze(np.reshape(input_list_x[y_id][x_id], (1, -1))),
                    np.squeeze(np.reshape(input_list_y[y_id][x_id],(1,-1))),
                    label     = label_list[y_id][x_id],
                    color     = color_list[y_id][x_id],
                    alpha     = alpha_list[y_id][x_id],
                    marker    = marker_list[y_id][x_id],
                    markersize= markersize_list[y_id][x_id],
                    linestyle = linestyle_list[y_id][x_id],
                    linewidth = linewidth_list[y_id][x_id],
                    )

            xxx[0] = np.minimum(xxx[0], np.nanmin(input_list_x[y_id][x_id].astype(float)))
            xxx[1] = np.maximum(xxx[1], np.nanmax(input_list_x[y_id][x_id].astype(float)))
            yyy[0] = np.minimum(yyy[0], np.nanmin(input_list_y[y_id][x_id].astype(float)))
            yyy[1] = np.maximum(yyy[1], np.nanmax(input_list_y[y_id][x_id].astype(float)))
            x_spc = (xxx[1]-xxx[0])*0.0495
            y_spc = (yyy[1]-yyy[0])*0.0495
    ############################### キーボードショートカット追加
    fig = q_addon(fig,
                  # keyboard_dict={'z': q_hotkey__zoomplot},
                  roi_coordinate='float')

    ############################### 表示
    for axe in fig.get_axes():
        axe.legend(loc='upper right', bbox_to_anchor=(1, 1),prop={ "weight":"bold","size": "large"})
        axe.set_xlim(xxx[0] - x_spc, xxx[1] + x_spc)
        axe.set_ylim(yyy[0] - y_spc, yyy[1] + y_spc)

    fig.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.1, hspace=0.1)# 表示範囲調整
    fig.show()

    return fig



def scatq(input_list_x,
          input_list_y=None,
          marker_list = 'o',
          markersize_list = 9,
          color_list = 'kutinawa_color',
          alpha_list = 1.0,
          label_list = None,
          overlay=True,
          fig = None,):

    plotq(input_list_x,
          input_list_y=input_list_y,
          marker_list = marker_list,
          markersize_list = markersize_list,
          linestyle_list='-',
          linewidth_list=0,
          color_list = color_list,
          alpha_list = alpha_list,
          label_list = label_list,
          overlay=overlay,
          fig = fig,)
    pass

# fig = imageq([np.random.rand(512,512),np.random.rand(512,512),wa.imread(r"C:\Users\daiki.nakagawa\Downloads\lena_gray.bmp")])
