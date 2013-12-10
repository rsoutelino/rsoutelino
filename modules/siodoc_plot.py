import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import urllib2
import numpy as np
import csv
import datetime as dt
import matplotlib.dates as mdates
from intdir2uv import intdir2uv
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import defaultdict


localpath = "/home/rsoutelino/admin/website/web2py/applications/rsoutelino/"

################################################################################################

def download_data():
    top_level_url = "http://metocean.fugrogeos.com/marinha/"
    target_url = "http://metocean.fugrogeos.com/marinha/Members/Data_month.csv"

    # create a password manager
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, top_level_url, "rsoutelino", "@#Upwelling")
    handler = urllib2.HTTPBasicAuthHandler(password_mgr)
    opener = urllib2.build_opener(handler)

    # use the opener to fetch a URL
    f = opener.open(target_url)
    urllib2.install_opener(opener)

    with open(os.path.basename(target_url), "wb") as local_file:
        local_file.write(f.read())

    filename = "Data_month.csv"
    new_filename = filename + "_corrected"
    os.system("grep -v Brazil %s > %s" %(filename, new_filename) )
    os.system( "mv %s %sstatic/tmp_files/%s" %(new_filename, localpath, new_filename) )
    os.system( "rm %s " %filename )

def smooth(x, window_len=200, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:  
        w = eval('np.' + window + '(window_len)')
        x = np.convolve(w / w.sum(), s, mode='same')
    return x[window_len : -window_len+1]

def remove_zeros(array):
    f = np.where(array==0)
    for index in f[0]:
        array[index] = array[index-1]
    return array

def fill_polygons(array):

    neg, pos = [], []
    npol, ppol = [], []
    neg_zeros, pos_zeros = [], []
    uplim, downlim = np.array([]), np.array([])
    
    for k in range( len(array) ):
        if array[k] < 0:
            neg.append(k)
        else:
            pos.append(k)

    neg.append(999), pos.append(999)

    k = -1
    for i in range( len(array) ):
        try: 
            if neg[i] != neg[i+1] - 1:
                npol.append( array[ neg[k+1  : i+1] ] )
                k = i  
        except IndexError:
            pass

    k = -1
    for i in range( len(array) ):
        try: 
            if pos[i] != pos[i+1] - 1:
                ppol.append( array[ pos[k+1  : i+1] ] )
                k = i   
        except IndexError:
            pass

    for r in range(len(ppol)):
        try:
            neg_zeros.append(np.zeros(len(ppol[r])))
        except IndexError:
            pass
   
    for r in range(len(npol)):
        try:
            pos_zeros.append(np.zeros(len(npol[r])))
        except IndexError:
            pass
    
    if array[0] > 0:        
        try:
            for k in range(len(ppol)):
                downlim = np.concatenate((downlim,ppol[k]))
                downlim = np.concatenate((downlim,pos_zeros[k]))
        except IndexError:
            pass    

        try:
            for k in range(len(neg_zeros)):
                uplim = np.concatenate((uplim,neg_zeros[k]))
                uplim = np.concatenate((uplim,npol[k]))
        except IndexError:
            pass

    elif array[0] < 0:
        
        try:
            for k in range(len(npol)):
                uplim = np.concatenate((uplim,npol[k]))
                uplim = np.concatenate((uplim,neg_zeros[k]))
        except IndexError:
            pass    

        try:
            for k in range(len(pos_zeros)):
                downlim = np.concatenate((downlim,pos_zeros[k]))
                downlim = np.concatenate((downlim,ppol[k]))
        except IndexError:
            pass

    return uplim, downlim  


# def plot(data=np.random.randn(100)):
#     fig = Figure()
#     fig.set_facecolor('white')
#     ax = fig.add_subplot(111)
#     p = ax.plot(data)
            
#     canvas = FigureCanvas(fig)
#     canvas.print_png('../static/images/siodoc_tmp.png')


###############################################################################################  


def plot(filename=localpath + 'static/tmp_files/Data_month.csv_corrected'):
    columns = defaultdict(list) 
    with open(filename,'rU') as f:
        reader = csv.DictReader(f) 
        for row in reader: 
            for (k,v) in row.items(): 
                columns[k].append(v) 

    varlist = ["gust", "wspd", "wdir", "cspd", "cdir", "sst"]
    
    
    date_window = 7
    tlim = -24 * date_window   # put date_window as function argument

    gust = np.array(map(float, columns.values()[7][tlim:])) 
    wspd = np.array(map(float, columns.values()[3][tlim:]))
    wdir = np.array(map(float, columns.values()[19][tlim:]))
    cspd = np.array(map(float, columns.values()[32][tlim:]))/100
    cdir = np.array(map(float, columns.values()[6][tlim:]))
    sst  = np.array(map(float, columns.values()[9][tlim:]))
    
    time = columns.values()[44][tlim:]
    year = dt.datetime.now().year
    days, months, hours, minutes = [], [], [], []
    datetime = []
    
    for d in range(len(time)):
        days.append(int(time[d].split()[0].replace('.','')[0:2]))
        months.append(int(time[d].split()[0].replace('.','')[2:4]))    
        hours.append(int(time[d].split()[1].replace(':','')[0:2]))   
        minutes.append(int(time[d].split()[1].replace(':','')[2:4]))
        datetime.append(dt.datetime(year, months[d], days[d], hours[d], minutes[d]))

    datetime = np.array(datetime)

    window_len = 12
    decmag = -22.3

    for var in varlist:
        if var != "sst":
            exec('%s = remove_zeros(%s)' %(var, var))
            exec('%s = smooth(%s, window_len)' %(var, var) )
        else:
            exec('%s = remove_zeros(%s)' %(var, var))
            exec('%s = smooth(%s, 5)' %(var, var) )


#### PLOT WIND & GUST  

    fig = plt.figure(facecolor='w', figsize=(17,10))
    plt.suptitle(u'Meteoceanographic Buoy ( SIODOC/IEAPM )\
                     \n$(lon: -42.18 \ \  lat: -22.99)$', fontsize='large')

    ax = fig.add_subplot(311)
    plt.title(u'Wind & Gust (m/s)', fontsize='smaller')

    gx = np.linspace(min(gust), max(gust), len(gust))
    gy = np.linspace(min(gust), max(gust), round(max(gust) + 2 ))
    gx, gy = np.meshgrid(gx, gy)
    xlimit = np.linspace(0, len(gust), len(gust) )
    ylimit = gx[0,-1]
    up = np.linspace(ylimit, ylimit, len(gust))

    ax.contourf(gy, 100, alpha=1, cmap=plt.cm.hot_r)
    ax.plot(xlimit, gust, 'w-')

    ax.set_axis_off()
    ax.fill_between(xlimit, up, gust, color='w')

    ax.plot(xlimit, wspd, 'k-', linewidth=2)
    ax.set_axis_off()
    ax.fill_between(xlimit, up, wspd, color='w', alpha=0.6)

    wu, wv   = intdir2uv(wspd, wdir, decmag, 0)
    wu2, wv2 = intdir2uv(wspd * 0 + 5, wdir, decmag, 0)
    wu2, wv2 = wu2 * -1, wv2 * -1

    gust_label, wind_label = map(int,gust), map(int,wspd)

    for r, gust_txt in enumerate(gust_label[::5]):
        ax.annotate(gust_txt, (xlimit[::5][r] - 0.4, gust[::5][r] + 0.5), 
                            fontsize=10, fontstyle='italic', alpha=0.5, ha='center')

    for i, wind_txt in enumerate(wind_label[::5]):
        ax.annotate(wind_txt, (xlimit[::5][i] - 0.6, wspd[::5][i] - 1.2), 
                            fontsize=10, fontstyle='italic', ha='center')

    ax.quiver(xlimit[::5], wspd[::5] - 2.2, wu2[::5], wv2[::5],width=0.0015, scale=400)

    plt.axis([xlimit.min(), xlimit.max(), 0, gust.max()])



####### PLOT CORRENTE

    ax = fig.add_subplot(312)
    plt.title(u'Along-shelf surface flow (cm/s)', fontsize='smaller')
    
    cu, cv   = intdir2uv(cspd, cdir, decmag, 0)

    cx = np.linspace(min(cu), max(cu), len(cu))     
    cy = np.linspace(round(min(cu))-0.1, round(max(cu))+0.1, 2)
    x = np.linspace(0, len(cu), len(cu))
    zr =  np.linspace(0, 0, len(cu))
    curr_cf = np.meshgrid(cx, cy)
    
    down = np.linspace(min(cu)-0.5, min(cu)-0.5, len(cu))
    up  = np.linspace(max(cu)+1, max(cu)+1, len(cu))

    if np.all([cu > 0]):
        plt.contourf(x, cy, curr_cf[1],100,alpha=0.5, cmap=plt.cm.Greens)
    elif np.all([cu < 0]):
        plt.contourf(x, cy, curr_cf[1],100,alpha=0.5, cmap=plt.cm.Purples_r)
    else:
        plt.contourf(x, cy, curr_cf[1],100,alpha=0.5, cmap=plt.cm.PRGn)       
    
    plt.plot_date(x, cu, 'k-', linewidth=2)
    
    uplim, downlim = fill_polygons(cu)
    plt.fill_between(x, down, uplim,color='w')
    plt.fill_between(x, up, downlim,color='w')
    ax.set_axis_off()

    curr_label = []
    uan = np.abs(cu*100)
    
    for l in range(len(cu)):
      curr_label.append("%2d" %uan[l])

    for r, txt_curr in enumerate(curr_label[::10]):
        ax.annotate(txt_curr, (x[::10][r], cu[::10][r] + 0.15), 
                  fontsize=10, fontstyle='italic', ha='center')

    ux = cu*0
    ux[np.where(ux > 0)] = 1
    ux[np.where(ux < 0)] = -1
    cu2, cv2 = intdir2uv(cspd*0 + 0.5, cdir, decmag, 0)
    
    plt.quiver(x[::5], cu[::5]-0.2, cu2[::5], cv2[::5], width=0.0015, scale=40)    


######## PLOT TEMP

    ax = fig.add_subplot(313)
    plt.title(u'Sea surface temperature ($^\circ$C)', fontsize='smaller')

    tempx = np.linspace(min(sst), max(sst), len(sst))   
    tempy = np.linspace(round(min(sst))-0.5, round(max(sst)) + 0.5, 24) #round(min(sst))
    x = np.linspace(0, len(sst), len(sst))
    temp_cf = np.meshgrid(tempx, tempy)

    uplim = np.linspace(round(max(sst)) + 1, round(max(sst)) + 1, len(sst))

    if np.all([ sst > 20]):
        plt.contourf(x, tempy, temp_cf[1], 60, alpha=0.8, cmap=plt.cm.Reds)
    elif np.all([sst <= 20]):
        plt.contourf(x, tempy, temp_cf[1], 60, alpha=0.8, cmap=plt.cm.Blues_r)
    else:    
        plt.contourf(x, tempy, temp_cf[1], 60, alpha=0.8, cmap=plt.cm.RdBu_r)
    
    plt.plot(x, sst,'k-', linewidth=2)
    plt.fill_between(x, uplim, sst, color= 'w')

    temp_label = map(int, sst)
    for i, txt_temp in enumerate(temp_label[::10]):
        ax.annotate(txt_temp, (x[::10][i]-0.4, sst[::10][i]+0.5), 
                  fontsize=10, fontstyle='italic', ha='center')

    nbs = 60
    plt.locator_params(axis='x', nbins=nbs)

    fig.canvas.draw()
    # index = [item.get_text() for item in ax.get_xticklabels()]
    # index = map(int, index)
    # index[-1] = index[-1]-1

    # lab = [datetime[i] for i in index]
    # label = [l.strftime('%d') for l in lab]
    # labels = map(int, np.ones((61)))

    # for c in np.arange(0,61,3):
    #   labels[c] = label[c]
    #   try: 
    #     labels[c+1] = ''
    #     labels[c+2] = ''
    #   except:
    #      'IndexError'

    # ax.set_yticklabels("")
    # ax.set_xticklabels(labels, fontsize='smaller', fontweight='bold')
    # ax.set_xlabel('\nNov/Dez 2013', fontsize='smaller', fontweight='semibold')
    ax.set_frame_on(False)


    plt.savefig(localpath + 'static/images/siodoc_tmp.png', dpi=96)
    plt.close('all')
    #plt.show()

