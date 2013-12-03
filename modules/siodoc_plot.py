import os
import urllib2
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import datetime as dt
import numpy as np
from intdir2uv import intdir2uv
# from matplotlib.font_manager import FontProperties

################################################################################################

def smooth(x, window_len=200, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:  
        w = eval('np.'+window+'(window_len)')
        x = np.convolve(w/w.sum(), s, mode='same')
    return x[window_len:-window_len+1]


def remove_zeros(array):
    f = np.where(array==0)
    for index in f[0]:
        array[index] = array[index-1]
    return array


###############################################################################################  


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
    os.system( "mv %s ../static/tmp_files/%s" %(new_filename, new_filename) )


def plot(filename='../static/tmp_files/Data_month.csv_corrected'):
    columns = defaultdict(list) 
    with open(filename,'rU') as f:
        reader = csv.DictReader(f) 
        for row in reader: 
            for (k,v) in row.items(): 
                columns[k].append(v) 

    varlist = ["gust", "wspd", "wdir", "cspd", "cdir", "sst"]
    tlim = -300
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

    wu, wv   = intdir2uv(wspd, wdir, decmag, 0)
    wu2, wv2 = intdir2uv(wspd*0 + 5, wdir, decmag, 0)
    wu2, wv2 = wu2*-1, wv2*-1

    cu, cv   = intdir2uv(cspd, cdir, decmag, 0)
    cu2, cv2 = intdir2uv(cspd*0 + 0.5, cdir, decmag, 0)

    gx = np.linspace(min(gust), max(gust), len(gust))
    gy = np.linspace(min(gust),max(gust),round(max(gust)+2))
    xlimit = np.linspace(0, len(gust), len(gust))

    gx, gy = np.meshgrid(gx, gy)

    ylimit = gx[0,-1] + 2
    up = np.linspace(ylimit, ylimit, len(gust))

    fig = plt.figure(facecolor='w', figsize=(17,10))
    ax = fig.add_subplot(311)

    plt.suptitle(u'Meteoceanographic Buoy ( SIODOC/IEAPM ) \n$(lon: -42.18 - lat: -22.99)$', 
                                          fontweight='bold', fontsize='large')

    # plt.title('\n$(lon: -42.18 - lat: -22.99)$', fontweight='semibold',
    #                                       fontsize='small' )

    fy = np.linspace(min(wspd), max(gust), len(gust))
    ax.fill_betweenx(fy, xlimit[132], xlimit[188], color='k', alpha=0.1)

    ax.contourf(gy, 100, alpha=1, cmap=plt.cm.hot_r)
    ax.plot(xlimit, gust, 'w-')
    plt.title(u'Wind & Gust (m/s)', fontstyle='oblique', 
                            fontsize='smaller', fontweight='semibold')
    ax.set_axis_off()
    ax.fill_between(xlimit, up, gust, color='w')

    # ax2 = ax.twinx()
    ax.plot(xlimit, wspd, 'k-', linewidth=2)
    ax.set_axis_off()
    ax.fill_between(xlimit, up, wspd, color='w', alpha=0.6)


    ventoraj_label = map(int,gust)
    vento_label = map(int,wspd)

    for r, txt_raj in enumerate(ventoraj_label[::10]):
        ax.annotate(txt_raj, (xlimit[::10][r]-0.4,gust[::10][r]+0.8), 
                            fontsize=10, fontstyle='italic', alpha=0.5, ha='center')

    for i, txt in enumerate(vento_label[::10]):
        ax.annotate(txt, (xlimit[::10][i]-0.6,wspd[::10][i]-1.5), 
                            fontsize=10, fontstyle='italic', ha='center')


    ax.quiver(xlimit[::5], wspd[::5]-2, wu2[::5], wv2[::5],width=0.0015, scale=400)

    plt.axis([xlimit.min(), xlimit.max(), 0, gust.max()])



    ####### PLOT CORRENTE


    fa = np.where(cu > 0)[0][0]
    fb = np.where(cu[fa:] < 0)[0][0]
    fc = np.where(cu[fa:] < 0)[0][-1]

    uplim = np.concatenate( ( cu[:fa], np.linspace(0, 0, len(cu[fa:][:fb])),
       cu[fa:][fb:fc], np.linspace(0, 0, len(cu[fa:][fc:])) ) ) 

    downlim = np.concatenate( ( np.linspace(0, 0, len(cu[:fa])), cu[fa:][:fb],
      np.linspace(0, 0, len(cu[fa:][fb:fc])), cu[fa:][fc:] ) )

    cx = np.linspace(min(cu), max(cu), len(cu))     
    cy = np.linspace(round(min(cu))-0.1, round(max(cu))+0.1, 2)
    x = np.linspace(0, len(cu), len(cu))
    zr =  np.linspace(0, 0, len(cu))
    down = np.linspace(-1.1, -1.1, len(cu))
    up  = np.linspace(1.1, 1.1, len(cu))

    curr_cf = np.meshgrid(cx, cy)

    #fig = plt.figure(facecolor='w', figsize=(15,5))
    ax = fig.add_subplot(312)
    plt.contourf(x, cy, curr_cf[1],100,alpha=0.5, cmap=plt.cm.PRGn)
    plt.plot_date(x, cu, 'k-', linewidth=2)
    #plt.plot(x, zr, 'k--', linewidth=0.5)
    plt.title(u'Along-shelf surface flow (cm/s)',
                            fontstyle='oblique', fontsize='smaller', fontweight='semibold')

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

    plt.quiver(x[::5], cu[::5]-0.2, cu2[::5], cv2[::5], width=0.0015, scale=40)    




    ######## PLOT TEMP

    tempx = np.linspace(min(sst), max(sst), len(sst))   
    tempy = np.linspace(round(min(sst))-0.5, round(max(sst)) + 0.5, 24) #round(min(sst))
    x = np.linspace(0, len(sst), len(sst))
    temp_cf = np.meshgrid(tempx, tempy)

    up_lim = 24
    up = np.linspace(up_lim, up_lim, len(sst))

    ax = fig.add_subplot(313)
    plt.contourf(x, tempy, temp_cf[1], 60, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.plot(x, sst,'k-', linewidth=2)
    plt.title(u'Sea surface temperature ($^\circ$C)',fontstyle='oblique',
                                   fontsize='smaller', fontweight='semibold')
    plt.fill_between(x, up, sst, color= 'w')

    temp_label = map(int, sst)
    for i, txt_temp in enumerate(temp_label[::10]):
        ax.annotate(txt_temp, (x[::10][i]-0.4, sst[::10][i]+0.5), 
                  fontsize=10, fontstyle='italic', ha='center')
      
    # plt.fill_betweenx(tempx, x[132],x[188], color='k', alpha=0.1)


    nbs = 60
    plt.locator_params(axis='x', nbins=nbs)

    fig.canvas.draw()
    index = [item.get_text() for item in ax.get_xticklabels()]
    index = map(int, index)
    index[-1] = index[-1]-1



    lab = [datetime[i] for i in index]
    label = [l.strftime('%d') for l in lab]

    labels = map(int, np.ones((61)))

    for c in np.arange(0,61,3):
      labels[c] = label[c]
      try: 
        labels[c+1] = ''
        labels[c+2] = ''
      except:
         'IndexError'

    ax.set_yticklabels("")
    ax.set_xticklabels(labels, fontsize='smaller', fontweight='bold')
    ax.set_xlabel('\nNov/Dez 2013', fontsize='smaller', fontweight='semibold')
    ax.set_frame_on(False)


    plt.savefig('../static/images/siodoc_tmp.png', dpi=96)
    plt.close('all')
    #plt.show()

