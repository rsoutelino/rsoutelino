# -*- coding: utf-8 -*-

#--===============================================================
#
#    Script for plotting buoy measurements in the windguru fashion
#
#    Authors: Rafael Soutelino and Phellipe Couto
#    Colaborator: Victor Godoi
#
#--===============================================================

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
from matplotlib.dates import date2num
import ephem

localpath = "/home/rsoutelino/web2py/applications/rsoutelino/"

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
    os.system( "cp %s %sstatic/tmp_files/%s" %(new_filename, localpath, new_filename) )
    os.system( "cp %s %sstatic/tmp_files/%s_%s" %(new_filename, localpath, 
                                       filename[:10], str(dt.datetime.today())[:10]) )
    os.system( "rm Data_month*")


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


def replace_zeros(array):
    f = np.where(array==0)
    for index in f[0]:
        try:
            array[index] = (array[index-1] + array[index+1]) / 2 
        except:
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


def get_rise_set(day, dst):
    obs = ephem.Observer()
    obs.date = ephem.Date(day)
    obs.lon = '-42'
    obs.lat = '-23'
    obs.elevation = 0
    sun = ephem.Sun(obs)
    rise = sun.rise_time.datetime() - dt.timedelta(hours=3)
    sett = sun.set_time.datetime() - dt.timedelta(hours=3)
    if dst:
        rise = rise + dt.timedelta(hours=1)
        sett = sett + dt.timedelta(hours=1)
    return rise, sett


def night_fill(ax, rise_and_set, datesnum, ymin, ymax):
    for d in range(len(rise_and_set)):
        if d == 0: # begin boundary
            left = datesnum[0]
            right = rise_and_set[d][0]
            # print "BEGIN"
        elif d == (len(rise_and_set)-1): # end boundary
            left = rise_and_set[d][1]
            right = datesnum[-1]
            # print "END"
        else: # middle
            left = rise_and_set[d-1][1]
            right = rise_and_set[d][0]
            # print "MIDDLE"

        # ax.plot([left, right], [12, 12], 'k*')
        ax.fill([left, right, right, left], [ymin, ymin, ymax, ymax], 
             color='0.6', alpha=0.2)


###############################################################################################  


def plot(dst=False, filename=localpath + 'static/tmp_files/Data_month.csv_corrected'):

    # dst=True
    # filename = localpath + 'static/tmp_files/Data_month.csv_corrected'

    window_len = 12
    decmag = -22.3

    # READING DATA
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

    # loading full temperature measurements
    depths = [0, 10, 15, 20, 25, 45]

    f = open(filename)
    lines = f.readlines()
    SST = np.zeros((len(depths), sst.size-1))

    c = 0
    for t in range(tlim, -1):
        SST[0,c] = float(lines[t].split(",")[-11])
        SST[1,c] = float(lines[t].split(",")[-10])
        SST[2,c] = float(lines[t].split(",")[-9])
        SST[3,c] = float(lines[t].split(",")[-8])
        SST[4,c] = float(lines[t].split(",")[-7])
        SST[5,c] = float(lines[t].split(",")[-5])
        c += 1 

    SST = remove_zeros(SST)

    for k in range(len(depths)):
        SST[k,:] = smooth(SST[k,:], 5)

    # loading full current measurements
    depths_c = [2, 10, 16, 20, 26, 30, 36]

    f = open(filename)
    lines = f.readlines()
    CSPD = np.zeros((len(depths_c), cspd.size-1))
    CDIR = np.zeros((len(depths_c), cspd.size-1))

    c = 0
    for t in range(tlim, -1):
        CSPD[0,c] = float(lines[t].split(",")[21])
        CSPD[1,c] = float(lines[t].split(",")[13])
        CSPD[2,c] = float(lines[t].split(",")[14])
        CSPD[3,c] = float(lines[t].split(",")[15])
        CSPD[4,c] = float(lines[t].split(",")[16])
        CSPD[5,c] = float(lines[t].split(",")[17])
        CSPD[6,c] = float(lines[t].split(",")[18])

        CDIR[0,c] = float(lines[t].split(",")[4])
        CDIR[1,c] = float(lines[t].split(",")[5])
        CDIR[2,c] = float(lines[t].split(",")[6])
        CDIR[3,c] = float(lines[t].split(",")[7])
        CDIR[4,c] = float(lines[t].split(",")[8])
        CDIR[5,c] = float(lines[t].split(",")[9])
        CDIR[6,c] = float(lines[t].split(",")[10])
        c += 1 

    # CSPD = remove_zeros(CSPD)
    # for k in range(len(depths_c)):
    #     CSPD[k,:] = smooth(CSPD[k,:], 5)

    # CDIR = remove_zeros(CDIR)
    # for k in range(len(depths_c)):
    #     CDIR[k,:] = smooth(CDIR[k,:], 5)

    CU, CV = intdir2uv(CSPD, CDIR, decmag, 0)


    # loading wave parameters
    wave_param = [1, 2, 3, 4]
    f = open(filename)
    lines = f.readlines()
    WAVE  = np.zeros((len(wave_param), sst.size))

    c = 0
    for t in range(tlim, -1):
        WAVE[0,c] = float(lines[t].split(",")[22])  # hm0
        WAVE[1,c] = float(lines[t].split(",")[25])  # hmax
        WAVE[2,c] = float(lines[t].split(",")[52])  # tp
        WAVE[3,c] = float(lines[t].split(",")[34])  # mdir
        c += 1

    # Replacing zeros with linear interpolation and smoothing temporal series
    for k in range(len(WAVE)):
        WAVE[k,:] = replace_zeros(WAVE[k,:])    
    for k in range(len(WAVE)):
        WAVE[k,:] = smooth(WAVE[k,:], 10)          

    # Preparing direction arrays
    WU, WV = np.ones((WAVE[3].shape)), np.ones((WAVE[3].shape))
    intensity = 5
    for k in range(WAVE[3].size):
        WU2, WV2 = intdir2uv(intensity, WAVE[3,k], decmag, 0)
        WU[k], WV[k] = WU[k]*WU2, WV[k]*WV2
    WU, WV = WU * -1, WV * -1     

    # =============================================================================

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

    years = list(np.array(days)*0 + year); del year
    dates, datesnum = [], []

    for year, month, day, hour, minute in zip(years, months, days, hours, minutes):
        d = dt.datetime(year, month, day, hour)
        dates.append( d )
        datesnum.append( date2num(d) )

    datetime = np.array(datetime)
    datesnum = np.array(datesnum)

    # PRE-PROCESSING DATA
    for var in varlist:
        if var != "sst":
            exec('%s = remove_zeros(%s)' %(var, var))
            exec('%s = smooth(%s, window_len)' %(var, var) )
        else:
            exec('%s = remove_zeros(%s)' %(var, var))
            exec('%s = smooth(%s, 5)' %(var, var) )


    # PLOTTING

    fig = plt.figure(facecolor='w', figsize=(17,10))
    plt.suptitle(u'Meteoceanographic Buoy ( SIODOC/IEAPM )\
                     \n$(lon: -42.18 \ \  lat: -22.99)$', fontsize='large')

    #### PLOT WIND & GUST  
    ax = fig.add_subplot(311)
    # print ax.get_position().bounds
    plt.title(u'Wind & Gust (m/s)', fontsize='smaller')

    # preparing arrays for colorfill
    ymin = 0
    ymax = 15
    gy = np.linspace(ymin, ymax, 100)
    gx, gy = np.meshgrid(datesnum, gy)
    xlimit = np.linspace(0, gust.size, gust.size )
    up = np.linspace(ymax+10, ymax+10, gust.size )

    ax.contourf(gx, gy, gy, 100, alpha=1, cmap=plt.cm.hot_r)
    ax.plot(datesnum, gust, 'w-')
    ax.fill_between(datesnum, up, gust, color='w')
    ax.plot(datesnum, wspd, 'k-', linewidth=2)
    ax.set_axis_off()
    ax.fill_between(datesnum, up, wspd, color='w', alpha=0.6)
    # preparing arrays
    wu, wv   = intdir2uv(wspd, wdir, decmag, 0)
    wu2, wv2 = intdir2uv(wspd * 0 + 5, wdir, decmag, 0)
    wu2, wv2 = wu2 * -1, wv2 * -1
    gust_label, wind_label = map(int,gust), map(int,wspd)
    d = 5
    for r, gl in enumerate(gust_label[::d]):
        ax.annotate(gl, (datesnum[::d][r], gust[::d][r] + 0.5), 
                         fontsize=10, fontstyle='italic', alpha=0.5, ha='center')
    for i, wl in enumerate(wind_label[::d]):
        ax.annotate(wl, (datesnum[::d][i], wspd[::d][i] - 1.2), 
                         fontsize=10, fontstyle='italic', ha='center')
    ax.quiver(datesnum[::5], wspd[::5] - 1.8, wu2[::5], wv2[::5],
              width=0.0015, scale=400, pivot='middle')
    plt.axis([datesnum.min(), datesnum.max(), 0, ymax])

    ####### PLOT SURFACE CURRENT
    ax = fig.add_subplot(312)
    # print ax.get_position().bounds
    plt.title(u'Along-shelf surface flow (cm/s)', fontsize='smaller')
    # preparing arrays for colorfill
    cu, cv   = intdir2uv(cspd, cdir, decmag, 0)
    ymax = 0.8  
    cy = np.linspace(ymax*-1, ymax, 100)
    x = np.linspace(0, cu.size, cu.size)
    zr =  np.linspace(0, 0, cu.size)
    cx, cy = np.meshgrid(datesnum, cy)
    down = np.linspace(ymax*-1 -1, ymax*-1 -1, cu.size)
    up  = np.linspace(ymax+1, ymax+1, cu.size)

    plt.contourf(cx, cy, cy, 100, cmap=plt.cm.PRGn)       
    plt.plot(datesnum, cu, 'k-', linewidth=2)
    uplim, downlim = fill_polygons(cu)
    plt.fill_between(datesnum, down, uplim, color='w')
    plt.fill_between(datesnum, up, downlim, color='w')
    ax.set_axis_off()

    yoffset = ymax*0.1
    curr_label = []
    uan = np.abs(cu*100)

    for l in range(len(cu)):
      curr_label.append("%2d" %uan[l])

    d = 5
    for r, cl in enumerate(curr_label[::d]):
        if cu[::d][r] >= 0:
            off = yoffset
        else:
            off = yoffset*-2
        ax.annotate(cl, (datesnum[::d][r], cu[::d][r]+off), 
                  fontsize=10, fontstyle='italic', ha='center')
    # preparing arrays
    ux = cu*0
    ux[np.where(ux > 0)] = 1
    ux[np.where(ux < 0)] = -1
    cu2, cv2 = intdir2uv(cspd*0 + 0.5, cdir, decmag, 0)

    for r in np.arange(0, cu.size, d):
        if cu[r] <= 0:
            off = yoffset*2
        else:
            off = yoffset*-2
        plt.quiver(datesnum[r], cu[r]+off, cu2[r], cv2[r], 
               width=0.0015, scale=40, pivot='middle')   
    plt.axis([datesnum.min(), datesnum.max(), ymax*-1, ymax])

    ######## PLOT TEMP
    ax = fig.add_subplot(313)
    # print ax.get_position().bounds
    plt.title(u'Sea surface temperature ($^\circ$C)', fontsize='smaller')
    # preparing arrays for colorfill
    ymin = 12
    ymax = 28   
    tempy = np.linspace(ymin, ymax, 100)
    tempx, tempy = np.meshgrid(datesnum, tempy)
    uplim = np.linspace(ymax+2, ymax+2, sst.size)

    ax.contourf(tempx, tempy, tempy, 60, cmap=plt.cm.RdBu_r)
    plt.plot(datesnum, sst, 'k-', linewidth=2)
    plt.fill_between(datesnum, uplim, sst, color= 'w')
    d = 5
    yoffset = ymax*0.03
    temp_label = map(int, sst)
    for i, tl in enumerate(temp_label[::d]):
        ax.annotate(tl, (datesnum[::d][i], sst[::d][i]+yoffset), 
                  fontsize=10, fontstyle='italic', ha='center')
    ax.set_axis_off()

    # plotting dates
    ax = fig.add_axes([0.125, 0.04, 0.775, 0.05])
    ax.plot(datesnum, sst*0, 'w')
    ax.set_axis_off()

    for d in np.arange(0, datesnum.size, 24):
        label = dates[d].strftime('%d/%b')
        ax.annotate(label, (datesnum[d], -0.8), 
                    fontweight='bold', ha='center')
    for d in np.arange(0, datesnum.size, 4):
        label = dates[d].strftime('%Hh')
        ax.annotate(label, (datesnum[d], 0.6), ha='center', fontsize=8)
    plt.axis([datesnum[0], datesnum[-1], -1, 1])

    # plotting rise and set shades
    rise_and_set = []

    for i in np.arange(0, datesnum.size, 24):
        day = dt.date(years[i], months[i], days[i])
        rise, sett = get_rise_set(day, dst)
        rise, sett = date2num(rise), date2num(sett)
        rise_and_set.append([rise, sett])

    ax = fig.add_axes([0.125, 0.07, 0.781, 0.8])
    night_fill(ax, rise_and_set, datesnum, ymin, ymax)
    ax.set_axis_off()

    plt.savefig(localpath + 'static/images/siodoc_tmp.png', dpi=96)
    

    ########################################################################


    fig = plt.figure(facecolor='w', figsize=(17,10))
    plt.suptitle(u'Meteoceanographic Buoy ( SIODOC/IEAPM )\
                     \n$(lon: -42.18 \ \  lat: -22.99)$', fontsize='large')

    # preparing arrays
    x, z = np.meshgrid(datesnum[:-1], np.array(depths))

    ax = fig.add_subplot(311)  
    plt.plot_date(datesnum[:-1], wspd[:-1]*0 + 17, 'w') 
    plt.title(u'Temperature ($^\circ$C)', fontsize='smaller')
    plt.contourf(x, -z, SST, np.arange(10, 25, 0.1))
    plt.colorbar(orientation='horizontal', aspect=40, shrink=0.5)
    plt.contour(x, -z, SST, [20, 20], colors='k')
    plt.ylabel("Depth [m]")
    plt.axis([datesnum[0], datesnum[-1], -45, 0])

    ax = fig.add_axes([0.125, 0.1, 0.775, 0.5]) 
    plt.title(u'Velocity (cm/s)', fontsize='smaller')
    plt.ylabel("Depth [m]")
    c = 0
    j = 2
    for d in depths_c:
        plt.quiver(datesnum[::j], datesnum[::j]*0-d,
                   CU[c,::j], CV[c,::j], width=0.001, scale=1200)
        c += 1
    plt.quiver(datesnum[-10], -5, -50, 0, width=0.003, scale=1200)
    plt.text(datesnum[-19], -7, "50 cm/s")
    plt.axis([datesnum[0], datesnum[-1], -45, 5])
    # ax.set_axis_off()
    ax.set_xticklabels([])

    ax = fig.add_axes([0.125, 0.07, 0.781, 0.83])
    night_fill(ax, rise_and_set, datesnum, ymin, ymax)
    ax.set_axis_off()

    # plotting dates
    ax = fig.add_axes([0.125, 0.04, 0.775, 0.05])
    ax.plot(datesnum, sst*0, 'w')
    ax.set_axis_off()

    for d in np.arange(0, datesnum.size, 24):
        label = dates[d].strftime('%d/%b')
        ax.annotate(label, (datesnum[d], -0.8), 
                    fontweight='bold', ha='center')
    for d in np.arange(0, datesnum.size, 4):
        label = dates[d].strftime('%Hh')
        ax.annotate(label, (datesnum[d], 0.6), ha='center', fontsize=8)
    plt.axis([datesnum[0], datesnum[-1], -1, 1])

    plt.savefig(localpath + 'static/images/siodoc_full.png', dpi=96)


##############################################################################

    #### PLOT WAVE PARAMETERS

    # Obs.: WAVE[0] => Significant wave height (hm0)
    #       WAVE[1] => Maximum wave height (hmax)
    #       WAVE[2] => Peak wave period (tp)
    #       WAVE[3] => Peak wave direction (mdir)


    fig = plt.figure(facecolor='w', figsize=(17,6))
    # plt.suptitle(u'Meteoceanographic Buoy ( SIODOC/IEAPM )\
    #                  \n$(lon: -42.18 \ \  lat: -22.99)$', fontsize='large')    

    ax = fig.add_subplot(111)
    plt.title(u'Significant wave height (m), Maximum wave height (m),\
 Peak wave direction (°) & Peak wave period (s)', fontsize='smaller')

    # preparing arrays for colorfill
    ymin = 0
    ymax = 7
    gy = np.linspace(ymin, ymax, 100)
    gx, gy = np.meshgrid(datesnum, gy)
    xlimit = np.linspace(0, WAVE[0].size, WAVE[0].size )
    up = np.linspace(ymax+2, ymax+2, WAVE[1].size )

    ax.contourf(gx, gy, gy, np.arange(0, 4, 0.05), cmap=plt.cm.Blues, extend='both')
    ax.plot(datesnum, WAVE[1], 'w-')
    ax.fill_between(datesnum, up, WAVE[1], color='w')
    ax.plot(datesnum, WAVE[0], 'k-', linewidth=2)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()
    ax.fill_between(datesnum, up, WAVE[0], color='w', alpha=0.7)

    ax2 = ax.twinx()        
    ax2.plot(datesnum, WAVE[2], 'r--', linewidth=2) 
    ax2.set_ylim(-10, 18)
    ax2.set_axis_off()

    hmax_label, hm0_label, tp_label = map(float,WAVE[1]), map(float,WAVE[0]), map(float,WAVE[2])
    hmax_label = map(float,['%.1f' % i for i in hmax_label]);
    hm0_label = map(float,['%.1f' % i for i in hm0_label]);
    tp_label = map(int,['%d' % i for i in tp_label]);

    d = 5
    for r, gl in enumerate(hmax_label[::d]):
        ax.annotate(gl, (datesnum[::d][r], hmax_label[::d][r] + 0.15), 
                         fontsize=8, fontstyle='italic', alpha=0.5, ha='center', 
                         fontweight='medium')    
    for i, wl in enumerate(hm0_label[::d]):
        ax.annotate(wl, (datesnum[::d][i], hm0_label[::d][i] - 0.25),
                         fontsize=9, fontstyle='italic', ha='center', fontweight='medium')
    for r, gl in enumerate(tp_label[::d]):
        ax2.annotate(gl, (datesnum[::d][r], tp_label[::d][r] + 1), 
                         fontsize=8, color='r', fontstyle='italic', ha='center', 
                         weight='medium')    

    ax.quiver(datesnum[::d], WAVE[0][::d] - 0.4, WU[::d], WV[::d],
              width=0.0015, scale=400, pivot='middle')

    # plotting dates
    ax = fig.add_axes([0.125, 0.04, 0.775, 0.05])
    ax.plot(datesnum, sst*0, 'w')
    ax.set_axis_off()

    for d in np.arange(0, datesnum.size, 24):
        label = dates[d].strftime('%d/%b')
        ax.annotate(label, (datesnum[d], -0.8), 
                    fontweight='bold', ha='center')
    for d in np.arange(0, datesnum.size, 4):
        label = dates[d].strftime('%Hh')
        ax.annotate(label, (datesnum[d], 0.6), ha='center', fontsize=8)
    plt.axis([datesnum[0], datesnum[-1], -1, 1])

    # plotting rise and set shades
    rise_and_set = []

    for i in np.arange(0, datesnum.size, 24):
        day = dt.date(years[i], months[i], days[i])
        rise, sett = get_rise_set(day, dst)
        rise, sett = date2num(rise), date2num(sett)
        rise_and_set.append([rise, sett])

    ax = fig.add_axes([0.125, 0.07, 0.781, 0.8])
    night_fill(ax, rise_and_set, datesnum, ymin, ymax)
    ax.set_axis_off()

    plt.savefig(localpath + 'static/images/siodoc_wave.png', dpi=96)

    plt.close('all')
