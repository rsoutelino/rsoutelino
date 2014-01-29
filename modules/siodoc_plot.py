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


def plot(dst=False, filename=localpath + 'static/tmp_files/Data_month.csv_corrected'):

    # dst=True
    # filename = localpath + 'static/tmp_files/Data_month.csv_corrected'

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

    years = list(np.array(days)*0 + year); del year
    dates, datesnum = [], []

    for year, month, day, hour, minute in zip(years, months, days, hours, minutes):
        d = dt.datetime(year, month, day, hour)
        dates.append( d )
        datesnum.append( date2num(d) )

    datetime = np.array(datetime)
    datesnum = np.array(datesnum)


    # getting rise and set times
    daysarr = np.array(days)
    obs = ephem.Observer()
    obs.lon = '-42'
    obs.lat = '-23'
    obs.elevation = 0

    rise_and_set = []

    for i in np.arange(0, datesnum.size, 24):
        day = dt.date(years[i], months[i], days[i])
        obs.date = ephem.Date(day)
        sun = ephem.Sun(obs)
        rise = sun.rise_time.datetime() - dt.timedelta(hours=3)
        sett = sun.set_time.datetime() - dt.timedelta(hours=3)
        if dst:
            rise = rise + dt.timedelta(hours=1)
            sett = sett + dt.timedelta(hours=1)
        rise, sett = date2num(rise), date2num(sett)
        rise_and_set.append([rise, sett])



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

    ymin = 0
    ymax = 15
    gy = np.linspace(ymin, ymax, 100)
    gx, gy = np.meshgrid(datesnum, gy)
    xlimit = np.linspace(0, gust.size, gust.size )
    up = np.linspace(ymax+10, ymax+10, gust.size )

    # night fill
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
                 color='0.4', alpha='0.2')

    ax.contourf(gx, gy, gy, 100, alpha=1, cmap=plt.cm.hot_r)
    ax.plot(datesnum, gust, 'w-')

    # ax.set_axis_off()
    ax.fill_between(datesnum, up, gust, color='w')

    ax.plot(datesnum, wspd, 'k-', linewidth=2)
    ax.set_axis_off()
    ax.fill_between(datesnum, up, wspd, color='w', alpha=0.6)

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

    offset = datesnum.mean()*0.00000004
    ax.quiver(datesnum[::5]+offset, wspd[::5] - 1.6, wu2[::5], wv2[::5],
              width=0.0015, scale=400)




    plt.axis([datesnum.min(), datesnum.max(), 0, ymax])

    # plt.show()
    # stop

    ####### PLOT CORRENTE

    ax = fig.add_subplot(312)
    plt.title(u'Along-shelf surface flow (cm/s)', fontsize='smaller')

    cu, cv   = intdir2uv(cspd, cdir, decmag, 0)

    ymax = 0.5  
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
        ax.annotate(cl, (datesnum[::d][r], cu[::d][r]+yoffset), 
                  fontsize=10, fontstyle='italic', ha='center')

    ux = cu*0
    ux[np.where(ux > 0)] = 1
    ux[np.where(ux < 0)] = -1
    cu2, cv2 = intdir2uv(cspd*0 + 0.5, cdir, decmag, 0)

    plt.quiver(datesnum[::d]+offset, cu[::d]-yoffset, cu2[::d], cv2[::d], 
               width=0.0015, scale=40)   

    plt.axis([datesnum.min(), datesnum.max(), ymax*-1, ymax])

    ######## PLOT TEMP

    ax = fig.add_subplot(313)
    plt.title(u'Sea surface temperature ($^\circ$C)', fontsize='smaller')

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


    plt.savefig(localpath + 'static/images/siodoc_tmp.png', dpi=96)
    plt.close('all')


