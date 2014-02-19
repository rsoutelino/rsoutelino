from siodoc_plot import download_data, plot
from tide_pilot import *
import ephem


web2pyPath = '/home/rsoutelino/web2py/applications/rsoutelino/'




# PACMARE
date = dt.date.today()
station = 'PORTO DO FORNO'

infoList, hora, altura, time2, h2 = pacMare(date, station)

datetime = dt.datetime.now()
obs = ephem.Observer()
obs.date = ephem.Date(datetime)
obs.lon = '-42'
obs.lat = '-23'
obs.elevation = 0
sun = ephem.Sun(obs)

rise_set_times = [sun.rise_time.datetime(), sun.set_time.datetime()]

plotMare(time2, h2, station, date, rise_set_times, True, web2pyPath)



# SIODOC
download_data()
plot(dst=False)