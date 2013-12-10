#!/usr/local/epd-7.2-1-rh5-x86_64/bin/python
# -*- coding:utf-8 -*-
# Modulo para previsao de mare, parte integrante do CONTROLE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import interp1d
import datetime as dt
import matplotlib.dates as mpldates
from matplotlib.patches import Polygon

################################################################################
 

web2pyPath = '/home/rsoutelino/web2py/applications/rsoutelino/'

class ReadTideDataBase(object):
    """Parent class to read all database text files """
    def __init__(self, filename, varnames):
        self.filename = filename
        f = open(self.filename)
        lines = f.readlines()
        lines.pop(0)
        self.data = {}

        for l in range(len(lines)):
            lines[l] = re.split("\s{2,300}", lines[l].strip('\r\n').rstrip(), 
                                maxsplit=len(varnames)-1)

        for i, name in enumerate(varnames):
            self.data[name] = [values[i] for values in lines]


class Estacao(ReadTideDataBase):
    def __init__(self):
        self.filename = web2pyPath + "modules/data/Estacao.txt"
        varnames = ['ID', 'name', 'latG', 'latM', 'hemlat', 'lonG',
                    'lonM', 'hemlon', 'ncomp', 'nm', 'fuso', 'carta']
        ReadTideDataBase.__init__(self, self.filename, varnames) 
        self.data['LON'] =  (np.array(self.data['lonG'], dtype=np.float32) +\
                             np.array(self.data['lonM'], dtype=np.float32)/60.) * -1
        self.data['LAT'] =  (np.array(self.data['latG'], dtype=np.float32) +\
                             np.array(self.data['latM'], dtype=np.float32)/60.)
        for k in range(len(self.data['hemlat'])):
            if self.data['hemlat'][k] == "S":
                self.data['LAT'][k] *= -1

        self.data['ncomp'] = str2int(self.data['ncomp'])
        self.data['nm'] = str2float(self.data['nm'])
        self.data['fuso'] = str2float(self.data['fuso'])
        self.data['carta'] = str2int(self.data['carta'])


class Constantes(ReadTideDataBase):
    def __init__(self):
        self.filename = web2pyPath + "modules/data/Constantes.txt"
        varnames = ['ID', 'const', 'amp', 'phase', 'ref']
        ReadTideDataBase.__init__(self, self.filename, varnames)
        self.data['amp'] = str2float(self.data['amp'])
        self.data['phase'] = str2float(self.data['phase'])

class Cadastro(ReadTideDataBase):
    def __init__(self):
        self.filename = web2pyPath + "modules/data/Cadastro.txt"
        varnames = ['const', 'cod', 'M']
        ReadTideDataBase.__init__(self, self.filename, varnames)


class Combinacoes(ReadTideDataBase):
    def __init__(self):
        self.filename = web2pyPath + "modules/data/Combinacoes.txt"
        varnames = ['ID', 'subs', 'comb']
        ReadTideDataBase.__init__(self, self.filename, varnames)
        self.data['ID'] = str2int(self.data['ID'])

def str2int(someList):
    for k in range(len(someList)):
        someList[k] = int(someList[k])
    return someList

def str2float(someList):
    for k in range(len(someList)):
        someList[k] = float(someList[k])
    return someList

def smooth(x, window_len=3, window='hanning'):
    """
    Smooth the data using a window with requested size.   
    """
    s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')             
    return y[window_len-1:-window_len+1]

def base60Tobase10(degree, minute, hemi):
    if hemi in ['W','S','O']:
        signal = '-'
    elif hemi in ['N','L','E']:
        signal = '+'
    minute = float(minute)/60.
    degree = float(signal + str(int(degree) + minute))
    return degree

def base10Tobase60(lat=None,lon=None):
    if lat != None:
        coord = lat
    elif lon != None:
        coord = lon
    else:
        return "You must specify if it's lat or lon input"
    degree = int(coord)
    if -1 < coord < 1:
        minute = coord * 60
    else:
        minute = (coord%degree)*60 
    if -1 < minute < 1:
        second = minute * 60
    else:
        second = int((minute%int(minute))*60)
    if degree >= 0:
        if lat != None:
            signal = 'N'
        elif lon != None:
            signal = 'L'
    else:
        if lat != None:
            signal = 'S'
        elif lon != None:
            signal = 'O'
    coord = '''%02d°%02d'%02d"%s''' % \
            (abs(degree),abs(int(minute)),abs(second),signal)
    return coord.decode('utf-8')

################################################################################



def pacMare(date, estac):
    """PacMare traduzido para python"""
    monthList = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL",
                 "AGO", "SET", "OUT", "NOV", "DEZ"]
    an     = date.year
    Mesl   = date.month
    strmes = monthList[Mesl-1]
    di     = date.day
    data1  = "%s/%s/%s" %(di, Mesl, an)

    DT = 1
    HI = -3
    d0 = 1

    estacoes    = Estacao()
    constantes  = Constantes()
    cadastro    = Cadastro()
    combinacoes = Combinacoes()

    f   = estacoes.data['name'].index(estac)
    Cod = estacoes.data['ID'][f]
    LA1 = estacoes.data['latG'][f]
    LA2 = estacoes.data['latM'][f]
    LO1 = estacoes.data['lonG'][f]
    LO2 = estacoes.data['lonM'][f]
    nc  = estacoes.data['ncomp'][f]
    NM  = estacoes.data['nm'][f]
    fu  = estacoes.data['fuso'][f]
    ca  = estacoes.data['carta'][f]
    hemlat = estacoes.data['hemlat'][f]
    hemlon = estacoes.data['hemlon'][f]
    
    infoList = []
    lat = base10Tobase60(lat=base60Tobase10(LA1, LA2, hemlat))
    lon = base10Tobase60(lon=base60Tobase10(LO1, LO2, hemlon))
    latSTR = u"Lat: %s" % lat
    lonSTR = u"Lon: %s" % lon
    ncSTR  = u"Componentes: %s" %(nc)
    nmSTR  = u"Nível Médio: %s cm" %(int(NM))
    fuSTR  = u"Fuso: - %sh" %(int(fu))
    caSTR  = u"Número Carta: %s" %(ca)

    infoList.append(latSTR)
    infoList.append(lonSTR)
    infoList.append(ncSTR)
    infoList.append(nmSTR)
    infoList.append(fuSTR)
    infoList.append(caSTR)

    f  = constantes.data['ID'].index(Cod)
    ai = constantes.data['const'][ f:f+nc ]
    h  = constantes.data['amp'][ f:f+nc ]
    G  = constantes.data['phase'][ f:f+nc ]
    HH = h[:]
    GG = G[:]

    MK, constID = [],[]
    for k in range(nc):
        f = cadastro.data['const'].index(ai[k])
        MK.append(cadastro.data['M'][f])
        constID.append(cadastro.data['cod'][f])
    MK = str2int(MK)
    constID = str2int(constID)

    BB, CC = [],[]
    for k in range(nc):
        f = combinacoes.data['ID'].index(constID[k])
        aux = combinacoes.data['subs'][ f: f+MK[k] ]
        aux = str2float(aux)
        BB.append(aux)
        aux = combinacoes.data['comb'][ f: f+MK[k] ]
        aux = str2float(aux)
        CC.append(aux)

    cdat = open(web2pyPath + "modules/data/Vdata.txt")
    V = []
    for line in cdat.readlines():
        line2 = line.strip('\r\n').split(',')
        line2 = str2float(line2)
        V.append(line2)

    D = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    n = 30

    # calculo dos elementos astronomicos
    MB = float(an % 4)
    MC = float(an % 100)
    MD = float(an % 400)
    dd = float(di)

    if MB == 0 and MC != 0 or MD == 0:
        D[2] = 29

    i1 = float(an / 100)
    i2 = i1 - 19
    if i2 != 0:
        t1 = i2
        j1 = abs(i2)
        c3 = j1 / i2
        t2 = t1 * t1 * c3
        c1 = int(j1 * 0.75 + 0.5) * c3
    else:
        t1 = 0.
        t2 = 0.
        c1 = 0.

    s0 = 277.0224 + 307.8831 * t1 - 0.0011 * t2 - 13.1764 * c1
    h0 = 280.1895 + 0.7689 * t1 + 0.0003 * t2 - 0.9856 * c1
    p0 = 334.3853 + 109.034 * t1 - 0.0103 * t2 - 0.1114 * c1
    nl = 100.7902 + 134.142 * t1 - 0.0021 * t2 - 0.053 * c1
    P1 = 281.2208 + 1.7192 * t1 + 0.00045 * t2 - 0.000047 * c1

    for i in range(Mesl):
        di = float(di + D[i])

    # bug de 2001
    if an <= 2000:
        di = di - 1 

    IA = i1 * 100
    BI = an - IA

    AI = int((BI - 1) * 0.25); AI = float(AI)
    if MD == 0: AI = AI + 1
    AD = AI + di
    N2 = n * DT * 0.5
    AV = N2
    SN = AV / 10000
    b = [None]
    b.append( s0 + 129.38481 * BI + 13.1764 * AD )
    b.append( h0 - 0.23872 * BI + 0.98565 * AD   )
    b.append( p0 + 40.66249 * BI + 0.1114 * AD   )
    b.append(None)
    b.append( nl + 19.32818 * BI + 0.05295 * AD  )
    b.append( P1 + 0.01718 * BI + 0.000047 * AD  )
    b[0] = b[2] - b[1]
    b[4] = 90.
    b.append( b[3] + N2 * 0.00464183  )
    b.append( b[5] + N2 * 0.00220641  )
    b.append( b[6] + N2 * 0.00000196  )

    a = [  [0.,1.,0.], [0.,2.,0.], [0.,3.,0.], [0.,0.,2.], [0.,1.,2.], [1.,0.,-1.], 
           [2.,-1.,-1.], [2.,-1.,0.], [2.,-1.,1.], [2.,0.,0.], [2.,1.,0.], 
           [2.,2.,0.], [2.,3.,0.]   ]

    b[0] = b[0] + HI * 14.49205211
    b[1] = b[1] + HI * 0.54902653
    b[2] = b[2] + HI * 0.0410686
    b[3] = b[3] + HI * 0.00464183
    b[5] = b[5] + HI * 0.00220641
    b[6] = b[6] + HI * 0.00000196

    z, Q = [], []
    for i in range(13):
        s = 0.
        for J in range(3):
            s = s + a[i][J] * b[J + 7]
      
        XX = s * 0.017453
        z.append(np.cos(XX))
        Q.append(np.sin(XX))

    W = []
    for i in range(37):
        WQ = 0.
        for J in range(5):
            WQ = WQ + V[i][J] * b[J]
        
        if i == 13 or i == 30:
            W.append( WQ + b[9] )
        elif i == 17 or i == 32:
            W.append( WQ - b[9] )
        else:
            W.append(WQ)

    F, U = [], []
    for k in range(38):
        F.append(None) # apenas para facilitar a copia do codigo em VB
        U.append(None) # depois, ambos serao popped-up
    z.insert(0, None) # idem
    Q.insert(0, None) # idem

    F[1] = 1
    F[2] = 1
    F[3] = 1 - 0.0307 * z[1] + 0.0007 * z[2] - 0.0534 * z[10] - 0.0218 * z[11] - 0.0059 * z[12]
    F[4] = 1 + 0.4142 * z[1] + 0.0377 * z[2] - 0.0008 * z[3] - 0.0028 * z[8] + 0.0431 * z[10] - 0.0023 * z[11]
    F[5] = 1 + 0.4141 * z[1] + 0.0384 * z[2] - 0.003 * z[7] - 0.003 * z[9] + 0.0179 * z[10] - 0.004 * z[12] - 0.0017 * z[13]
    F[6] = 1 + 0.1885 * z[1] - 0.0063 * z[2] - 0.0063 * z[12]
    F[7] = 1 + 0.1884 * z[1] - 0.0061 * z[2] - 0.0087 * z[10]
    F[8] = 1 + 0.1884 * z[1] - 0.0057 * z[2] + 0.0007 * z[6] - 0.0028 * z[10] - 0.0039 * z[12] - 0.0007 * z[13]
    F[9] = 1 + 0.1881 * z[1] - 0.0058 * z[2] - 0.0576 * z[10] + 0.0175 * z[11]
    F[10] = 1 + 0.1885 * z[1] - 0.0058 * z[2] + 0.0001 * z[8] - 0.0054 * z[10] - 0.001 * z[11]
    F[11] = 1 - 0.2454 * z[1] - 0.0142 * z[2] + 0.0445 * z[10]
    F[12] = 1 + 0.1714 * z[1] - 0.0054 * z[2] + 0.3596 * z[10] + 0.0664 * z[11] - 0.0057 * z[12]
    F[13] = 1 + 0.1905 * z[1]
    F[14] = 1 - 0.0078 * z[1]
    F[15] = 1 - 0.0112 * z[1] + 0.0007 * z[2] - 0.0004 * z[4] - 0.0015 * z[10] - 0.0003 * z[11]
    F[16] = 1
    F[17] = 1 + 0.1158 * z[1] - 0.0029 * z[2] + 0.0001 * z[11]
    F[18] = 1 + 0.019 * z[1]
    F[19] = 1 - 0.0384 * z[1] - 0.0185 * z[2] + 0.0132 * z[4] + 0.0105 * z[8] + 0.0344 * z[10]
    F[20] = 1 + 0.1676 * z[1] + 0.03 * z[11]
    F[21] = 1 + 0.1685 * z[1] - 0.0047 * z[2] - 0.0152 * z[10] - 0.0098 * z[11] - 0.0057 * z[12]
    F[22] = 1 + 0.6398 * z[1] + 0.1342 * z[2] + 0.008500001 * z[3] + 0.0296 * z[8] + 0.1496 * z[10] - 0.0037 * z[11]
    F[23] = 1 - 0.0337 * z[1]
    F[24] = 1 - 0.0374 * z[1] - 0.061 * z[12]
    F[25] = 1 - 0.0375 * z[1]
    F[26] = 1 - 0.0373 * z[1] + 0.0004 * z[2] + 0.0007 * z[6] - 0.0039 * z[12]
    F[27] = 1 - 0.0373 * z[1] + 0.0042 * z[10] - 0.0036 * z[11]
    F[28] = 1 - 0.0373 * z[1] + 0.0004 * z[2] + 0.0005 * z[10] - 0.0001 * z[11]
    F[29] = 1 - 0.0448 * z[1]
    F[30] = 1 - 0.0367 * z[1] + 0.0047 * z[8] - 0.2505 * z[10] - 0.1102 * z[11] - 0.0156 * z[12]
    F[31] = 1
    F[32] = 1 - 0.0022 * z[1]
    F[33] = 1 - 0.2535 * z[4] + 0.0141 * z[5]
    F[34] = 1 + 0.2852 * z[1] + 0.0324 * z[2]
    F[35] = 1 + 0.4389 * z[1] + 0.0487 * z[2] + 0.0487 * z[10] + 0.065 * z[11]
    F[36] = 1 + 0.4168 * z[1] + 0.0466 * z[2] - 0.078 * z[10]
    F[37] = 1 - 0.0564 * z[1]

    U[1] = 0
    U[2] = 0
    U[3] = 0.0007 * Q[1] - 0.0008 * Q[2] - 0.0534 * Q[10] - 0.0218 * Q[11] - 0.0059 * Q[12]
    U[4] = 0.4142 * Q[1] + 0.0377 * Q[2] - 0.0008 * Q[3] + 0.0027 * Q[8] - 0.0432 * Q[10] + 0.0022 * Q[11]
    U[5] = 0.4142 * Q[1] + 0.0384 * Q[2] + 0.003 * Q[7] + 0.003 * Q[9] - 0.018 * Q[10] - 0.004 * Q[12] - 0.0017 * Q[13]
    U[6] = -0.1885 * Q[1] + 0.0062 * Q[2] + 0.0062 * Q[12]
    U[7] = -0.1884 * Q[1] + 0.006 * Q[2] - 0.0087 * Q[10]
    U[8] = -0.1884 * Q[1] + 0.0057 * Q[2] - 0.0008 * Q[6] - 0.0028 * Q[10] + 0.0039 * Q[12] + 0.0007 * Q[13]
    U[9] = -0.1882 * Q[1] + 0.0057 * Q[2] - 0.0576 * Q[10] + 0.0175 * Q[11]
    U[10] = -0.1885 * Q[1] + 0.0057 * Q[2] + 0.0001 * Q[8] - 0.0064 * Q[10] - 0.001 * Q[11]
    U[11] = -0.1886 * Q[1] - 0.0142 * Q[2] - 0.0446 * Q[10]
    U[12] = -0.2294 * Q[1] - 0.3596 * Q[10] - 0.0665 * Q[11] + 0.0057 * Q[12]
    U[13] = 0.246 * Q[1]
    U[14] = 0.0077 * Q[1]
    U[15] = 0.0111 * Q[1] - 0.0008 * Q[2] - 0.0004 * Q[4] - 0.0015 * Q[10] - 0.0003 * Q[11]
    U[16] = 0
    U[17] = 0.1554 * Q[1] - 0.003 * Q[2] - 0.0002 * Q[11]
    U[18] = 0.019 * Q[1]
    U[19] = -0.0384 * Q[1] - 0.0185 * Q[2] - 0.0132 * Q[4] - 0.0106 * Q[8] - 0.0344 * Q[10]
    U[20] = 0.231 * Q[1] - 0.03 * Q[11]
    U[21] = 0.2274 * Q[1] - 0.0047 * Q[2] - 0.0152 * Q[10] - 0.0098 * Q[11] - 0.0057 * Q[12]
    U[22] = 0.6398 * Q[1] + 0.1342 * Q[2] - 0.0296 * Q[8] - 0.1497 * Q[10] + 0.0037 * Q[11]
    U[23] = 0.0373 * Q[1]
    U[24] = 0.0373 * Q[1] + 0.006 * Q[12]
    U[25] = 0.0373 * Q[1] - 0.0005 * Q[2] - 0.0008 * Q[6] + 0.0039 * Q[12]
    U[26] = 0.0373 * Q[1] - 0.0005 * Q[2] - 0.0008 * Q[6] + 0.0039 * Q[12]
    U[27] = 0.0373 * Q[1] + 0.0042 * Q[10] + 0.0036 * Q[11]
    U[28] = 0.0373 * Q[1] - 0.0005 * Q[2] + 0.0005 * Q[9] + 0.0001 * Q[11]
    U[29] = 0.0487 * Q[1]
    U[30] = 0.0366 * Q[1] + 0.0047 * Q[8] - 0.2505 * Q[9] - 0.1102 * Q[11]
    U[31] = 0
    U[32] = -0.0022 * Q[1]
    U[33] = -0.2535 * Q[4] + 0.0141 * Q[5]
    U[34] = 0.3108 * Q[1] + 0.0324 * Q[2]
    U[35] = 0.4389 * Q[1] + 0.0487 * Q[2] - 0.0488 * Q[9] - 0.065 * Q[11]
    U[36] = 0.4542 * Q[1] + 0.0466 * Q[2] - 0.0078 * Q[10]
    U[37] = 0.0563 * Q[1]

    z.pop(0)
    Q.pop(0)
    F.pop(0)
    U.pop(0)
    AV = n * DT * 0.5

    for i in range(37):
        XX = F[i]
        YY = U[i]
        F[i] = np.sqrt( XX ** 2 + YY ** 2 )
        U[i] = W[i] + np.arctan(YY / XX) * 57.29578
        U[i] = U[i] - int(U[i] / 360) * 360
        if U[i] < 0: U[i] = U[i] + 360


    # calculo das alturas
    HC, GC = [],[]
    for k in range(110):
        HC.append(0)
        GC.append(0)

    for i in range(nc):
        s = 0.
        WQ = 0.
        T = 1.

        for J in range(MK[i]):
            jj = int(BB[i][J])
            kk = CC[i][J]
            T = T * F[jj-1] ** abs(kk)
            s = s + U[jj-1] * kk
            WQ = WQ + V[jj-1][5] * kk
            ZQ = s
      
        h[i] = T * h[i]
        s = s - G[i]
        if s < 0: s = s + 360.
        G[i] = s
        try: 
            W[i] = WQ * DT
        except IndexError:
            W.append( WQ * DT )
        HC[i] = T * HC[i]
        ZQ = ZQ - GC[i]
        if ZQ < 0: ZQ = ZQ + 360.
        GC[i] = ZQ

    x, Y2, y = [],[],[]
    MM = 0
    for i in range(n):
        s = 0.
        ZQ = 0.

        for j in range(nc):
            AA = G[j] * 0.017453
            s = s + h[j] * np.cos(AA)
            G[j] = G[j] + W[j]
            AC = GC[j] * 0.017453
            ZQ = ZQ + HC[j] * np.cos(AC)
            GC[j] = GC[j] + W[j]

        x.append(s + NM)
        Y2.append(x[i])
        y.append(ZQ + MM)

    x  = np.array(x, dtype=np.float32)
    x = x/100.
    h = x[3:-3]
    hours = np.arange(24)
    years, months, days = 0*hours+an, 0*hours+Mesl, 0*hours+int(dd)
    time = []
    for year, month, day, hour in zip(years, months, days, hours):
        time.append( dt.datetime(year, month, day, hour) )

    time = mpldates.date2num(time)
    time2 = np.linspace(time[0], time[-1], 500)

    interp = interp1d(time, h, kind='cubic')
    h2 = interp(time2)

    dh = np.gradient(h2)
    dhSign = dh > 0
    # gathering pairs
    pairs = []
    for k in range(len(dh)-1):
        pairs.append([dhSign[k], dhSign[k+1]])

    f = []
    for k in range(len(pairs)):
        if pairs[k] == [True, False] or pairs[k] == [False, True]:
            f.append(k)

    datas = mpldates.num2date(time2[f])
    hora = []
    for data in datas:
        hora.append("%02i:%02i" %(data.hour, data.minute))
    altura = h2[f]
    altura = ['%.1f' % a for a in altura]

    return infoList, hora, altura, time2, h2



def plotMare(t, h, station, date, rise_set_times, dst, web2pyPath):
    # Process rise and set sun times to fit at the graph
    rise_time, set_time = rise_set_times
    if dst==True:
        r, s = rise_time, set_time
        rise_time = dt.datetime(r.year, r.month, r.day, r.hour-2, r.minute, r.second)
        set_time = dt.datetime(s.year, s.month, s.day, s.hour-2, s.minute, s.second)
    else:
        r, s = rise_time, set_time
        rise_time = dt.datetime(r.year, r.month, r.day, r.hour-3, r.minute, r.second)
        set_time = dt.datetime(s.year, s.month, s.day, s.hour-3, s.minute, s.second)
        
    rise_time_dec = mpldates.date2num(rise_time)
    set_time_dec = mpldates.date2num(set_time)
    verts = [(t.min(), 0)] + zip(t, h) + [(t.max(), 0)]
    poly = Polygon(verts, facecolor='b', edgecolor='b',alpha=0.3)
    fig = plt.figure(figsize=(7,3), facecolor='w')
    p = plt.subplot(111)
    plt.plot_date(t, h, 'b')
    p.add_patch(poly)
    p.xaxis.set_major_formatter(mpldates.DateFormatter('%Hh'))
    plt.grid()
    # plt.xlabel("Horas do Dia")
    plt.ylabel(u"Nível do Mar [m]")
    data = "%s/%s/%s" %(date.day, date.month, date.year)
    plt.title(station.decode('utf-8','replace') +' : '+data, fontsize=10, fontweight='bold')
    fig.autofmt_xdate()
    ylim = p.get_ylim()
    now = mpldates.date2num(dt.datetime.now())
    plt.axvline( x=now, color='r' )
    plt.axvspan( t[0], rise_time_dec,facecolor='0.8',edgecolor='0.8',zorder=0)
    plt.axvspan(set_time_dec, t[-1],facecolor='0.8',edgecolor='0.8',zorder=0)
    plt.plot( [ now, now ], [ 0, ylim[1] ], 'r' )
    p.set_xlim(t[0], t[-1])
    plt.savefig(web2pyPath + 'static/images/tide.png', dpi=96)
    plt.close('all')

    
