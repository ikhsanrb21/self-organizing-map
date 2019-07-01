import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as random
import math
import numpy as np

#membuat variable untuk menampung dataset
vector = []

#melakukan load dataset yg berformat csv
with open('dataset.csv', newline='') as datatrain:
        spamreader = csv.reader(datatrain, delimiter=',')
        for a in spamreader:
            #memasukan dataset kedalam variable yg telah dibuat 
            vector.append([float(a[0]),float(a[1])])

#membuat variable yg berisi nilai random yg dibangkitkan sebanyak 15
#nilai random dibatasi hanya antara 10 sampe 11
xr = np.random.rand(15,2)+10

#membuat variable lalu mendefinisikannya
lr = 0.6
tlr = 15
sig = 0.2
tsig = 50/np.log(15)
t = 0

#melakukan looping sebanyak iterasi
while t != 50:
    #melakukan reduksi nilai learning rate dan sigma ditiap iterasinya
    lrt = lr*math.exp(-t/tlr)
    sigt = sig*math.exp(-t/tsig)
    #melakukan looping terhadap dataset
    for b in range(len(vector)):
        d = []
        #melakukan looping terhadap nilai random yg dibangkitkan
        for c in range(len(xr)):
            #melakukan perhitungan jarak antara tiap data pada dataset dengan nilai random
            d_temp = ((vector[b][0]-xr[c][0])**2 + (vector[b][1]-xr[c][1])**2)
            d.append(d_temp)
        #mencari indeks dari nilai terkecil
        da = np.argmin(d)
        xrmin = xr[da]
        #melakukan looping terhadap nilai random yg dibangkitkan
        for e in range(len(xr)):
            #melakukan looping terhadap dimensi data
            for j in range(2):
                #melakukan perhitungan topologi
                topo = (-(math.sqrt((xrmin[0]-xr[e][0])**2 + (xrmin[1]-xr[e][1])**2)))/(2*(sigt**2))
                topo1 = math.exp(topo)
                #melakukan perhitungan delta nilai random
                dw = lrt*topo1*(vector[b][j]-xr[e][j])
                #melakukan update nilai
                xr[e][j] = xr[e][j] + dw
    t+=1

#melakukan plot data kedalam grafik
for i in vector:
    plt.scatter(i[0],i[1],c='red')
for i in xr:
    plt.scatter(i[0],i[1],c='yellow')
plt.show()