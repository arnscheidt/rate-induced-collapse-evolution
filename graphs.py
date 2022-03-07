#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# some function definitions

def read_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            data.append([float(x) for x in row])     
    return data

def plot_data(types,n,tstart,tint,skip,end):
    # skip is an integer that means: only plot a point for every "skip" timesteps
    # if end=0, do auto end assignment

    alpha = 1
    size = 20 
    start = 0
    
    auto_end = False
    if end==0:
        auto_end = True
        
        end = 0
        temp_end = 0

        for i in range(1,len(types)//skip):
            temp_end = np.amax(n[i])
            if temp_end>end:
                end = temp_end

    plt.scatter(np.ones_like(types[0])*tstart,types[0],c=n[0],cmap='copper_r',alpha=alpha,s=size,vmin=start)
    plt.xlabel('Time')
    plt.ylabel(r'$x$')
    for i in range(1,len(types)//skip):
        i = i*skip
        plt.scatter(np.ones_like(types[i])*(tstart+i*tint),types[i],c=n[i],cmap='copper_r',alpha=alpha,s=size,vmin=start,vmax=end)

    if auto_end:
        cbar = plt.colorbar()
    else:
        cbar = plt.colorbar(extend='max')

    cbar.set_label('Number of individuals')


# FIGURE 5: EVOLUTION TO QESS
st = read_data("qess_types.csv")
sn = read_data("qess_n.csv")
plot_data(st,sn,0,1000,5,600)

# FIGURE 6: RATE-INDUCED EXTINCTION
fig = plt.figure()
vlinecol = (0,0,0)
plt.subplot(211)
plt.axvline(50000,color=vlinecol,alpha=0.5)
plt.axvline(80000,color=vlinecol,alpha=0.5)

st = read_data("ss_ab_rit_recovery_types.csv")
sn = read_data("ss_ab_rit_recovery_n.csv")

plot_data(st,sn,0,100,10,0)
plt.xlabel('')
plt.gca().set_xticklabels([])
plt.xlim(0,120000)
plt.ylim(0,330)
plt.text(25000,290,r'$b=1.0\times10^{-5}$',horizontalalignment='center')
plt.text(65000,290,r'linear ramp',horizontalalignment='center')
plt.text(100000,290,r'$b=1.3\times10^{-4}$',horizontalalignment='center')
plt.text(100000,150,r'$\textit{persistence}$',horizontalalignment='center')

plt.subplot(212) 
plt.axvline(50000,color=vlinecol,alpha=0.5)
plt.axvline(70000,color=vlinecol,alpha=0.5)

st = read_data("ss_ab_rit_extinction_types.csv")
sn = read_data("ss_ab_rit_extinction_n.csv")
plot_data(st,sn,0,100,10,0)
plt.xlim(0,120000)
plt.ylim(0,330)
plt.text(25000,290,r'$b=1.0\times10^{-5}$',horizontalalignment='center')
plt.text(60000,290,r'linear ramp',horizontalalignment='center')
plt.text(95000,290,r'$b=1.3\times10^{-4}$',horizontalalignment='center')
plt.text(95000,150,r'$\textit{extinction}$',horizontalalignment='center')

fig.text(0.02,0.95,r'\textbf{A}',fontsize=20)
fig.text(0.02,0.5,r'\textbf{B}',fontsize=20)
fig.subplots_adjust(left=0.086,bottom=0.11,right=1,top=0.967,wspace=0.2,hspace=0.145)

# FIGURE 7: CRITICAL RATE FOR EXTINCTION 
critical_rate = np.loadtxt("critical_rate.csv",delimiter=',')
rate = np.logspace(-9,-6,20)
time = np.logspace(2,5,20)
t2,r2 = np.meshgrid(time,rate)

# set up colormap to start at white 
lev = 21 
colmap_arr = mpl.cm.Reds(np.linspace(0,256*0.9,lev).astype(int))
colmap_arr[0,:] = [1,1,1,1]
colmap = mpl.colors.ListedColormap(colmap_arr, name='colmap', N=colmap_arr.shape[0])

plt.figure()
ax1 = plt.subplot(111)
lev = np.linspace(0,1,lev)
cplt = plt.contourf(t2,r2,critical_rate,lev,cmap=colmap)
cbar = plt.colorbar(cplt)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(10**(-9),10**(-6.001))

plt.xlabel(r'Timescale')
plt.ylabel(r'Rate of change of forcing')
cbar.set_label('Probability of extinction')
cbar.set_ticks([0,0.25,0.5,0.75,1])

# plot scaling law line
scaling_col = (0.3,0.3,0.9)
scaling_x = np.logspace(2.3,3.8,10)
scaling_y = scaling_x**(-1)*10**(-4.5)
ax1.plot(scaling_x,scaling_y,linewidth=3,color=scaling_col)
ax1.text(10**2.7,10**(-7.9),r'$\propto \tau^{-1}$',color=scaling_col)

ax2 = ax1.twiny() 
ax2.set_xlim(2*10**2,10**5)
ax2.set_xscale('log')
ax2.set_xticks([2*10**4])
ax2.set_xticklabels([r'$\simeq \tau_{\rm ev}$'])

plt.show()
