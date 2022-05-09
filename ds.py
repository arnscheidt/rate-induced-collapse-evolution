#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import ode,odeint

plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':16})
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

rstar = 1
a = 1
b = 2
c = 3  
xstar = 5
k = 0.001
epsilon=0.0001

def model(t,X,F):
    n,x = X

    ndot = n*(rstar-a*(F*xstar-x)**2-b+c*n-n**2)
    xdot = -2*np.heaviside(n-epsilon,0)*k*a*(x-F*xstar)
    return [ndot,xdot]

#####################################################
## FIGURE 1 - NULLCLINES
#####################################################

n = np.linspace(0,100,100)

x_nullcline = np.linspace(0,10,1000000)
n_nullcline_1 = (c+np.sqrt(c**2 - 4*(b-rstar+a*(xstar - x_nullcline)**2)))/2
n_nullcline_2 = (c-np.sqrt(c**2 - 4*(b-rstar+a*(xstar - x_nullcline)**2)))/2

arrow_color = (0,0,0)
arrow_overhang = 0.2
arrow_head_width = 0.07
nullc_arrow_head_width = 0.12
arrow_line_width = 1.1
scatter_width = 70

nullc_width = 3 
n_nullc_color = (0.5,0.5,0.5)
n_nullc_text_color = (0.3,0.3,0.3)
x_nullc_color = (0.7,0.0,0.0)

ax1 = plt.subplot(111)
plt.axhline(xstar,linewidth=nullc_width,color=x_nullc_color)
plt.axvline(0,linewidth=nullc_width,color=(0,0,0))
plt.plot(n_nullcline_1,x_nullcline,linewidth=nullc_width,color=n_nullc_color)
plt.plot(n_nullcline_2,x_nullcline,linewidth=nullc_width,color=n_nullc_color)

plt.scatter(2.618,5,s=scatter_width,color=(0,0,0),zorder=20)
plt.scatter(0.382,5,s=scatter_width,color=(1,1,1),zorder=20,edgecolors=(0,0,0))

plt.xlabel(r'$n$ (population, fast variable)')
plt.ylabel('$x$ (trait, slow variable)')
plt.ylim(2.9,7.1)
plt.xlim(-0.01,3.6)

# text for nullclines
plt.text(1.95,3.35,r'$\displaystyle \frac{dn}{dt}=0$',color=n_nullc_text_color)
plt.text(2.9,5.3,r'$\displaystyle \frac{dx}{dt}=0$',color=x_nullc_color)

ax1.arrow(1.55,3.3,-0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(1.55,6.7,-0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(3.4,4,-0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(3.4,6,-0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)

ax1.arrow(1.25,4.4,0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(1.25,5.6,0.5,-0,color=arrow_color,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)

ax1.arrow(3.35,5,-0.1,-0,color=x_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(0.35,5,-0.1,-0,color=x_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(0.7,5,0.1,-0,color=x_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(2.0,5,0.1,-0,color=x_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(2.15,5.9,0.01,-0.008,color=n_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(2.15,4.1,0.01,0.008,color=n_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)

ax1.arrow(0.85,5.9,-0.01,-0.0075,color=n_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax1.arrow(0.85,4.1,-0.01,0.0075,color=n_nullc_color,head_width=nullc_arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)

#####################################################
# FIGURE 2 - CANARD TRAJECTORIES
#####################################################

plt.figure()

# trajectory 1 - recovery
t = np.linspace(0,500,10000)
dt = t[1]-t[0]
F = np.ones_like(t)
r = ode(model).set_integrator("lsoda")

f1 = np.zeros((len(t),2))
f1[0,:] = (2.5,3.875)
r.set_initial_value(f1[0,:]).set_f_params(F[0])

for i in range(1,len(t)):
    r.set_f_params(F[i])
    r.integrate(r.t+dt)
    f1[i] = r.y

# trajectory 2 - canard 1
f2 = np.zeros((len(t),2))
f2[0,:] = (2.5,3.861058754)
r.set_initial_value(f2[0,:]).set_f_params(F[0])

for i in range(1,len(t)):
    r.set_f_params(F[i])
    r.integrate(r.t+dt)
    f2[i] = r.y

# trajectory 3 - extinction 
f3 = np.zeros((len(t),2))
f3[0,:] = (2.5,3.85)
r.set_initial_value(f3[0,:]).set_f_params(F[0])

for i in range(1,len(t)):
    r.set_f_params(F[i])
    r.integrate(r.t+dt)
    f3[i] = r.y

# trajectory 4 - canard 2
f4 = np.zeros((len(t),2))
f4[0,:] = (2.5,3.86105875392)
r.set_initial_value(f4[0,:]).set_f_params(F[0])

for i in range(1,len(t)):
    r.set_f_params(F[i])
    r.integrate(r.t+dt)
    f4[i] = r.y



arrow_color = (0,0,0)
arrow_overhang = 0.2
arrow_head_width = 0.05
arrow_width = 0.01
arrow_line_width = 0.0001 
traj_width=3
can_width=3

nullc_width = 7 
n_nullc_color = (0.7,0.7,0.7)
n_nullc_text_color = (0.5,0.5,0.5)

t1_col = (0,0.7,0)
t2_col = (0.2,0.2,0.8)
t3_col = (0,0,0)

ax1 = plt.subplot(111)
plt.plot(n_nullcline_1,x_nullcline,linewidth=nullc_width,color=n_nullc_color)
plt.plot(n_nullcline_2,x_nullcline,linewidth=nullc_width,color=n_nullc_color)
plt.text(1.41,3.92,r'$\displaystyle \frac{dn}{dt}=0$',color=n_nullc_text_color,fontsize=20)

plt.plot(f1[:,0],f1[:,1],linewidth=traj_width,color=t1_col)
plt.plot(f2[:,0],f2[:,1],linewidth=can_width,color=t2_col)
plt.plot(f3[:,0],f3[:,1],linewidth=traj_width,color=t3_col)
plt.plot(f4[:,0],f4[:,1],linewidth=can_width,color=t2_col)


plt.text(1.7,3.89,'evolutionary rescue',rotation=62,color=t1_col)
plt.text(1.05,3.88,'canard trajectories',rotation=-65,color=t2_col)
plt.text(0.95,3.867,'extinction',rotation=-5,color=t3_col)

plt.xlabel(r'$n$ (population)')
plt.ylabel(r'$x$ (trait)')

plt.xlim(0.85,2.25)
plt.ylim(3.84,3.95)

ax2 = ax1.twinx()
ax2.tick_params(right=False,labelright=False)
ax2.arrow(2,1.308,-0.01,0.001,color=t1_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(2,1.130,-0.01,0.001,color=t2_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(2,0.989,-0.01,0.001,color=t3_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(1.76,1.8,0.01,0.03,color=t1_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(1.3,1.564,-0.015,0.04,color=t2_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(1.08,1.17,-0.01,0.002,color=t3_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(1,2.177,-0.01,0.002,color=t2_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)
ax2.arrow(1.36,2.177,0.01,0.0015,color=t2_col,width=arrow_width,head_width=arrow_head_width,linewidth=arrow_line_width,overhang=arrow_overhang)

ax2.set_ylim(0.85,2.25)

#####################################################
# FIGURE 3 - RATE-INDUCED EXTINCTION 
#####################################################

# generate trajectories

total_duration = 4000
total_steps = 2000
t = np.linspace(0,total_duration,total_steps)

start=1
end=1.6
duration1=1400
duration2=1000
line_width=3

t1_col = (0,0.7,0)
t2_col = (0.2,0.2,0.8)

F_rescue = np.concatenate((np.linspace(start,end,int(duration1*total_steps/total_duration)),end*np.ones(int(total_steps - duration1*total_steps/total_duration))))
F_extinct = np.concatenate((np.linspace(start,end,int(duration2*total_steps/total_duration)),end*np.ones(int(total_steps - duration2*total_steps/total_duration))))

dt = t[1]-t[0]
r = ode(model).set_integrator("dop853")

f1 = np.zeros((len(t),2))
f1[0,:] = (2.618,5) # stable population
r.set_initial_value(f1[0,:]).set_f_params(F_rescue[0])

for i in range(1,len(t)):
    r.set_f_params(F_rescue[i])
    r.integrate(r.t+dt)
    f1[i] = r.y

f2 = np.zeros((len(t),2))
f2[0,:] = (2.618,5) # stable population
r.set_initial_value(f2[0,:]).set_f_params(F_extinct[0])

for i in range(1,len(t)):
    r.set_f_params(F_extinct[i])
    r.integrate(r.t+dt)
    f2[i] = r.y

fig = plt.figure()
plt.subplot(221)
plt.plot(t,f1[:,0],linewidth=line_width,color=t1_col)
plt.ylabel(r'$n$ (population)')
plt.gca().set_xticklabels([])
plt.title(r'\textbf{Evolutionary rescue}')
plt.ylim(-0.1,2.7)

plt.subplot(223)
plt.plot(t,F_rescue*xstar,linewidth=line_width,color=t1_col)
plt.ylabel(r'$x^*$ (optimal trait)')
plt.xlabel(r'Time')

plt.subplot(222)
plt.plot(t,f2[:,0],linewidth=line_width,color=t2_col)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.title(r'\textbf{Rate-induced extinction}')
plt.ylim(-0.1,2.7)

plt.subplot(224)
plt.plot(t,F_extinct*xstar,linewidth=line_width,color=t2_col)
plt.xlabel(r'Time')
plt.gca().set_yticklabels([])

fig.subplots_adjust(left=0.076,bottom=0.124,right=0.979,top=0.91,wspace=0.15,hspace=0.148)
fig.text(0.02,0.924,r'\textbf{A}',fontsize=20)
fig.text(0.52,0.924,r'\textbf{B}',fontsize=20)

f3d = plt.figure() 
tw = 3
ax = f3d.add_subplot(111, projection='3d') 
ax.plot(f1[:,0],F_rescue*xstar,f1[:,1],linewidth=tw,color=t1_col,zorder=10,label='evolutionary rescue')
ax.plot(f2[:,0],F_extinct*xstar,f2[:,1],linewidth=tw,color=t2_col,zorder=10,label='rate-induced extinction')

def nullc(n,xstar):
    # returns x as a function of n and xstar
    x = xstar - np.sqrt(1/a * (-n**2 + c*n - b + rstar))
    return x

N = np.linspace(0.5,2.618,500)
Xstar = np.linspace(4.8,8,501)
N2,Xstar2 = np.meshgrid(N,Xstar,indexing='xy')
X2 = nullc(N2,Xstar2)

# get fold location 
xpoints = [np.amin(X2[0,:]),np.amin(X2[-1,:])]

ax.plot_wireframe(N2,Xstar2,X2,color=(0.8,0.8,0.8),zorder=-10)

ax.plot([1.5,1.5],[4.8,8],[xpoints[0],xpoints[1]],linewidth=2,linestyle=':',color=(0,0,0),zorder=5)

ax.plot([2.618,2.618],[4.8,8],[4.8,8],linewidth=tw,color=(0,0,0),zorder=10,label='stable fixed point')

ax.set_zlim(8.1,3.9) 
 
ax.set_zlabel(r'$x$ (trait)') 
ax.set_xlabel(r'$n$ (population)') 
ax.set_ylabel(r'$x^*$ (optimal trait)')

ax.xaxis.labelpad=10
ax.yaxis.labelpad=10
ax.zaxis.labelpad=7.5

for axis in ax.w_xaxis, ax.w_yaxis, ax.w_zaxis:
  axis.pane.set_visible(False)

f3d.text(0.22,0.75,r'\textbf{C}',fontsize=20)



#####################################################
# FIGURE 4 - CRITICAL RATE FOR EXTINCTION
#####################################################

# for each given timescale, do a binary-style search for the threshold in log space
def test_extinction(rate,timescale,total_time,total_steps):
    t = np.linspace(0,total_time,total_steps)

    timescale_tsteps = int(timescale*total_steps/total_time)
    remaining_tsteps = total_steps-timescale_tsteps

    end = 1 + rate*timescale

    F = np.concatenate((np.linspace(1,end,timescale_tsteps),end*np.ones(remaining_tsteps)))
    
    dt = t[1]-t[0]
    r = ode(model).set_integrator("dop853",nsteps=10000)
    
    f = np.zeros((len(t),2))
    f[0,:] = (2.618,5) # stable population
    r.set_initial_value(f[0,:]).set_f_params(F[0])
    
    for i in range(1,len(t)):
        r.set_f_params(F[i])
        r.integrate(r.t+dt)
        f[i] = r.y

    return f[-1,0]==0

## actual search
#tau_vec = np.logspace(0,5,21)
#critical_rate = np.zeros_like(tau_vec)
#rate_range = [-4,0] # in log10 coords, assume critical rate lies in this range
#
#for itau in range(0,len(tau_vec)):
#    temp_range = rate_range[:]
#    for i in range(0,10):
#        trial = (temp_range[0]+temp_range[1])/2
#        print(temp_range,tau_vec[itau])
#        if test_extinction(10**(trial),tau_vec[itau],tau_vec[itau]+100,int(tau_vec[itau])+100):
#            temp_range[1] = trial
#        else:
#            temp_range[0] = trial
#    critical_rate[itau] = 10**trial
#
#save_data = np.zeros((2,len(tau_vec)))
#save_data[0,:] = tau_vec
#save_data[1,:] = critical_rate
#np.save("sm_critical_rate",save_data)
save_data = np.load("sm_critical_rate.npy")

plt.figure()
linecol = (0.7,0,0)
lw = 4
ax1 = plt.subplot(111)
plt.plot(save_data[0,:],save_data[1,:],color=linecol,linewidth=lw)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Timescale')
ax1.set_ylabel(r'Rate of change of forcing')
ax1.set_xlim(10**(-0.2),10**5)

ax1.text(10**2.5,10**(-1),r'\textit{extinction}')
ax1.text(10**0.5,10**(-2.5),r'\textit{survival}')

scaling_col = (0.3,0.3,0.9)
scaling_x = np.logspace(0.5,3,10)
scaling_y = scaling_x**(-1)*1.3
ax1.plot(scaling_x,scaling_y,linewidth=2,color=scaling_col)
ax1.text(10**1.3,10**(-1),r'$\propto \tau^{-1}$',color=scaling_col)

ax2 = ax1.twiny() 
ax2.set_xlim(10**(-0.2),10**5)
ax2.set_xscale('log')
ax2.set_xticks([10**0,500])
ax2.set_xticklabels([r'$\simeq \tau_{\rm ec}$',r'$\simeq \tau_{\rm ev}$'])

plt.show()
