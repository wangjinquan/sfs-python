"""
Create some examples in the time domain.

Simulate and plot impulse behavior for NFC-HOA.

"""
import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
fs = sfs.defs.fs
grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.002)
my_cmap = 'YlOrRd'
my_cmap = 'Blues'
N = 60  # number of secondary sources
R = 1.5  # radius of spherical/circular array
x0, n0, a0 = sfs.array.circular(N, R)  # get secondary source positions

# Signals
L = 1024

dirac = np.zeros(L)
dirac[0] = 1

f0 = 2333
time = np.linspace(0,L,num=L)*(1/fs)
harm = (1/4)*np.cos(2*np.pi*f0*time)

signal = dirac

# NFC-HOA setup
max_order = None
max_order = sfs.mono.drivingfunction._max_order_circular_harmonics(N, max_order)

# POINT SOURCE
#xs = [1.5, 1.5, 0]  # position of virtual source
#t = (np.sqrt(2)*1.5)/sfs.defs.c
#delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_point(x0,R,xs,max_order=max_order,normalize=True)

# PLANE WAVE
npw = [0,-1,0]  
t = 0/sfs.defs.c
delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_plane(x0,R,npw,max_order=max_order, normalize=True)
t = 0

#t += L/fs/2

# Driving signals
d, t_offset = sfs.time.drivingfunction.nfchoa_driving_signals(delay, weight, sos, phaseshift, signal, max_order=max_order)

plt.figure(figsize=(4,4))
plt.imshow(sfs.util.db(d.T),interpolation=None,cmap='Blues')
plt.axis('tight')
plt.colorbar()
plt.clim([-120,0])

plt.figure(figsize=(4,4))
plt.imshow(np.real(d.T),interpolation=None,cmap='coolwarm')
plt.axis('tight')
plt.colorbar()
plt.clim([-0.1,0.1])

t -= t_offset
scale = 1

# Synthesized sound field
a0 = np.ones(len(x0))
p = sfs.time.soundfield.p_array(x0, d.T, a0, t, grid)

plt.figure(figsize=(4, 4))
im = sfs.plot.level(p, grid, cmap='Blues')
sfs.plot.loudspeaker_2d(x0, n0)
plt.grid()
sfs.plot.virtualsource_2d(npw)
sfs.plot.virtualsource_2d([0,1], npw, type='plane')
#plt.title('impulse_ps_nfchoa_25d')
#plt.savefig('impulse_ps_nfchoa_25d.eps')
#plt.savefig('impulse_ps_nfchoa_25d.png')

plt.figure(figsize=(4, 4))
sfs.plot.soundfield(scale*p, grid, cmap='coolwarm')
sfs.plot.loudspeaker_2d(x0, n0)
plt.grid()
plt.title('$M=%d$'%max_order)
#sfs.plot.virtualsource_2d(npw)
sfs.plot.virtualsource_2d([0, 1], npw, type='plane')
#plt.title('impulse_ps_nfchoa_25d')
#plt.savefig('impulse_ps_nfchoa_25d_M%d.eps'%(max_order), bbox_inches='tight')
#plt.savefig('impulse_ps_nfchoa_25d_M%d.png'%(max_order), bbox_inches='tight')
#plt.savefig('impulse_pw_nfchoa_25d_M%d.eps'%(max_order), bbox_inches='tight')
#plt.savefig('impulse_pw_nfchoa_25d_M%d.png'%(max_order), bbox_inches='tight')

