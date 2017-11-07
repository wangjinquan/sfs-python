"""
Create some examples in the time domain.

Simulate and plot impulse behavior for NFC-HOA.

"""
import numpy as np
import matplotlib.pyplot as plt
import sfs

# simulation parameters
fs = 44100
grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.002)
N = 60      # number of secondary sources
R = 1.5     # radius of spherical/circular array
x0, n0, a0 = sfs.array.circular(N, R)  # get secondary source positions

# Signal
L = 1024
dirac = np.zeros(L)
dirac[0] = 1
signal = dirac

# NFC-HOA order
max_order = None
#max_order = 150
#max_order = 29

# POINT SOURCE
#xs = np.r_[1.5, 1.5, 0]  # position
#delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_point(x0, R, xs, max_order=max_order, normalize=True)
#t = np.linalg.norm(xs) / sfs.defs.c

# PLANE WAVE
npw = [0, -1, 0]
delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_plane(x0, R, npw, max_order=max_order, normalize=True)
t = 0


# Driving signals
d, fs, t_offset = sfs.time.drivingfunction.nfchoa_driving_signals(delay, weight, sos, phaseshift, signal, max_order=max_order)

plt.figure()
plt.imshow(sfs.util.db(d), interpolation='None', cmap='Blues')
plt.axis('tight')
plt.colorbar()
plt.clim([-120,0])

plt.figure()
plt.imshow(np.real(d), interpolation='None', cmap='coolwarm')
plt.axis('tight')
plt.colorbar()
plt.clim([-0.1,0.1])


# Synthesized sound field
a0 = np.ones(len(x0))
p = sfs.time.soundfield.p_array(x0, (d, fs, t_offset), a0, t, grid)

plt.figure()
sfs.plot.level(p, grid, cmap='Blues')
sfs.plot.loudspeaker_2d(x0, n0)

plt.figure()
sfs.plot.soundfield(p, grid, cmap='coolwarm')
sfs.plot.loudspeaker_2d(x0, n0)
