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
signal = dirac, fs, 0

# NFC-HOA order
max_order = None
#max_order = 150
#max_order = 29

# POINT SOURCE
#xs = np.r_[1.5, 1.5, 0]  # position
#delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_point(x0, R, xs, fs, max_order=max_order)
#t = np.linalg.norm(xs) / sfs.defs.c

# PLANE WAVE
npw = [0, -1, 0]
delay, weight, sos, phaseshift = sfs.time.drivingfunction.nfchoa_25d_plane(x0, R, npw, fs, max_order=max_order)
t = 0

# Driving signals
d, fs, t_offset = sfs.time.drivingfunction.nfchoa_driving_signals(delay, weight, sos, phaseshift, signal)

# Synthesized sound field
p = sfs.time.soundfield.p_array(x0, (d, fs, t_offset), a0, t, grid)

plt.figure()
sfs.plot.level(p, grid, cmap='Blues')
sfs.plot.loudspeaker_2d(x0, n0)
#sfs.plot.virtualsource_2d(xs, type='point')
sfs.plot.virtualsource_2d([0, 0], ns=npw, type='plane')
plt.savefig('pw_level.png')

plt.figure()
sfs.plot.soundfield(p, grid, cmap='coolwarm')
sfs.plot.loudspeaker_2d(x0, n0)
#sfs.plot.virtualsource_2d(xs, type='point')
sfs.plot.virtualsource_2d([0, 0], ns=npw, type='plane')
plt.savefig('pw_soundfield.png')
