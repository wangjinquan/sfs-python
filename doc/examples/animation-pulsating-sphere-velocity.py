"""Particle velocity of a pulsating sphere.
"""
import sfs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, patches

# Pulsating sphere
x0 = [0, 0, 0]  # position
radius = 0.25  # radius
amplitude = 0.05  # amplitude of the surface displacement
f = 750  # frequency
omega = 2 * np.pi * f  # angular frequency
ka = sfs.util.wavenumber(omega) * radius

# Temporal sampling
fs = f * 15  # sampling frequency
L = int(np.round(fs / f))  # number of frames for one period
t = np.arange(L) / fs  # time

# Uniform grid
xmin, xmax = -1, 1
ymin, ymax = -1, 1
grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.04)

# Particle velocity
velocity = sfs.mono.source.pulsating_sphere_velocity(omega,
                                                     x0,
                                                     radius,
                                                     amplitude,
                                                     grid)

# Animation
clim = -omega * amplitude, omega * amplitude
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(xlim=(-xmax, xmax), ylim=(-ymax, ymax))
patch = ax.add_patch(patches.Circle(x0[:2],
                                    radius=radius,
                                    facecolor='RoyalBlue',
                                    linestyle='None',
                                    alpha=0.4))
quiv = sfs.plot.vectors(velocity, grid, clim=clim)
text = ax.text(0.9 * xmax, 0.9 * ymin, '<t>',
               fontsize=16,
               horizontalalignment='right',
               verticalalignment='center')
ax.set_title('$a={:0.2f}$ m, $f={:g}$ Hz ($ka={:0.1f}$)'
             .format(radius, f, ka), fontsize=16)
ax.set_xlabel('$x$ / m')
ax.set_ylabel('$y$ / m')
ax.axis('off')


def animate(i, quiv, patch, text):
    phase_shift = np.exp(1j * omega * t[i])
    quiv.set_UVC(*(velocity * phase_shift).apply(np.real)[:2])
    patch.set_radius(radius + amplitude * np.real(phase_shift))
    patch.set_alpha(0.5 - 0.1 * np.real(phase_shift))
    text.set_text('{:0.2f} ms'.format(t[i] * 1000))
    return quiv, patch, text


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=L,
                              fargs=(quiv, patch, text))
ani.save('pulsating_sphere_velocity.gif',
         fps=10,
         dpi=80,
         writer='imagemagick')
