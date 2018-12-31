"""Particle displacement of a pulsating sphere."""
import sfs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, patches

# Pulsating sphere
center = [0, 0, 0]
radius = 0.25
amplitude = 0.05  # amplitude of the surface displacement
f = 750  # frequency
omega = 2 * np.pi * f  # angular frequency
ka = sfs.util.wavenumber(omega) * radius

# Temporal sampling for animation
fs = f * 10  # sampling frequency
L = int(np.round(fs / f))  # number of frames corresponding to one period
t = np.arange(L) / fs  # time

# Uniform grid
xmin, xmax = -1, 1
ymin, ymax = -1, 1
grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.02)

# Particle displacement
velocity = sfs.mono.source.pulsating_sphere_velocity(omega,
                                                     center,
                                                     radius,
                                                     amplitude,
                                                     grid)
displacement = sfs.util.displacement(velocity, omega)

# Animation
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(xlim=(-xmax, xmax), ylim=(-ymax, ymax))
patch = ax.add_patch(patches.Circle(center[:2],
                                    radius=radius,
                                    facecolor='RoyalBlue',
                                    linestyle='None',
                                    alpha=0.4))
scat = sfs.plot.particles(grid + displacement,
                          s=15,
                          c='gray',
                          marker='.')
text = ax.text(0.9 * xmax, 0.9 * ymin, '<t>',
               fontsize=16,
               horizontalalignment='right',
               verticalalignment='center')
ax.set_title('$a={:0.2f}$ m, $f={:g}$ Hz ($ka={:0.1f}$)'
             .format(radius, f, ka), fontsize=16)
ax.set_xlabel('$x$ / m')
ax.set_ylabel('$y$ / m')
ax.axis('off')


def animate(i, scat, patch, text):
    """Update frame."""
    phase_shift = np.exp(1j * omega * t[i])
    X = grid + (displacement * phase_shift).apply(np.real)
    X = np.column_stack([X[0].flatten(), X[1].flatten()])
    scat.set_offsets(X)
    patch.set_radius(radius + amplitude * np.real(phase_shift))
    text.set_text('{:0.2f} ms'.format(t[i] * 1000))
    return scat, patch, text


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=L,
                              fargs=(scat, patch, text))
ani.save('pulsating_sphere_displacement.gif',
         fps=10,
         dpi=80,
         writer='imagemagick')
