"""Animations of pulsating sphere."""
import sfs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, patches


def init_2d_plot():
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    patch = ax.add_patch(patches.Circle(center[:2],
                                        radius=radius,
                                        facecolor='RoyalBlue',
                                        linestyle='None',
                                        alpha=0.4))
    text = ax.text(0.05 * xmin + 0.95 * xmax,
                   0.95 * ymin + 0.05 * ymax,
                   '',
                   fontsize=16,
                   horizontalalignment='right',
                   verticalalignment='center')
    ax.set_title('$a={:0.2f}$ m, $f={:g}$ Hz ($ka={:0.1f}$)'
                 .format(radius, f, ka), fontsize=16)
    ax.set_xlabel('$x$ / m')
    ax.set_ylabel('$y$ / m')
    ax.axis('off')
    return fig, ax, patch, text


def update_displacement(scat, d):
    d = d.apply(np.real)
    d = np.column_stack([d[0].flatten(), d[1].flatten()])
    scat = scat.set_offsets(d)
    return scat


def update_velocity(quiv, v):
    v = v[:2].apply(np.real)
    quiv.set_UVC(*v)
    return quiv


def update_pressure(im, p):
    im.set_array(np.real(p))
    return im


def update_sphere(patch, radius, alpha=0.5):
    patch.set_radius(radius)
    patch.set_alpha(alpha)
    return patch


def update_time(text, t):
    text = text.set_text('{:0.3f} ms'.format(t * 1000))
    return text


def update_frame_displacement(i, displacement, scat, patch, text):
    phase_shift = np.exp(1j * omega * t[i])
    scat = update_displacement(scat, grid + displacement * phase_shift)
    patch = update_sphere(patch, radius + amplitude * np.real(phase_shift),
                          alpha=0.5 - 0.1 * np.real(phase_shift))
    text = update_time(text, t[i])
    return scat, patch, text


def update_frame_velocity(i, velocity, quiv, patch, text):
    phase_shift = np.exp(1j * omega * t[i])
    quiv = update_velocity(quiv, velocity * phase_shift)
    patch = update_sphere(patch, radius + amplitude * np.real(phase_shift),
                          alpha=0.5 - 0.1 * np.real(phase_shift))
    text = update_time(text, t[i])
    return quiv, patch, text


def update_frame_pressure(i, pressure, im, patch, text):
    phase_shift = np.exp(1j * omega * t[i])
    im = update_pressure(im, pressure * phase_shift)
    patch = update_sphere(patch, radius + amplitude * np.real(phase_shift),
                          alpha=0.5 - 0.1 * np.real(phase_shift))
    text = update_time(text, t[i])
    return im, patch, text


def animation_particle_displacement(omega, center, radius, amplitude, grid):
    velocity = sfs.mono.source.pulsating_sphere_velocity(omega,
                                                         center,
                                                         radius,
                                                         amplitude,
                                                         grid)
    displacement = sfs.util.displacement(velocity, omega)

    fig, ax, patch, text = init_2d_plot()
    scat = sfs.plot.particles(grid + displacement,
                              s=15,
                              c='gray',
                              marker='.')
    return animation.FuncAnimation(fig, update_frame_displacement, frames=L,
                                   fargs=(displacement, scat, patch, text))


def animation_particle_velocity(omega, center, radius, amplitude, grid):
    velocity = sfs.mono.source.pulsating_sphere_velocity(omega,
                                                         center,
                                                         radius,
                                                         amplitude,
                                                         grid)

    fig, ax, patch, text = init_2d_plot()
    quiv = sfs.plot.vectors(velocity, grid,
                            clim=[-omega * amplitude, omega * amplitude])
    return animation.FuncAnimation(fig, update_frame_velocity, frames=L,
                                   fargs=(velocity, quiv, patch, text))


def animation_sound_pressure(omega, center, radius, amplitude, grid):
    pressure = sfs.mono.source.pulsating_sphere(omega,
                                                center,
                                                radius,
                                                amplitude,
                                                grid)
    impedance_pw = sfs.defs.rho0 * sfs.defs.c
    fig, ax, patch, text = init_2d_plot()
    im = sfs.plot.soundfield(np.real(pressure),
                             grid,
                             vmin=-impedance_pw * omega * amplitude,
                             vmax=impedance_pw * omega * amplitude,
                             colorbar=False)
    return animation.FuncAnimation(fig, update_frame_pressure, frames=L,
                                   fargs=(pressure, im, patch, text))


if __name__ == '__main__':

    # Pulsating sphere
    center = [0, 0, 0]
    radius = 0.25
    amplitude = 0.05  # amplitude of the surface displacement
    f = 750  # frequency
    omega = 2 * np.pi * f  # angular frequency
    ka = sfs.util.wavenumber(omega) * radius

    # Temporal sampling for animation
    fs = f * 15  # sampling frequency
    L = int(np.round(fs / f))  # number of frames corresponding to one period
    t = np.arange(L) / fs  # time

    # Axis limits
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1

    # Particle displacement
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.025)
    ani = animation_particle_displacement(omega, center, radius, amplitude,
                                          grid)
    ani.save('pulsating_sphere_displacement.gif',
             fps=10,
             dpi=80,
             writer='imagemagick')

    # Particle velocity
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.04)
    ani = animation_particle_velocity(omega, center, radius, amplitude, grid)
    ani.save('pulsating_sphere_velocity.gif',
             fps=10,
             dpi=80,
             writer='imagemagick')

    # Sound pressure
    grid = sfs.util.xyz_grid([xmin, xmax], [ymin, ymax], 0, spacing=0.005)
    ani = animation_sound_pressure(omega, center, radius, amplitude, grid)
    ani.save('pulsating_sphere_pressure.gif',
             fps=10,
             dpi=80,
             writer='imagemagick')
