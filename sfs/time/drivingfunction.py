"""Compute time based driving functions for various systems.

.. include:: math-definitions.rst

"""
from __future__ import division
import numpy as np
from numpy.core.umath_tests import inner1d  # element-wise inner product
from scipy.signal import besselap, sosfilt, zpk2sos
from .. import defs
from .. import util


def wfs_25d_plane(x0, n0, n=[0, 1, 0], xref=[0, 0, 0], c=None):
    r"""Plane wave model by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    n : (3,) array_like, optional
        Normal vector (propagation direction) of synthesized plane wave.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.

    Notes
    -----
    2.5D correction factor

    .. math::

        g_0 = \sqrt{2 \pi |x_\mathrm{ref} - x_0|}

    d using a plane wave as source model

    .. math::

        d_{2.5D}(x_0,t) = h(t)
        2 g_0 \scalarprod{n}{n_0}
        \dirac{t - \frac{1}{c} \scalarprod{n}{x_0}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.pw.2.5D

    """
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    n = util.normalize_vector(n)
    xref = util.asarray_1d(xref)
    g0 = np.sqrt(2 * np.pi * np.linalg.norm(xref - x0, axis=1))
    delays = inner1d(n, x0) / c
    weights = 2 * g0 * inner1d(n, n0)
    return delays, weights


def wfs_25d_point(x0, n0, xs, xref=[0, 0, 0], c=None):
    r"""Point source by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    xs : (3,) array_like
        Virtual source position.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.

    Notes
    -----
    2.5D correction factor

    .. math::

         g_0 = \sqrt{2 \pi |x_\mathrm{ref} - x_0|}


    d using a point source as source model

    .. math::

         d_{2.5D}(x_0,t) = h(t)
         \frac{g_0  \scalarprod{(x_0 - x_s)}{n_0}}
         {2\pi |x_0 - x_s|^{3/2}}
         \dirac{t - \frac{|x_0 - x_s|}{c}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.ps.2.5D

    """
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    g0 = np.sqrt(2 * np.pi * np.linalg.norm(xref - x0, axis=1))
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    delays = r/c
    weights = g0 * inner1d(ds, n0) / (2 * np.pi * r**(3/2))
    return delays, weights


def wfs_25d_focused(x0, n0, xs, xref=[0, 0, 0], c=None):
    r"""Point source by 2.5-dimensional WFS.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    n0 : (N, 3) array_like
        Sequence of secondary source orientations.
    xs : (3,) array_like
        Virtual source position.
    xref : (3,) array_like, optional
        Reference position
    c : float, optional
        Speed of sound

    Returns
    -------
    delays : (N,) numpy.ndarray
        Delays of secondary sources in seconds.
    weights: (N,) numpy.ndarray
        Weights of secondary sources.

    Notes
    -----
    2.5D correction factor

    .. math::

         g_0 = \sqrt{\frac{|x_\mathrm{ref} - x_0|}
         {|x_0-x_s| + |x_\mathrm{ref}-x_0|}}


    d using a point source as source model

    .. math::

         d_{2.5D}(x_0,t) = h(t)
         \frac{g_0  \scalarprod{(x_0 - x_s)}{n_0}}
         {|x_0 - x_s|^{3/2}}
         \dirac{t + \frac{|x_0 - x_s|}{c}}

    with wfs(2.5D) prefilter h(t), which is not implemented yet.

    References
    ----------
    See http://sfstoolbox.org/en/latest/#equation-d.wfs.fs.2.5D

    """
    if c is None:
        c = defs.c
    x0 = util.asarray_of_rows(x0)
    n0 = util.asarray_of_rows(n0)
    xs = util.asarray_1d(xs)
    xref = util.asarray_1d(xref)
    ds = x0 - xs
    r = np.linalg.norm(ds, axis=1)
    g0 = np.sqrt(np.linalg.norm(xref - x0, axis=1)
                 / (np.linalg.norm(xref - x0, axis=1) + r))
    delays = -r/c
    weights = g0 * inner1d(ds, n0) / (2 * np.pi * r**(3/2))
    return delays, weights


def driving_signals(delays, weights, signal):
    """Get driving signals per secondary source.

    Returned signals are the delayed and weighted mono input signal
    (with N samples) per channel (C).

    Parameters
    ----------
    delays : (C,) array_like
        Delay in seconds for each channel, negative values allowed.
    weights : (C,) array_like
        Amplitude weighting factor for each channel.
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the driving signals (in a `numpy.ndarray`
        with shape ``(N, C)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    delays = util.asarray_1d(delays)
    weights = util.asarray_1d(weights)
    data, samplerate, signal_offset = apply_delays(signal, delays)
    return util.DelayedSignal(data * weights, samplerate, signal_offset)


def apply_delays(signal, delays):
    """Apply delays for every channel.

    Parameters
    ----------
    signal : (N,) array_like + float
        Excitation signal consisting of (mono) audio data and a sampling
        rate (in Hertz).  A `DelayedSignal` object can also be used.
    delays : (C,) array_like
        Delay in seconds for each channel (C), negative values allowed.

    Returns
    -------
    `DelayedSignal`
        A tuple containing the delayed signals (in a `numpy.ndarray`
        with shape ``(N, C)``), followed by the sampling rate (in Hertz)
        and a (possibly negative) time offset (in seconds).

    """
    data, samplerate, initial_offset = util.as_delayed_signal(signal)
    data = util.asarray_1d(data)
    delays = util.asarray_1d(delays)
    delays += initial_offset

    delays_samples = np.rint(samplerate * delays).astype(int)
    offset_samples = delays_samples.min()
    delays_samples -= offset_samples
    out = np.zeros((delays_samples.max() + len(data), len(delays_samples)))
    for column, row in enumerate(delays_samples):
        out[row:row + len(data), column] = data
    return util.DelayedSignal(out, samplerate, offset_samples / samplerate)


def nfchoa_25d_plane(x0, r0, npw, fs, max_order=None, c=None):
    """Virtual plane wave by 2.5-dimensional NFC-HOA.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the circular secondary source distribution
    npw : (3,) array_like
        Unit vector (propagation direction) of plane wave.
    fs : int
        Sampling frequency in Hz
    max_order : int, optional
        Ambisonics order
    c : float, optional
        Speed of sound

    Returns
    -------
    delay : float
        Overall delay in seconds (common to all secondary sources)
    weight : float
        Overall weight (common to all secondary sources)
    sos : list of array_like
        Second-order section filters
    phaseshift : float
        Phase shift in radians

    References
    ----------
        S. Spors, V. Kuscher, J. Ahrens (2011) - "Efficient realization of
       model-based rendering for 2.5-dimensional near-field compensated higher
       order Ambisonics", WASPAA, p. 61-64

       See Eq.(10)

    """
    max_order = util.max_order_circular_harmonics(len(x0), max_order)
    if c is None:
        c = defs.c

    x0 = util.asarray_of_rows(x0)
    npw = util.asarray_1d(npw)
    phipw, _, _ = util.cart2sph(*npw)
    phi0, _, _ = util.cart2sph(*x0.T)

    delay = -r0/c
    weight = 2
    sos = []
    for m in range(max_order+1):
        _, p, _ = besselap(m, norm='delay')
        # TODO: modify "besselap" for very high orders (>150)
        s0 = np.zeros(m)
        sinf = (c/r0)*p
        z0 = np.ones(m)
        zinf = np.exp(sinf/fs)
        # TODO: select matched-z or bilinear transform
        k = _normalize_digital_filter_gain(s0, sinf, z0, zinf, fs)
        sos.append(zpk2sos(z0, zinf, k, pairing='nearest'))
        # TODO: normalize the SOS filters individually?
    phaseshift = phipw + np.pi - phi0
    return delay, weight, sos, phaseshift


def nfchoa_25d_point(x0, r0, xs, fs, max_order=None, c=None):
    """Virtual Point source by 2.5-dimensional NFC-HOA.

    Parameters
    ----------
    x0 : (N, 3) array_like
        Sequence of secondary source positions.
    r0 : float
        Radius of the circular secondary source distribution
    xs : (3,) array_like
        Virtual source position.
    fs : int
        Sampling frequency in Hz
    max_order : int, optional
        Ambisonics order
    c : float, optional
        Speed of sound

    Returns
    -------
    delay : float
        Overall delay in seconds (common to all secondary sources)
    weight : float
        Overall weight (common to all secondary sources)
    sos : list of array_like
        Second-order section filters
    phaseshift : float
        Phase shift in radians

    References
    ----------
        S. Spors, V. Kuscher, J. Ahrens (2011) - "Efficient realization of
       model-based rendering for 2.5-dimensional near-field compensated higher
       order Ambisonics", WASPAA, p. 61-64

       See Eq.(11)

    """
    max_order = util.max_order_circular_harmonics(len(x0), max_order)
    if c is None:
        c = defs.c

    x0 = util.asarray_of_rows(x0)
    xs = util.asarray_1d(xs)
    phi0, _, _ = util.cart2sph(*x0.T)
    phi, _, r = util.cart2sph(*xs)

    delay = (r-r0)/c
    weight = 1/2/np.pi/r
    sos = []
    for m in range(max_order+1):
        _, p, k = besselap(m, norm='delay')
        # TODO: modify "besselap" for very high orders (>150)
        s0 = (c/r)*p
        sinf = (c/r0)*p
        z0 = np.exp(s0/fs)
        zinf = np.exp(sinf/fs)
        # TODO: select matched-z or bilinear transform
        k = _normalize_digital_filter_gain(s0, sinf, z0, zinf, fs)
        sos.append(zpk2sos(z0, zinf, k, pairing='nearest'))
        # TODO: normalize the SOS filters individually?
    phaseshift = phi0 - phi
    return delay, weight, sos, phaseshift


def nfchoa_driving_signals(delay, weight, sos, phaseshift, signal, max_order=None):
    """Get NFC-HOA driving signals per secondary source.

    Parameters
    ----------
    delay : float
        Overall delay in seconds (common to all secondary source)
    weight : float
        Overall weight (common to all secondary sources)
    sos : list of array_like
        Second-order section filters
    phaseshift : (N,) array_like
        Phase shift in radians
    signal : tuple of (L,) array_like, followed by 1 or 2 scalars
        Excitation signal consisting of (mono) audio data, sampling rate
        (in Hertz) and optional starting time (in seconds).
    fs: int
        Sampling frequency in Hz
    max_order : int, optional

    Returns
    -------
    driving_signals : (L, N) numpy.ndarray
        Driving signals.
    t_offset : float
        Simulation point in time offset (seconds).

    """
    max_order = util.max_order_circular_harmonics(len(phaseshift), max_order)
    data, fs, t_offset = util.as_delayed_signal(signal)
    N = len(phaseshift)
    L = len(data)
    d = np.zeros((L, N))
#    TODO : check FOS/SOS structure

    d = np.tile(np.expand_dims(sosfilt(sos[0], data), 1), (1, N))
    for m in range(1, max_order+1):
        modal_response = sosfilt(sos[np.abs(m)], data)
        for n in range(N):
            d[:, n] += modal_response * 2 * np.cos(m*phaseshift[n])
    t_offset += delay
    return np.real(d) * weight, fs, t_offset


def _normalize_digital_filter_gain(s0, sinf, z0, zinf, fs):
    """Match the digital filter gain at the Nyquist frequency.

    Parameters
    ----------
    s0 : (N,) array_like
        zeros in the Laplace domain
    sinf : (N,) array_like
        polse in the Laplace domain
    z0 : (N,) array_like
        zeros in the z-domain
    zinf : (N,) array_like
        zeros in the z-domain
    fs : int
        Sampling frequency in Hz

    """
    # TODO: check the number of poles and zeros
    k = 1
    if np.shape(sinf) is 0:
        k = 1
    else:
        omega = 1j*np.pi*fs
        k *= np.prod((omega-s0)/(omega-sinf))
        k *= np.prod((-1-zinf)/(-1-z0))
    return np.abs(k)
