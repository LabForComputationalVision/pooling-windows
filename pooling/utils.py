#!/usr/bin/env python3
import numpy as np
import warnings
import torch
try:
    from IPython.display import HTML
except ImportError:
    warnings.warn("Unable to import IPython.display.HTML")


def to_numpy(x):
    r"""Cast tensor to numpy in the most conservative way possible.

    Parameters
    ----------
    x: `torch.Tensor` or `np.ndarray`
       Tensor to be converted to `np.ndarray` on CPU. If it's already an array,
       we do nothing. We also cast it to float32.

    Returns
    -------
    x : np.ndarray
       array version of `x`

    """
    try:
        x = x.detach().cpu().numpy().astype(np.float32)
    except AttributeError:
        # in this case, it's already a numpy array
        pass
    return x


def polar_radius(size, exponent=1, origin=None, device=None):
    """Make distance-from-origin (r) matrix.

    Compute a matrix of given size containing samples of a radial ramp
    function, raised to given exponent, centered at given origin.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size,
        size)`. if a tuple, must be a 2-tuple of ints specifying the
        dimensions
    exponent : `float`
        the exponent of the radial ramp function.
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at
        `(origin, origin)`. if a tuple, must be a 2-tuple of ints
        specifying the origin (where `(0, 0)` is the upper left).  if
        None, we assume the origin lies at the center of the matrix,
        `(size+1)/2`.
    device : str or torch.device
        the device to create this tensor on

    Returns
    -------
    res : torch.Tensor
        the polar radius matrix

    """
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(torch.arange(1, size[0]+1, device=device)-origin[0],
                                  torch.arange(1, size[1]+1, device=device)-origin[1])

    if exponent <= 0:
        # zero to a negative exponent raises:
        # ZeroDivisionError: 0.0 cannot be raised to a negative power
        r = xramp ** 2 + yramp ** 2
        res = np.power(r, exponent / 2.0, where=(r != 0))
    else:
        res = (xramp ** 2 + yramp ** 2) ** (exponent / 2.0)
    return res


def polar_angle(size, phase=0, origin=None, device=None):
    """Make polar angle matrix (in radians).

    Compute a matrix of given size containing samples of the polar angle (in radians, CW from the
    X-axis, ranging from -pi to pi), relative to given phase, about the given origin pixel.

    Arguments
    ---------
    size : `int` or `tuple`
        if an int, we assume the image should be of dimensions `(size, size)`. if a tuple, must be
        a 2-tuple of ints specifying the dimensions
    phase : `float`
        the phase of the polar angle function (in radians, clockwise from the X-axis)
    origin : `int`, `tuple`, or None
        the center of the image. if an int, we assume the origin is at `(origin, origin)`. if a
        tuple, must be a 2-tuple of ints specifying the origin (where `(0, 0)` is the upper left).
        if None, we assume the origin lies at the center of the matrix, `(size+1)/2`.
    device : str or torch.device
        the device to create this tensor on

    Returns
    -------
    res : torch.Tensor
        the polar angle matrix

    """
    if not hasattr(size, '__iter__'):
        size = (size, size)

    if origin is None:
        origin = ((size[0]+1)/2., (size[1]+1)/2.)
    elif not hasattr(origin, '__iter__'):
        origin = (origin, origin)

    # for some reason, torch.meshgrid returns them in the opposite order
    # that np.meshgrid does. So, in order to get the same output, we
    # grab them as (yramp, xramp) instead of (xramp, yramp). similarly,
    # we have to reverse the order from (size[1], size[0]) to (size[0],
    # size[1])
    yramp, xramp = torch.meshgrid(torch.arange(1, size[0]+1, device=device)-origin[0],
                                  torch.arange(1, size[1]+1, device=device)-origin[1])

    res = torch.atan2(yramp, xramp)

    res = ((res+(np.pi-phase)) % (2*np.pi)) - np.pi

    return res


def convert_anim_to_html(anim):
    r"""convert a matplotlib animation object to HTML (for display)

    This is a simple little wrapper function that allows the animation
    to be displayed in a Jupyter notebook

    Parameters
    ----------
    anim : `matplotlib.animation.FuncAnimation`
        The animation object to convert to HTML
    """

    # to_html5_video will call savefig with a dpi kwarg, so our
    # custom figure class will raise a warning. we don't want to
    # worry people, so we go ahead and suppress it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return HTML(anim.to_html5_video())
