# pooling-windows

PyTorch implementation of pooling windows like those used in Freeman and
Simoncelli, 2011[^1].

[Matlab code](https://github.com/freeman-lab/metamers/) for these windows exist
as part of Jeremy Freeman's repo for the original paper. This repo is not a
direct port of that code, but a conceptual reimplementation, following the math
outlined in the supplemental materials, and also includes a version using
Gaussian windows. Note this means we do not test whether our outputs match the
original implementation (and, as pointed out by Wallis et al, 2019[^2], there
is a minor bug in that code).

The included Guassian windows overlap more than the original windows (which used
a raised-cosine falloff), and thus give a smoother representation. They were
created by William Broderick for his foveated metamer project ([VSS 2020
poster](https://osf.io/aketq/), [VSS 2023 poster](https://osf.io/8hdaz/),
[preprint](https://www.biorxiv.org/content/10.1101/2023.05.18.541306)), and
notably improved the quality of V1 and retinal metamers. A more detailed
discussion of the differences between the Gaussian and raised-cosine windows can
be found at the top of
[pooling.py](https://github.com/LabForComputationalVision/pooling-windows/blob/main/pooling/pooling.py#L3).

## Requirements

This code works with python 3.7, 3.8, 3.9, and 3.10. The packages required to
use this code can be found in `requirements.txt`. In order to install them in
your virtual environment of choice, run `pip install -r requirements.txt`; you
will then be able to use the code here from within this directory (`pooling` is
not itself installed, and thus will not be on your path). There is one function,
`PoolingWindows.plot_windows()`, which requires
[plenoptic](https://github.com/LabForComputationalVision/plenoptic/) -- if you
wish to use it, you must install that package as well (follow the instructions
in its README).

If you wish to view the included notebook (which contains a simple demonstration
of some sampling and aliasing issues), you will also need to install
[jupyter](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).

## Usage

The `PoolingWindows` class is the main way of interacting with this code. To
use, you instantiate it and then call it on a 4d image-like tensor. Let's break
that down:

- First, load in an image. This can be either grayscale or color. In the
  following code snippet, we treat it as if it were 8-bit integers (which is
  generally the case) and divide it by 255 because we need all values to lie
  between 0 and 1. 

``` python
import torch
import matplotlib.pyplot as plt
import numpy as np
import pooling

img = torch.from_numpy(plt.imread('path/to/image.png').astype(np.float32)) / 255
```

- Unsqueeze the image until it's 4d. Like many pytorch modules, `PoolingWindows`
  operate on 4d images as input: batch by channel by height by width. The batch
  dimension is used for multiple images and channel for RGB(A) (or different
  channels in convolution layers). `PoolingWindows` supports both single- and
  multi-batch and channel inputs, operating independently along batch and
  channel dimensions.
  
``` python
# if this is a grayscale image, it will be unsqueezed twice; if it's RGB, it will be unsqueezed once.
while img.ndim < 4:
    img = img.unsqueeze(0)
```

- Construct the `PoolingWindows` object. There are several possible
  initialization arguments, and you are encouraged to investigate them yourself.
  The only necessary ones are `scaling`, which sets the relationship between the
  width of the windows and eccentricity, and `img_res`, which gives the height
  and width of the input. In the example below, we construct Gaussian windows.
  To instead construct raised-cosine ones, set `window_type='cosine'` and remove
  the `std_dev` argument.

``` python
pw = pooling.PoolingWindows(.5, img.shape[-2:], window_type='gaussian', std_dev=1)
```

- Call `PoolingWindows.forward()` on the image! Note that `pw(img)` and
  `pw.forward(img)` are the same.

``` python
pooled = pw(img)
```

You can also run `PoolingWindows` on dictionaries of 4d tensors, like that
created by the steerable pyramid implementation found in
[plenoptic](https://github.com/LabForComputationalVision/plenoptic) (in this
case, the keys of the input dictionary must be tuples where the first value is
an int, giving the scale):

``` python
import torch
import matplotlib.pyplot as plt
import numpy as np
import pooling
import plenoptic as po

img = torch.from_numpy(plt.imread('path/to/image.png').astype(np.float32)) / 255
while img.ndim < 4:
    img = img.unsqueeze(0)
# create the pyramid
pyr = po.simul.Steerable_Pyramid_Freq(img.shape[-2:], height=4)
# get the pyramid coefficients
pyr_coeffs = pyr(img)
# remove the residuals, which PoolingWindows doesn't support natively
for k in ['residual_highpass', 'residual_lowpass']:
    pyr_coeffs.pop(k)
# let's see their shape
for k, v in pyr_coeffs.items():
    print(f'scale {k[0]}, orientation band {k[1]}: {v.shape}')
# create the windows. Note that we're now setting the number of scales! 
# This must be the same as the height of the pyramid
pw = pooling.PoolingWindows(.5, img.shape[-2:], window_type='gaussian', 
                            std_dev=1, num_scales=4)
# pooled_coeffs will have the same keys as pyr_coeffs, and its values will 
# be the pooled versions of the corresponding value in pyr_coeffs
pooled_coeffs = pw(pyr_coeffs)
for k, v in pooled_coeffs.items():
    print(f'scale {k[0]}, orientation band {k[1]}: {v.shape}')
```

For an example of a more elaborate usage of these windows, see the
[PooledVentralStream](https://github.com/LabForComputationalVision/plenoptic/blob/cdbc56886c3cf57822ae8fd8b71b78ef80670210/plenoptic/simulate/models/ventral_stream.py#L18)
models (note that these are no longer part of `plenoptic` and so will not be
found on the `master` branch).

## Notes

1. These windows are exact, and so are incredibly memory-intensive for small
   scaling values and large resolutions -- creating windows with `scaling=.01`
   and `img_res=(2048, 2600)`, for example, can take 15 minutes to an hour
   (depending on your machine) and more than 100 GB of RAM, and a single
   `forward()` call can take 4 minutes. This is something to be aware of. It is
   possible to come up with more memory-efficient approximations of foveation,
   but it is likely that synthesis will exploit the approximation errors.
4. Because creating the windows can take so long, we have a `cache_dir`
   argument: set this to a directory on your machine and the windows will be
   saved there during creation and, if the appropriate windows are found when
   instantiating a new instance, they will loaded in. If the directory does not
   exist, we raise a `FileNotFoundError`.
1. `PoolingWindows` supports multi-scale windows (by setting `num_scales`
   argument). To do this, we independently construct the windows at each scale
   (note this is different from [[1]](1#), which constructed the windows at the
   input resolution and then blurred them).
2. `PoolingWindows` works on the GPU. If you have a CUDA-compliant GPU available
   and have properly installed pytorch to make use of it, call the
   `.to(torch.device('cuda'))` method of an instantiated `PoolingWindows` object
   to send it to the GPU (note that you will also need to send over your input
   images to the same device).
2. We construct windows smaller than a pixel -- this is a bit of an
   implementation detail and changed back and forth over development. Doing so
   gives a more gradual transition during metamer synthesis from the "pixel
   match" region at the center to the pooled region. Setting the
   `min_eccentricity` will set the eccentricity (in degrees) where we start
   creating the windows, regardless of their size (in pixels) at that point.
3. These windows have only been used on the output of plenoptic's steerable
   pyramid (as described above) and images. Any other use-case will probably
   require some modification.
4. `PoolingWindows` has a variety of helper methods to help understand what it's
   doing, including several that create plots. For example `pw.plot_windows()`
   will create contour plots showing the windows at each eccentricity and a
   small number of angles, and `pw.plot_window_areas()` will show the area of
   the windows (in pixels or degrees) in each eccentricity ring. They should all
   have complete docstrings, so you are encouraged to explore.
5. The code in this repo was originally part of
   [plenoptic](https://github.com/LabForComputationalVision/plenoptic/) but
   removed in March 2021. In moving over the code, I broke the git history; if,
   for some reason, you wish to see the history or git blame, [this plenoptic
   commit](https://github.com/LabForComputationalVision/plenoptic/tree/fb1c4d29c645c9a054baa021c7ffd07609b181d4)
   contains all the code before transferring it to this repo.

## Code structure

  - `tests/test_pooling.py`: some tests (run on every push using Github actions)
    to make sure pooling windows don't change drastically. They're not a
    complete suite of tests for these windows.
  - `pooling/`: python module containing the code for the pooling windows.
    - `pooling_windows.py`: contains the `PoolingWindows` class, which is how
      users should interact with this code.
    - `pooling.py`: variety of calculations used for constructing and
      investigating the windows, used by `PoolingWindows`.
    - `utils.py`: miscellaneous utility functions.
    - `__init__.py`: boilerplate file for making `pooling` a module.
    - `sampling.py`: checks sampling and aliasing issues, used by the
      `Sampling_and_aliasing.ipynb` notebook, but not by `PoolingWindows`.
  - `Sampling_and_aliasing.ipynb`: example of how to check for sampling /
    aliasing issues. I would recommend you do something similar to this if you
    construct your own windows or modify these in a significant way.

## References

[^1]: Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream.
Nature Neuroscience, 14(9), 1195–1201. http://dx.doi.org/10.1038/nn.2889.
[reprint](https://www.cns.nyu.edu/pub/eero/freeman10-reprint.pdf)

[^2]: Wallis, T. S., Funke, C. M., Ecker, A. S., Gatys, L. A., Wichmann, F. A., &
Bethge, M. (2019). Image content is more important than bouma's law for scene
metamers. eLife, 8(), . http://dx.doi.org/10.7554/elife.42512

