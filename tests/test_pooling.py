#!/usr/bin/env python3
import pytest
import sys
import numpy as np
import torch
import os.path as op
import matplotlib.pyplot as plt
import plenoptic as po
sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..'))
import pooling


class TestPooling(object):

    def test_creation(self):
        ang_windows, ecc_windows = pooling.pooling.create_pooling_windows(.87, (256, 256))

    def test_creation_args(self):
        ang, ecc = pooling.pooling.create_pooling_windows(.87, (100, 100), .2, 30, 1.2,
                                                           transition_region_width=.7)
        ang, ecc = pooling.pooling.create_pooling_windows(.87, (100, 100), .2, 30, 1.2,
                                                           transition_region_width=.5)
        ang, ecc = pooling.pooling.create_pooling_windows(.87, (100, 100), .2, 30, 1.2,
                                                           'gaussian', std_dev=1)

    def test_ecc_windows(self):
        windows = pooling.pooling.log_eccentricity_windows((256, 256), n_windows=4)
        windows = pooling.pooling.log_eccentricity_windows((256, 256), n_windows=4.5)
        windows = pooling.pooling.log_eccentricity_windows((256, 256), window_spacing=.5)
        windows = pooling.pooling.log_eccentricity_windows((256, 256), window_spacing=1)

    def test_angle_windows(self):
        windows = pooling.pooling.polar_angle_windows(4, (256, 256))
        windows = pooling.pooling.polar_angle_windows(4, (1000, 1000))
        with pytest.raises(Exception):
            windows = pooling.pooling.polar_angle_windows(1.5, (256, 256))
        with pytest.raises(Exception):
            windows = pooling.pooling.polar_angle_windows(1, (256, 256))

    def test_calculations(self):
        # these really shouldn't change, but just in case...
        assert pooling.pooling.calc_angular_window_spacing(2) == np.pi
        assert pooling.pooling.calc_angular_n_windows(2) == np.pi
        with pytest.raises(Exception):
            pooling.pooling.calc_eccentricity_window_spacing()
        assert pooling.pooling.calc_eccentricity_window_spacing(n_windows=4) == 0.8502993454155389
        assert pooling.pooling.calc_eccentricity_window_spacing(scaling=.87) == 0.8446653390527211
        assert pooling.pooling.calc_eccentricity_window_spacing(5, 10, scaling=.87) == 0.8446653390527211
        assert pooling.pooling.calc_eccentricity_window_spacing(5, 10, n_windows=4) == 0.1732867951399864
        assert pooling.pooling.calc_eccentricity_n_windows(0.8502993454155389) == 4
        assert pooling.pooling.calc_eccentricity_n_windows(0.1732867951399864, 5, 10) == 4
        assert pooling.pooling.calc_scaling(4) == 0.8761474337786708
        assert pooling.pooling.calc_scaling(4, 5, 10) == 0.17350368946058647
        assert np.isinf(pooling.pooling.calc_scaling(4, 0))

    @pytest.mark.parametrize('num_scales', [1, 3])
    @pytest.mark.parametrize('transition_region_width', [.5, 1])
    def test_PoolingWindows_cosine(self, num_scales, transition_region_width):
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        pw = pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                     transition_region_width=transition_region_width,
                                     window_type='cosine',)
        pw(im)

    @pytest.mark.parametrize('num_scales', [1, 3])
    def test_PoolingWindows(self, num_scales):
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        pw = pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                     window_type='gaussian', std_dev=1)
        pw(im)
        # we only support std_dev=1
        with pytest.raises(Exception):
            pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                    window_type='gaussian', std_dev=2)
        with pytest.raises(Exception):
            pooling.PoolingWindows(.5, im.shape[2:], num_scales=num_scales,
                                    window_type='gaussian', std_dev=.5)

    def test_PoolingWindows_project(self):
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        pw = pooling.PoolingWindows(.5, im.shape[2:])
        pooled = pw(im)
        pw.project(pooled)
        pw = pooling.PoolingWindows(.5, im.shape[2:], num_scales=3)
        pooled = pw(im)
        pw.project(pooled)

    def test_PoolingWindows_nonsquare(self):
        # test PoolingWindows with weirdly-shaped iamges
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        for sh in [(256, 128), (256, 127), (256, 125), (125, 125), (127, 125)]:
            tmp = im[..., :sh[0], :sh[1]]
            pw = pooling.PoolingWindows(.9, tmp.shape[-2:])
            pw(tmp)

    def test_PoolingWindows_plotting(self):
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        pw = pooling.PoolingWindows(.8, im.shape[-2:], num_scales=2)
        pw.plot_window_areas()
        pw.plot_window_widths()
        for i in range(2):
            pw.plot_window_areas('pixels', i)
            pw.plot_window_widths('pixels', i)
        fig = po.imshow(im)
        pw.plot_windows(fig.axes[0])
        plt.close('all')

    def test_PoolingWindows_caching(self, tmp_path):
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        # first time we save, second we load
        pw = pooling.PoolingWindows(.8, im.shape[-2:], num_scales=2, cache_dir=tmp_path)
        pw = pooling.PoolingWindows(.8, im.shape[-2:], num_scales=2, cache_dir=tmp_path)

    def test_PoolingWindows_sep(self):
        # test the window and pool function separate of the forward function
        im = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        pw = pooling.PoolingWindows(.5, im.shape[2:])
        pw.pool(pw.window(im))
