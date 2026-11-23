# -*- coding: utf-8 -*-
"""
Image test(s) for obspy.signal.spectral_exstimation.PPSD.
"""
import matplotlib.pyplot as plt


class TestPPSD:
    """
    Test cases for PPSD plotting.
    """
    def test_ppsd_plot(self, ignore_numpy_errors, image_path, _ppsd):
        """
        Test plot of ppsd example data, normal (non-cumulative) style.
        """
        _ppsd.plot(
            show=False, show_coverage=True, show_histogram=True,
            show_percentiles=True, percentiles=[75, 90],
            show_noise_models=True, grid=True, max_percentage=50,
            period_lim=(0.02, 100), show_mode=True, show_mean=True)
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_ylim(-160, -130)
        plt.draw()
        fig.savefig(image_path)

    def test_ppsd_plot_frequency(self, _ppsd, image_path, ignore_numpy_errors):
        """
        Test plot of ppsd example data, normal (non-cumulative) style.
        """
        _ppsd.plot(
            show=False, show_coverage=False, show_histogram=True,
            show_percentiles=True, percentiles=[20, 40],
            show_noise_models=True, grid=False, max_percentage=50,
            period_lim=(0.2, 50), show_mode=True, show_mean=True,
            xaxis_frequency=True)
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_ylim(-160, -130)
        plt.draw()
        fig.savefig(image_path, dpi=50)

    def test_ppsd_plot_into_existing_axes(self, _ppsd):
        """PPSD.plot can render into caller-owned subplot axes."""
        fig, axes = plt.subplots(2, 1)
        result = _ppsd.plot(fig=fig, axes=axes, show=False,
                            show_coverage=False, show_noise_models=False)
        assert result is fig
        assert axes[0].images or axes[0].collections
        assert not axes[1].images and not axes[1].collections

    def test_ppsd_plot_cumulative(self, ignore_numpy_errors, image_path,
                                  _ppsd):
        """
        Test plot of ppsd example data, cumulative style.
        """
        _ppsd.plot(
            show=False, show_coverage=True, show_histogram=True,
            show_noise_models=True, grid=True, period_lim=(0.02, 100),
            cumulative=True,
            # This does not do anything but silences a warning that
            # the `cumulative` and `max_percentage` arguments cannot
            #  be used at the same time.
            max_percentage=None)
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.set_ylim(-160, -130)
        plt.draw()
        fig.savefig(image_path)
