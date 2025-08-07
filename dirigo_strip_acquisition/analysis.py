from pathlib import Path
import json

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numba import njit, prange
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter 

from dirigo.components.io import SystemConfig, config_path
from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.logger import Logger
from dirigo.sw_interfaces.acquisition import AcquisitionProduct, Loader
from dirigo.sw_interfaces.processor import ProcessorProduct
from dirigo.hw_interfaces.digitizer import DigitizerProfile
from dirigo.plugins.acquisitions import LineAcquisitionRuntimeInfo
from dirigo.plugins.loggers import TiffLogger
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo.plugins.loaders import deserialize_float64_list
from dirigo_strip_acquisition.acquisitions import (
    RectangularFieldStagePositionHelper, RasterScanStitchedAcquisitionSpec
)
from dirigo_strip_acquisition.processors import StripProcessor, StripStitcher, TileBuilder
from dirigo_strip_acquisition.loggers import PyramidLogger




class StripAcquisitionLoader(Loader): 
    def __init__(self, file_path: str | Path):
        super().__init__(file_path, thread_name="Strip acquisition data loader")

        with tifffile.TiffFile(self._file_path) as tif:   

            self._init_product_pool(
                n=4, 
                shape=tif.pages[0].shape, 
                dtype=tif.pages[0].dtype
            )

            tags = tif.pages[0].tags

            cfg_dict = json.loads(tags[TiffLogger.SYSTEM_CONFIG_TAG].value)
            self.system_config = SystemConfig(cfg_dict)

            runtime_dict = json.loads(tags[TiffLogger.RUNTIME_INFO_TAG].value)
            self.runtime_info = LineAcquisitionRuntimeInfo.from_dict(runtime_dict)

            spec_dict = json.loads(tags[TiffLogger.ACQUISITION_SPEC_TAG].value)
            self.spec = RasterScanStitchedAcquisitionSpec(**spec_dict)

            digi_dict = json.loads(tags[TiffLogger.DIGITIZER_PROFILE_TAG].value)
            self.digitizer_profile = DigitizerProfile.from_dict(digi_dict)

        self.positioner = RectangularFieldStagePositionHelper(
            scan_axis   = self.system_config.fast_raster_scanner['axis'],
            axis_error  = self.runtime_info.stage_scanner_angle, # type: ignore
            line_width  = self.spec.line_width,
            spec        = self.spec
        )
        
        if self.system_config.fast_raster_scanner['axis'] == 'x':
            n_pixels_scan = round(self.spec.x_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.y_range.range / self.spec.pixel_size)
        else:
            n_pixels_scan = round(self.spec.y_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.x_range.range / self.spec.pixel_size)
        n_channels = sum([c.enabled for c in self.digitizer_profile.channels])

        self.final_shape = (
            self.spec.z_steps,
            n_pixels_scan,
            n_pixels_web,
            n_channels
        )

    def run(self):
        try:
            with tifffile.TiffFile(self._file_path) as tif:   
                frames_read = 0
                n_frames = len(tif.pages)
                
                self._timestamps = deserialize_float64_list(
                    tif.pages[0].tags[TiffLogger.TIMESTAMPS_TAG].value
                )
                self._positions = deserialize_float64_list(
                    tif.pages[0].tags[TiffLogger.POSITIONS_TAG].value
                )

                while frames_read < n_frames:
                    frame = self.get_free_product()

                    # Copy raw data
                    frame.data[...] = tif.pages[frames_read].asarray()

                    # Copy metadata
                    frame.timestamps = self._timestamps[frames_read]
                    frame.positions = self._positions[frames_read]

                    print(f"publishing frame {frames_read}")
                    self._publish(frame)

                    frames_read += 1
        finally:
            self._publish(None) # sentinel coding finished
        


    def get_free_product(self) -> AcquisitionProduct:
        return self._product_pool.get()
    

@njit(fastmath=True, cache=True)
def _sum_high_signal_lines(frame, thresh):
    ny, nx, nc = frame.shape
    out = np.zeros((nx, nc), dtype=np.int64)
    for i in range(ny):
        if np.all(frame[i] > thresh):
            out += frame[i] 
    return out


class SignalGradientLogger(Logger):
    def __init__(self, upstream):
        super().__init__(upstream)
        self._strip_sum = None
        self.filepath = config_path() / "optics" / ("line_gradient.csv")
        self.show_results = False

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        try:
            while True:
                with self._receive_product() as frame:

                    if self._strip_sum is not None:
                        self._strip_sum += _sum_high_signal_lines(frame.data, 100)
                    else:
                        self._strip_sum = _sum_high_signal_lines(frame.data, 100)

        except:
            self.save_data(self._strip_sum)

        finally:
            self._publish(None)

    def save_data(self, strip_sum: np.ndarray):
        # average traces
        spec = self._acquisition.spec
        w = spec.line_width
        n_x = round(w / spec.pixel_size)
        x = np.linspace(-w/2, w/2, n_x)
        strip_sum = strip_sum.astype(np.float64)

        # smooth traces
        smoothed = savgol_filter(
            x               = strip_sum, 
            window_length   = 101, 
            polyorder       = 3,
            axis            = 0
        )

        # Plot results (optional)
        if self.show_results:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            y_label = "Intensity (au)"
            x_label = "Position (um)"

            ax1.scatter(x*1e6, strip_sum[:,0], color="tab:blue", label=y_label)
            ax1.plot(x*1e6, smoothed[:,0], color="tab:blue", label=y_label)
            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label, color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.scatter(x*1e6, strip_sum[:,1], color="tab:red", label=y_label)
            ax2.plot(x*1e6, smoothed[:,1], color="tab:red", label=y_label)
            ax2.set_ylabel(y_label, color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            
            ax1.grid(True, which="both", ls="--", alpha=0.4)
            fig.tight_layout()
            plt.show()

        # Normalize & record traces
        smoothed = smoothed / np.max(smoothed, axis=0, keepdims=True)
        np.savetxt(
            fname       = self.filepath, 
            X           = np.concat([x[:,None],smoothed], axis=1), 
            delimiter   = ',',
            header      = "x (m),channel 1, channel 2"
        )



class LineTimestampLogger(Logger):
    """Log all the line timestamps"""
    def __init__(self, upstream):
        super().__init__(upstream)
        self._timestamps = []

    def _receive_product(self) -> AcquisitionProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        try:
            while True:
                with self._receive_product() as frame:
                    self._timestamps.append(frame.timestamps)
        except EndOfStream:
            self._publish(None)

    def save_data(self):
        timestamps = np.array(self._timestamps).ravel()
        dt = np.diff(timestamps)
        dt_filtered = uniform_filter1d(dt, size=256)
        plt.plot(1/dt_filtered)
        plt.xlabel("Trigger number")
        plt.ylabel("Frequency (Hz)")
        plt.show()


class PhaseLogger(Logger):
    """Log all the line timestamps"""
    def __init__(self, upstream):
        super().__init__(upstream)
        self._phases = []

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        try:
            while True:
                with self._receive_product() as frame:
                    self._phases.append(frame.phase)
        except EndOfStream:
            self._publish(None)

    def save_data(self):
        phases = np.array(self._phases).ravel()
        plt.plot(phases)
        plt.xlabel("Trigger number")
        plt.ylabel("Phase (rad)")
        plt.show()


class PositionLogger(Logger):
    """Log all positions"""
    def __init__(self, upstream):
        super().__init__(upstream)
        self._positions = []

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        try:
            while True:
                with self._receive_product() as frame:
                    self._positions.append(frame.positions)
        except EndOfStream:
            self._publish(None)

    def save_data(self):
        all_positions = np.array(self._positions)
        x, y = all_positions[...,0].ravel(), all_positions[...,1].ravel()
        x -= x[0]
        y -= y[0]
        t = np.arange(len(x))        # use the sample‐index as the colour

        plt.scatter(1000*x, 1000*y, c=t, cmap='plasma', s=5)   # s=dot‐size
        plt.xlabel("X location (mm)")
        plt.ylabel("Y location (mm)")
        plt.colorbar(label='sample index')
        plt.show()


if __name__ == "__main__":
    # Use to reprocess raw saved datasets
    fn = r"D:\dirigo test data\gyn3_scan_raw_0.tif"

    loader = StripAcquisitionLoader(fn)
    timestamper = LineTimestampLogger(upstream=loader)
    #positioner = PositionLogger(upstream=loader)
    processor = RasterFrameProcessor(upstream=loader)
    # phaser = PhaseLogger(upstream=processor)
    strip_processor = StripProcessor(upstream=processor)
    strip_stitcher = StripStitcher(upstream=strip_processor)
    # strip_logger = TiffLogger(upstream=strip_stitcher)
    # strip_logger.frames_per_file = 100
    tile_builder = TileBuilder(upstream=strip_stitcher)
    logger = PyramidLogger(upstream=tile_builder)

    loader.add_subscriber(timestamper)
    #loader.add_subscriber(positioner)
    loader.add_subscriber(processor)
    # processor.add_subscriber(phaser)
    processor.add_subscriber(strip_processor)
    #strip_processor.add_subscriber(strip_logger)
    strip_processor.add_subscriber(strip_stitcher)
    strip_stitcher.add_subscriber(tile_builder)
    # strip_stitcher.add_subscriber(strip_logger)
    tile_builder.add_subscriber(logger)

    timestamper.start()
    #positioner.start()
    processor.start()
    # phaser.start()
    strip_processor.start()
    # strip_logger.start()
    strip_stitcher.start()
    tile_builder.start()
    logger.start()

    loader.start()

    logger.join(30)

    # timestamper.save_data()