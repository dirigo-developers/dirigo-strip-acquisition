from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import math
import time
import threading
from typing import Optional
from dataclasses import dataclass

import numpy as np

from dirigo import units, io
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.plugins.acquisitions import LineAcquisition, LineAcquisitionSpec
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner
from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.hw_interfaces.camera import LineScanCamera
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder, LinearEncoder

from dirigo_e2v_line_scan_camera.dirigo_e2v_line_scan_camera import TriggerModes # TODO eliminate need for this


"""
Comments:
Scan dimension: parallel to the fastest acquired line (fast raster scanner or 
camera pixel array)
Web dimension: orthogonal to scan dimension

Scan/web dimensions may correspond to either global X or Y, however due to
limitations in the way tiled tiffs are saved, in the final stitched image the 
web dimension will be horizontal and scan dimension will be vertical.

Limitations:
Pixel size can only be square: will use pixel size (along scan dimension as
pixel height)
Overlap will be adjusted such that an integer pixel overlap is used.
"""


class StripAcquisitionSpec(LineAcquisitionSpec):
    """Specification for a strip scan acquisition."""
    def __init__(self, 
                 x_range: dict, 
                 y_range: dict, 
                 strip_overlap: float = 0.1, # default 10% overlap 
                 integration_time: Optional[str] = None,
                 integration_duty_cycle: Optional[float] = None,
                 **kwargs): 
        """
        Create a spec object for strip scan. 
        """
        if "buffers_per_acquisition" not in kwargs:
            kwargs["buffers_per_acquisition"] = -1 # -1 codes for unlimited
        
        self.x_range = units.PositionRange(**x_range)
        self.y_range = units.PositionRange(**y_range)
       
        if "lines_per_buffer" not in kwargs:
            # kwargs["lines_per_buffer"] = round(self.x_range.range  # TODO, allow web/scan axis switch
            #                                    / units.Position(kwargs["pixel_size"]))
            kwargs["lines_per_buffer"] = -1
        super().__init__(**kwargs)
        self.pixel_height = self.pixel_size # force square pixel

        if not (0 <= strip_overlap < 1):
            raise ValueError(f"`overlap` must be a float between 0 and 1")
        # round to nearest integer pixel overlap
        self.strip_overlap = round(strip_overlap * self.pixels_per_line) / self.pixels_per_line

        if integration_time:
            self.integration_time = units.Time(integration_time)
            if integration_duty_cycle:
                self.integration_duty_cyle = integration_duty_cycle
            else:
                self.integration_duty_cyle = 0.5

    @property
    def lines_per_frame(self) -> int:
        """Alias for lines_per_buffer"""
        return self.lines_per_buffer 
      
class RectangularFieldStagePositionHelper:
    """Encapsulates stage runtime position calculations."""
    def __init__(self, scan_axis: str, axis_error: units.Angle, spec: StripAcquisitionSpec):
        self._scan_axis = scan_axis
        self._axis_error = axis_error
        self.spec = spec
    
    @cached_property
    def web_limits(self) -> units.PositionRange:
        if self._scan_axis == "x":
            return self.spec.y_range
        else:
            return self.spec.x_range

    def scan_center(self, strip_index: int) -> units.Position:
        """Return the center position (along scan dimension) for `strip_index`"""
        effective_line_width = self.spec.line_width * (1 - self.spec.strip_overlap) # reduced by the overlap
        relative_position = (strip_index + 0.5) * effective_line_width

        if self._scan_axis == "x":
            return units.Position(self.spec.x_range.min + relative_position)
        else:
            return units.Position(self.spec.y_range.min + relative_position)
        
    def web_min(self, strip_index: int) -> units.Position:
        shear = (self.scan_center(strip_index) - self.scan_center(self.n_strips//2)) \
            * float(self._axis_error)
        if self._scan_axis == "x":
            return units.Position(self.spec.y_range.min) - shear
        else:
            return units.Position(self.spec.x_range.min) - shear

    def web_max(self, strip_index: int) -> units.Position:
        shear = (self.scan_center(strip_index) - self.scan_center(self.n_strips//2)) \
            * float(self._axis_error)
        if self._scan_axis == "x":
            return units.Position(self.spec.y_range.max) - shear
        else:
            return units.Position(self.spec.x_range.max) - shear

    @cached_property
    def n_strips(self) -> int:
        effective_line_width = self.spec.line_width * (1 - self.spec.strip_overlap)
        if self._scan_axis == "x":
            scan_axis_range = self.spec.x_range.range
        else:
            scan_axis_range = self.spec.y_range.range
        N = (scan_axis_range - self.spec.line_width) / effective_line_width + 1
        return max(math.ceil(N), 1)
    


class StripBaseAcquisition(Acquisition, ABC):
    """
    Base class for strip scan acquistion worker.

    Uses coordinated line acquisition and stage movement to capture a large 
    field. Define 'scan axis' as the axis parallel to the acquired lines. Define
    'web axis' as perpendicular to scan axis.

    Concrete subclasses need to define required_resources, spec_location, 
    Spec, and provide a method for setup_line_acquisition()
    """
    
    def __init__(self, hw, system_config, spec: StripAcquisitionSpec):
        super().__init__(hw, system_config, spec)
        self.spec: StripAcquisitionSpec

        # set up internal line acquisition        
        self._line_acquisition = self.setup_line_acquisition()

        # define functional axes
        if self._line_acquisition.axis == 'x':
            self._scan_axis_stage = self.hw.stages.x
            self._web_axis_stage = self.hw.stages.y # web axis = perpendicular to the fast raster scanner / line-scan camera axis
        else:
            self._scan_axis_stage = self.hw.stages.y
            self._web_axis_stage = self.hw.stages.x

        self.positioner = RectangularFieldStagePositionHelper(
            scan_axis=self.system_config.fast_raster_scanner['axis'],
            axis_error=self.hw.laser_scanning_optics.stage_scanner_angle,
            spec=spec
        )

        if isinstance(self._line_acquisition.data_acquisition_device, Digitizer):
            n_channels = self._line_acquisition.hw.digitizer.acquire.n_channels_enabled
        else:
            n_channels = 1

        if self._line_acquisition.axis == 'x':
            n_pixels_scan = round(self.spec.x_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.y_range.range / self.spec.pixel_size)
        else:
            n_pixels_scan = round(self.spec.y_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.x_range.range / self.spec.pixel_size)

        self._final_shape = (
            n_pixels_scan,
            n_pixels_web,
            n_channels
        )

    @property
    def final_shape(self) -> tuple[int,int,int]:
        """Shape of the final stitched image with dimensions: (scan, web, chan)"""
        return self._final_shape
        

    @abstractmethod
    def setup_line_acquisition(self) -> "PointScanLineAcquisition | LineScanCameraLineAcquisition":
        pass

    def add_subscriber(self, subscriber):
        """
        Adds a subscriber to subscriber list for internal LineAcquisition worker.

        When workers subscribe to StripAcquisition, they are actually subscribing
        to internal LineAcquisition.
        """
        self._line_acquisition.add_subscriber(subscriber)

    def run(self):
        # move to start (2 axes)
        self._scan_axis_stage.move_to(self.positioner.scan_center(strip_index=0))
        self._web_axis_stage.move_to(self.positioner.web_limits.min)
        self._scan_axis_stage.wait_until_move_finished()
        self._web_axis_stage.wait_until_move_finished()

        # set strip velocity
        self._original_web_velocity = self._web_axis_stage.max_velocity
        self._web_axis_stage.max_velocity = self._web_velocity

        # start line acquisition
        self._line_acquisition.start() 
        while not self._line_acquisition.active.is_set():
            time.sleep(0.001) # wait (active event indicates data is acquiring)

        # If slow raster scanner present, park in
        if self.hw.slow_raster_scanner:
            self.hw.slow_raster_scanner.center()
        
        try:
            for strip_index in range(self.positioner.n_strips):
                if self._stop_event.is_set():
                    # Terminate acquisitions
                    break

                # start web axis movement
                if strip_index % 2:
                    strip_end_position = self.positioner.web_limits.min
                else:
                    strip_end_position = self.positioner.web_limits.max

                self._web_axis_stage.move_to(strip_end_position)

                # wait until web axis decceleration
                time.sleep(self._web_period + units.Time('10 ms')) # the last part is empirical

                # begin lateral movement to the next strip
                if strip_index < (self.positioner.n_strips - 1):
                    self._scan_axis_stage.move_to(
                        self.positioner.scan_center(strip_index=strip_index + 1)
                    )

                # wait for web axis movement to come to complete stop
                self._web_axis_stage.wait_until_move_finished()

        finally:
            # Stop the line acquisition worker
            if self.hw.slow_raster_scanner:
                self.hw.slow_raster_scanner.park()
            self._line_acquisition.stop()

            # Revert the web axis velocity
            self._web_axis_stage.max_velocity = self._original_web_velocity

    @property
    def runtime_info(self):
        return self._line_acquisition.runtime_info
    
    @property
    def digitizer_profile(self):
        return self._line_acquisition.digitizer_profile

    @cached_property
    def _web_velocity(self) -> units.Velocity:
        """The target velocity for web axis movement."""
        return units.Velocity(
            float(self._line_acquisition.line_frequency) * float(self.spec.pixel_height)
        )
    
    @cached_property
    def _web_period(self) -> units.Time:
        """The approximate period of time required to capture 1 strip."""
        return self.positioner.web_limits.range / self._web_velocity
    

"""
PointScanLineAcquisition and PointScanStripAcquisition provides a concrete
implementation for strip scanning areas using a point-scanned focused, usually
with a fast scanner like a resonant or polygon scanner. 

Stage position is read out in sync with the line clock for post-hoc dewarping.
"""
class PointScanLineAcquisition(LineAcquisition):
    """
    Customized LineAcquisition that starts linear position measurement 
    (encoders) and allows measuring them by overriding the read_position method.
    """
    def __init__(self, hw, system_config, spec: LineAcquisitionSpec):
        super().__init__(hw, system_config, spec)

        self._read_positions = 0

    @property
    def axis(self) -> str:
        return self.hw.fast_raster_scanner.axis
    
    @property
    def line_frequency(self) -> units.Frequency:
        if self.spec.bidirectional_scanning:
            return 2 * self.hw.fast_raster_scanner.frequency
        else:
            return self.hw.fast_raster_scanner.frequency

    def run(self):
        """
        Identical to StripAcquisition's run(), except also starts and stops the
        encoders.
        """
        # pass hw reference to allow the method to get initial positions, scanner frequency
        self.hw.encoders.start_logging(self.hw) 
        try:
            super().run()
        finally:
            self.hw.encoders.stop()

    def read_positions(self):
        """Override provides sample positions from linear position encoders."""  # TODO: order dimensions scan, web?
        if self._read_positions == 0:
            # skip first position read
            self.hw.encoders.read_positions(1)
            
        positions = self.hw.encoders.read_positions(self.spec.records_per_buffer) 
        self._read_positions += self.spec.records_per_buffer
        return positions


class PointScanStripAcquisition(StripBaseAcquisition):
    required_resources = [Digitizer, FastRasterScanner, MultiAxisStage]
    SPEC_LOCATION: Path = io.config_path() / "acquisition/point_scan_strip"
    Spec = StripAcquisitionSpec

    def setup_line_acquisition(self):
        return PointScanLineAcquisition(self.hw, self.system_config, self.spec)

@dataclass
class CameraAcquisitionRuntimeInfo:
    pass

"""
LineScanCameraLineAcquisition and LineScanCameraStripAcquisition provides a 
concrete implementation for strip scanning areas using a line-scan camera.

A position encoder task provides an external trigger to the camera.
"""
class LineScanCameraLineAcquisition(Acquisition):
    """
    Customized LineAcquisition that starts an encoder-derived trigger task to
    linearize the camera acquisitions.
    """
    required_resources = [LineScanCamera, Illuminator, MultiAxisLinearEncoder]

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: StripAcquisitionSpec
        self.active = threading.Event()  # to indicate data acquisition occuring

        # Set line camera properties
        self.configure_camera()

        # get reference to the encoder channel perpendicular the line-scan camera axis
        if self.hw.line_scan_camera.axis == 'y':
            if self.hw.encoders.x is None:
                raise RuntimeError("Encoder not initialized")
            self._encoder = self.hw.encoders.x
            self._lines_per_strip = round(self.spec.x_range.range / self.spec.pixel_height)
        else:
            if self.hw.encoders.y is None:
                raise RuntimeError("Encoder not initialized")
            self._encoder = self.hw.encoders.y
            self._lines_per_strip = round(self.spec.y_range.range / self.spec.pixel_height)

        self.hw.frame_grabber.prepare_buffers(nbuffers=self._lines_per_strip) # TODO may not want to allocate all lines here

        self.runtime_info = CameraAcquisitionRuntimeInfo()

    @property
    def axis(self) -> str:
        return self.hw.line_scan_camera.axis
    
    @property
    def line_frequency(self) -> units.Frequency:
        f = self.spec.integration_duty_cyle \
            / float(self.hw.line_scan_camera.integration_time)
        return units.Frequency(f)
    
    def configure_camera(self):
        self.hw.line_scan_camera.integration_time = self.spec.integration_time
        self.hw.line_scan_camera.trigger_mode = TriggerModes.EXTERNAL_TRIGGER
        # gain, etc.

        # sensor size based on spec
        obj_pixel_size = self.hw.line_scan_camera.pixel_size / self.hw.camera_optics.magnification
        roi_width = round(self.spec.line_width / obj_pixel_size)
        self.hw.frame_grabber.roi_width = roi_width

        self.hw.frame_grabber.roi_left = (self.hw.frame_grabber.pixels_width - roi_width) // 2
    
    def _get_free_product(self) -> AcquisitionProduct:
        return super()._get_free_product() # type: ignore

    def run(self):
        # Set up acquisition buffer pool
        shape = (self._lines_per_strip, self.hw.frame_grabber.roi_width, 1)
        self.init_product_pool(n=1, shape=shape, dtype=np.uint16) # TODO, switch between 8/16 bit, RGB/mono

        self.hw.illuminator.turn_on()

        self.hw.frame_grabber.start()
        self.active.set()
        self._encoder.start_triggering(self.spec.pixel_height)

        try:
            while not self._stop_event.is_set() and \
                self.hw.frame_grabber.buffers_acquired < self._lines_per_strip:
                
                # TODO get rid of this crude logging
                time.sleep(0.05)
                print(self.hw.frame_grabber.buffers_acquired)

            # confirm we got em all
            print("final acquired count:", self.hw.frame_grabber.buffers_acquired)
            superbuffer = self.hw.frame_grabber.get_next_completed_superbuffer() # type: ignore
            
            if superbuffer.shape[1] == 1:
                # Remove the singleton lines dimension, since each sub-buffer is 1-line
                superbuffer = np.squeeze(superbuffer, axis=1)
            
            # if self._encoder._timestamp_trigger_events: # type: ignore
            #     timestamps = self._encoder.read_timestamps(
            #         self.hw.frame_grabber.buffers_acquired - 1
            #     )
            # else:
            #     timestamps = None

            product = self._get_free_product()
            product.data = superbuffer
            #product.timestamps = np.array(timestamps)
            self._publish(product)

        finally:
            self.cleanup()

    def cleanup(self):
        self._encoder.stop()
        self.hw.illuminator.turn_off()
        self.hw.frame_grabber.stop()

        # Put None into queue to signal finished, stop scanning
        self._publish(None)


class LineScanCameraStripAcquisition(StripBaseAcquisition):
    required_resources = [LineScanCamera, Illuminator, MultiAxisStage]
    spec_location = io.config_path() / "acquisition/line_scan_camera_strip"
    Spec = StripAcquisitionSpec

    def setup_line_acquisition(self):
        return LineScanCameraLineAcquisition(self.hw, self.system_config, self.spec)