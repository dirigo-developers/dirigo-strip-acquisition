from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import math
import time
from typing import Optional

from platformdirs import user_config_dir
import numpy as np

from dirigo import units
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.plugins.acquisitions import LineAcquisition, LineAcquisitionSpec
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner
from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.hw_interfaces.camera import LineScanCamera
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder

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
        super().__init__(buffers_per_acquisition=float('inf'), **kwargs)

        self.x_range = units.PositionRange(**x_range)
        self.y_range = units.PositionRange(**y_range)

        self.pixel_height = self.pixel_size # constrain to square pixel

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
    def __init__(self, scan_axis: str, spec: StripAcquisitionSpec):
        self._scan_axis = scan_axis
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

    Concrete subclasses need to define REQUIRED_RESOURCES, SPEC_LOCATION, 
    SPEC_OBJECT, and provide a method of setup_line_acquisition()
    """
    
    def __init__(self, hw, spec: StripAcquisitionSpec):
        super().__init__(hw, spec)
        self.spec: StripAcquisitionSpec

        # set up internal line acquisition
        # shares hw, and spec (internal _stop_event is not shared)
        self.setup_line_acquisition(hw, spec)
        self._line_acquisition: 'PointScanLineAcquisition' | 'LineScanCameraLineAcquisition'

        # define functional axes
        if self._line_acquisition.axis == 'x':
            self._scan_axis_stage = self.hw.stage.x
            self._web_axis_stage = self.hw.stage.y # web axis = perpendicular to the fast raster scanner / line-scan camera axis
        else:
            self._scan_axis_stage = self.hw.stage.y
            self._web_axis_stage = self.hw.stage.x

        self._positioner = RectangularFieldStagePositionHelper(
            scan_axis=self._line_acquisition.axis,
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
    def setup_line_acquisition(self):
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
        self._scan_axis_stage.move_to(self._positioner.scan_center(strip_index=0))
        self._web_axis_stage.move_to(self._positioner.web_limits.min)
        self._scan_axis_stage.wait_until_move_finished()
        self._web_axis_stage.wait_until_move_finished()

        # set strip velocity
        self._original_web_velocity = self._web_axis_stage.max_velocity
        self._web_axis_stage.max_velocity = self._web_velocity

        # start line acquisition
        self._line_acquisition.start() 
        while not self._line_acquisition.active.is_set():
            time.sleep(0.001) # spin until active (indicates data acquiring)
        
        try:
            for strip_index in range(self._positioner.n_strips):
                if self._stop_event.is_set():
                    # Terminate acquisitions
                    break

                # start web axis movement
                if strip_index % 2:
                    strip_end_position = self._positioner.web_limits.min
                else:
                    strip_end_position = self._positioner.web_limits.max

                self._web_axis_stage.move_to(strip_end_position)

                # wait until web axis decceleration
                time.sleep(self._web_period + units.Time('10 ms')) # the last part is empirical

                # begin lateral movement to the next strip
                if strip_index < (self._positioner.n_strips - 1):
                    self._scan_axis_stage.move_to(
                        self._positioner.scan_center(strip_index=strip_index + 1)
                    )

                # wait for web axis movement to come to complete stop
                self._web_axis_stage.wait_until_move_finished()

                # TODO, call some sort of line acquisition reset

        finally:
            # Stop the line acquisition worker
            self._line_acquisition.stop()

            # Revert the web axis velocity
            self._web_axis_stage.max_velocity = self._original_web_velocity

    @cached_property
    def _web_velocity(self) -> units.Velocity:
        """The target velocity for web axis movement."""
        return units.Velocity(
            float(self._line_acquisition.line_frequency) * float(self.spec.pixel_height)
        )
    
    @cached_property
    def _web_period(self) -> units.Time:
        """The approximate period of time required to capture 1 strip."""
        return self._positioner.web_limits.range / self._web_velocity
    

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

    def __init__(self, hw, spec: LineAcquisitionSpec):
        super().__init__(hw, spec)

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
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, MultiAxisStage] 
    SPEC_LOCATION: str = Path(user_config_dir("Dirigo")) / "acquisition/point_scan_strip"
    SPEC_OBJECT = StripAcquisitionSpec

    def setup_line_acquisition(self, hw, spec):
        self._line_acquisition = PointScanLineAcquisition(hw, spec)


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
    REQUIRED_RESOURCES = [LineScanCamera, Illuminator, MultiAxisLinearEncoder]

    def __init__(self, hw, spec):
        super().__init__(hw, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: StripAcquisitionSpec

        # Set line camera properties
        self.configure_camera()

        # get reference to the encoder channel perpendicular the line-scan camera axis
        if self.hw.line_scan_camera.axis == 'y':
            self._encoder = self.hw.encoders.x
            self._lines_per_strip = round(self.spec.x_range.range / self.spec.pixel_height)
        else:
            self._encoder = self.hw.encoders.y
            self._lines_per_strip = round(self.spec.y_range.range / self.spec.pixel_height)

        self.hw.frame_grabber.prepare_buffers(nbuffers=self._lines_per_strip) # TODO may not want to allocate all lines here

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
        

    def run(self):
        self.hw.illuminator.turn_on()

        self.hw.frame_grabber.start()
        self._encoder.start_triggering(self.spec.pixel_height)

        try:
            while not self._stop_event.is_set() and \
                self.hw.frame_grabber.buffers_acquired < self._lines_per_strip:
                
                # TODO get rid of this crude logging
                time.sleep(0.05)
                print(self.hw.frame_grabber.buffers_acquired)

            # confirm we got em all
            print("final acquired count:", self.hw.frame_grabber.buffers_acquired)
            superbuffer = self.hw.frame_grabber.get_next_completed_superbuffer()
            
            if superbuffer.shape[1] == 1:
                # Remove the singleton lines dimension, since each sub-buffer is 1-line
                superbuffer = np.squeeze(superbuffer, axis=1)

            self.publish(AcquisitionProduct(
                data=superbuffer,
                timestamps=np.array(
                    self._encoder.read_timestamps(self.hw.frame_grabber.buffers_acquired-1)
                )
            ))

        finally:
            self.cleanup()

    def cleanup(self):
        self._encoder.stop()
        self.hw.illuminator.turn_off()
        self.hw.frame_grabber.stop()

        # Put None into queue to signal finished, stop scanning
        self.publish(None)


class LineScanCameraStripAcquisition(StripBaseAcquisition):
    REQUIRED_RESOURCES = [LineScanCamera, Illuminator, MultiAxisStage] 
    SPEC_LOCATION: str = Path(user_config_dir("Dirigo")) / "acquisition/line_scan_camera_strip"
    SPEC_OBJECT = StripAcquisitionSpec

    def setup_line_acquisition(self, hw, spec):
        self._line_acquisition = LineScanCameraLineAcquisition(hw, spec)