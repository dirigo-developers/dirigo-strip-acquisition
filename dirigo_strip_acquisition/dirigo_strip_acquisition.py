from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import math
import time

from platformdirs import user_config_dir

from dirigo import units
from dirigo.sw_interfaces.acquisition import Acquisition
from dirigo.plugins.acquisitions import LineAcquisition, LineAcquisitionSpec
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner
from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.hw_interfaces.camera import LineScanCamera
from dirigo.hw_interfaces.stage import MultiAxisStage



class StripAcquisitionSpec(LineAcquisitionSpec):
    """Specification for a strip scan acquisition."""
    def __init__(self, x_range: dict, y_range: dict, strip_overlap: float, pixel_height: str = None, **kwargs):
        super().__init__(**kwargs, buffers_per_acquisition=float('inf'))

        self.x_range = units.PositionRange(**x_range)
        self.y_range = units.PositionRange(**y_range)

        # If no pixel height specified, assume square pixel shape
        if pixel_height is not None:
            self.pixel_height = units.Position(pixel_height)
        else:
            self.pixel_height = self.pixel_size

        if not (0 <= strip_overlap < 1):
            raise ValueError(f"`overlap` must be a float between 0 and 1")
        self.strip_overlap = strip_overlap

    @property
    def lines_per_frame(self) -> int:
        """Alias for lines_per_buffer"""
        return self.lines_per_buffer 

    
class RectangularFieldStagePositionHelper:
    """Encapsulates stage runtime position calculations."""
    def __init__(self, scanner: FastRasterScanner, spec: StripAcquisitionSpec):
        self._scan_axis = scanner.axis
        self.spec = spec
    
    @cached_property
    def web_limits(self) -> units.PositionRange:
        if self._scan_axis == "x":
            return self.spec.y_range
        else:
            return self.spec.x_range

    def scan_center(self, strip_index: int) -> units.Position:
        effective_line_width = self.spec.line_width * (1 - self.spec.strip_overlap) # reduced by the overlap
        relative_position = (strip_index + 0.5) * effective_line_width

        if self._scan_axis == "x":
            return units.Position(self.spec.x_range.min + relative_position)
        else:
            return units.Position(self.spec.y_range.min + relative_position)

    @cached_property
    def nstrips(self) -> int:
        effective_line_width = self.spec.line_width * (1 - self.spec.strip_overlap)
        if self._scan_axis == "x":
            scan_axis_range = self.spec.x_range.range
        else:
            scan_axis_range = self.spec.y_range.range
        N = (scan_axis_range - self.spec.line_width) / effective_line_width
        return math.ceil(N)
    


class _StripAcquisition(Acquisition, ABC):
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
        self._line_acquisition: 'PointScanLineAcquisition'

        # Set up stage
        # -record intial state (position, velocity) to restore later

        # define functional axes
        if self.hw.fast_raster_scanner.axis == 'x':
            self._scan_axis_stage = self.hw.stage.x
            self._web_axis_stage = self.hw.stage.y
        else:
            self._scan_axis_stage = self.hw.stage.y
            self._web_axis_stage = self.hw.stage.x

        self._positioner = RectangularFieldStagePositionHelper(
            scanner=self.hw.fast_raster_scanner,
            spec=spec
        )

    @abstractmethod
    def setup_line_acquisition(self):
        pass

    def add_subscriber(self, subscriber):
        """
        Adds a subscriber to subscriber list for internal LineAcquisition worker.

        When workers subscribe to StripAcquisition, they are actually subscribing
        to internal LineAcquisition.
        """
        self._line_acqusition.add_subscriber(subscriber)

    def run(self):
        # move to start (2 axes)
        self._scan_axis_stage.move_to(self._positioner.scan_center(strip_index=0))
        self._web_axis_stage.move_to(self._positioner.web_limits.min)
        self._scan_axis_stage.wait_until_move_finished()
        self._web_axis_stage.wait_until_move_finished()

        # set velocity
        self._web_axis_stage.max_velocity = self._web_velocity

        # start line acquisition
        self._line_acqusition.start() 
        
        try:
            for strip_index in range(self._positioner.nstrips):
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
                if strip_index < (self._positioner.nstrips - 1):
                    self._scan_axis_stage.move_to(
                        self._positioner.scan_center(strip_index=strip_index + 1)
                    )

                # wait for web axis movement to come to complete stop
                self._web_axis_stage.wait_until_move_finished()

        finally:
            # Stop the line acquisition worker
            self._line_acqusition.stop()

    @cached_property
    def _web_velocity(self) -> units.Velocity:
        """The target velocity for web axis movement."""
        if self.spec.bidirectional_scanning:
            v = 2 * self.hw.fast_raster_scanner.frequency * self.spec.pixel_height
        else:
            v = self.hw.fast_raster_scanner.frequency * self.spec.pixel_height
        return units.Velocity(v)
    
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
    """Customized LineAcquisition that starts linear position measurement 
    (encoders) and allows measuring them by overriding the read_position method.
    """

    def __init__(self, hw, spec: LineAcquisitionSpec):
        super().__init__(hw, spec)

    def run(self):
        """Identical to LineAcquisition's run(), except also starts and stops 
        the encoders.
        """
        self.hw.encoders.start(self.hw)
        try:
            super().run()
        finally:
            self.hw.encoders.stop()

    def read_positions(self):
        """Override provides sample positions from linear position encoders."""
        return self.hw.encoders.read(self.spec.records_per_buffer) 


class PointScanStripAcquisition(_StripAcquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, MultiAxisStage] 
    SPEC_LOCATION: str = Path(user_config_dir("Dirigo")) / "acquisition/point_scan_strip"
    SPEC_OBJECT = StripAcquisitionSpec

    def setup_line_acquisition(self, hw, spec):
        self._line_acqusition = PointScanLineAcquisition(hw, spec)


"""
LineScanCameraLineAcquisition and LineScanCameraStripAcquisition provides a 
concrete implementation for strip scanning areas using a line-scan camera.

A position encoder task provides an external trigger to the camera.
"""

class LineScanCameraLineAcquisition(Acquisition):
    REQUIRED_RESOURCES = [Illuminator, LineScanCamera]

    def __init__(self, hw, spec):
        super().__init__(hw, spec) # sets up thread, inbox, stores hw, checks resources

        # Set line camera properties
        self.configure_camera()
    
    def configure_camera(self):
        # TODO make into profile
        self.hw.line_scan_camera.integration_time = units.Time("0.2 ms")
        # gain, external trigger, etc.

    def run(self):
        # set up buffers
        self.hw.frame_grabber.prepare_buffers(nbuffers=100)

        self.hw.line_scan_camera.start()

        try:
            while not self._stop_event.is_set() and \
                self.hw.frame_grabber.buffers_acquired < 100: #self.spec.buffers_per_acquisition:

                # buffer = self.hw.frame_grabber.get_next_completed_buffer()
                # self.publish(buffer)
                pass

        finally:
            self.cleanup()

    def cleanup(self):
        pass


class LineScanCameraStripAcquisition(_StripAcquisition):
    REQUIRED_RESOURCES = [Illuminator, LineScanCamera, MultiAxisStage] 
    SPEC_LOCATION: str = Path(user_config_dir("Dirigo")) / "acquisition/line_scan_camera_strip"
    SPEC_OBJECT = StripAcquisitionSpec

    def setup_line_acquisition(self, hw, spec):
        self._line_acqusition = LineScanCameraLineAcquisition(hw, spec)