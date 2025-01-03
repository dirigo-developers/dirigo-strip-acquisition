from functools import cached_property
from pathlib import Path
import math
import time

from platformdirs import user_config_dir

from dirigo import units
from dirigo.sw_interfaces.acquisition import Acquisition
from dirigo.builtins.acquisitions import LineAcquisition, LineAcquisitionSpec
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner
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


class StripAcquisition(Acquisition):
    """
    Strip scan acquistion worker implementing the Dirigo Acquisition interface.

    Uses coordinated line acquisition and stage movement to capture a large 
    field. Define 'scan axis' as the axis parallel to the acquired lines. Define
    'web axis' as perpendicular to scan axis.

    Limitation: currently only supports pointed-scanned line acquisition.
    """
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, MultiAxisStage] # TODO, offer linecamera alternative
    SPEC_LOCATION: str = Path(user_config_dir("Dirigo")) / "acquisition/strip"
    SPEC_OBJECT = StripAcquisitionSpec

    def __init__(self, hw, data_queue, spec):
        super().__init__(hw, data_queue, spec)
        self.spec: StripAcquisitionSpec

        # set up internal line acquisition
        # shares hw, data_queue, and spec (internal _stop_event is not shared)
        self._line_acqusition = LineAcquisition(hw, data_queue, spec)

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
        # add some buffer time?

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