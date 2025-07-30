from typing import Optional, Literal
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import math, time

import numpy as np

from dirigo import units, io
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct, AcquisitionSpec
from dirigo.plugins.acquisitions import (
    LineAcquisitionSpec, LineAcquisition, 
    LineCameraLineAcquisitionSpec, LineCameraLineAcquisition
)
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner
from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.hw_interfaces.camera import FrameGrabber, LineCamera
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder, LinearEncoder


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




# ---------- Strip acquisitions ----------
class RasterScanStripAcquisitionSpec(LineAcquisitionSpec):
    """Alias of LineAcquisitionSpec"""
    pass


class RasterScanStripAcquisition(LineAcquisition):
    """
    Customized LineAcquisition that starts linear position measurement 
    (encoders) and allows measuring them by overriding the read_position method.
    """
    def __init__(self, hw, system_config, spec,
                 thread_name: str = "Raster strip acquisition"):
        super().__init__(hw, system_config, spec, thread_name)
        self.spec: RasterScanStripAcquisitionSpec

        self.hw.encoders.x.sample_clock_channel \
            = self.system_config.encoders['x_config']["sample_clock_channel"]
        self.hw.encoders.y.sample_clock_channel \
            = self.system_config.encoders['y_config']["sample_clock_channel"]

        self._n_positions_read = 0

    @property
    def axis(self) -> str:
        return self.hw.fast_raster_scanner.axis
    
    @property
    def line_rate(self) -> units.Frequency:
        if self.spec.bidirectional_scanning:
            return 2 * self.hw.fast_raster_scanner.frequency
        else:
            return self.hw.fast_raster_scanner.frequency

    def run(self):
        """
        Adds to LineAcquisition's run():
        - starts and stops the position encoders
        - centers and parks the slow axis scanner (if present)
        """
        init_pos = (self.hw.stages.x.position, self.hw.stages.y.position)
        self.hw.encoders.start_logging(
            initial_position=init_pos,
            line_rate=self.hw.fast_raster_scanner.frequency
        )  # type: ignore

        try:
            # If slow raster scanner present, center 
            if self.hw.exists('slow_raster_scanner'):
                self.hw.slow_raster_scanner.center()

            super().run()

        finally:
            if self.hw.exists('slow_raster_scanner'):
                self.hw.slow_raster_scanner.park()

            self.hw.encoders.stop()


    def read_positions(self):
        """Override provides sample positions from linear position encoders."""  # TODO: order dimensions scan, web?
        if self.spec.bidirectional_scanning and self._n_positions_read == 0:
            self._prev_pos = self.hw.encoders.read_positions(1)
            
        positions = self.hw.encoders.read_positions(self.spec.records_per_buffer) 
        self._n_positions_read += self.spec.records_per_buffer

        # Bidi scanning: 1 trigger/sample per 2 lines, interpolate missing line positions
        if self.spec.bidirectional_scanning:
            pos_0 = np.concatenate(
                (self._prev_pos, positions[:-1,:]),
                axis=0
            )
            pos_1 = positions.copy()
            self._prev_pos[:] = positions[-1,:]

            # interlace the interpolated positions
            positions = np.empty(shape=(2*self.spec.records_per_buffer, 2))
            positions[0::2,:] = pos_0
            positions[1::2,:] = (pos_0 + pos_1) / 2

        return positions


class LineCameraStripAcquisitionSpec(LineCameraLineAcquisitionSpec):
    """Alias of LineCameraLineAcquisitionSpec"""
    pass


class LineCameraStripAcquisition(LineCameraLineAcquisition): 
    """
    Customized LineCameraLineAcquisition that adds functionality to run()
    """
    required_resources = [FrameGrabber, LineCamera, MultiAxisLinearEncoder, Illuminator]
    Spec = LineCameraStripAcquisitionSpec

    def __init__(self, hw, system_config, spec,
                 thread_name: str = "Line camera strip acquisition"):
        super().__init__(hw, system_config, spec, thread_name)
        self.spec: LineCameraLineAcquisitionSpec

        self._n_positions_read = 0

        # TODO legitamize this hack
        # self.hw.encoders.x._sample_clock_channel = self.hw.encoders.x._trigger_channel
        # self.hw.encoders.y._sample_clock_channel = self.hw.encoders.x._trigger_channel
        #self.hw.encoders.x._timestamp_trigger_events = True

        w = ("x" if self.hw.line_camera.axis == "y" else "y") +  "_config"
        sample_clock_channel = self.system_config.encoders[w]["trigger_channel"]
        self.hw.encoders.x.sample_clock_channel = sample_clock_channel
        self.hw.encoders.y.sample_clock_channel = sample_clock_channel

    # property: axis defined in super class
    # property: line_rate defined in super class

    def run(self):
        """
        Adds to LineCameraLineAcquisition's run():
        - starts and stop illuminator
        - starts and stops the linearizing trigger
        - starts and stops the position encoders
        - centers and parks the slow axis scanner (if present)
        """
        self.hw.illuminator.turn_on() 
        
        try:
            super().run()

        finally:
            self.hw.illuminator.turn_off() 
            self.hw.encoders.stop()

    def _read_positions(self):
        """Override provides sample positions from linear position encoders.""" 
        positions = self.hw.encoders.read_positions(self.spec.lines_per_buffer)         
        self._n_positions_read += self.spec.lines_per_buffer
        return positions



# ---------- Stitched acquisition base class ----------
class StitchedAcquisitionSpec(AcquisitionSpec):
    """Specification for a stitched multi-strip acquisition."""
    def __init__(self, 
                 x_range: units.PositionRange | dict, 
                 y_range: units.PositionRange | dict, 
                 z_range: units.PositionRange | dict, 
                 z_step: units.Position | str,
                 strip_overlap: float = 0.05
                 ) -> None:
        
        if isinstance(x_range, units.PositionRange):
            self.x_range = x_range
        else:
            self.x_range = units.PositionRange(**x_range)

        if isinstance(y_range, units.PositionRange):
            self.y_range = y_range
        else:
            self.y_range = units.PositionRange(**y_range)

        if isinstance(z_range, units.PositionRange):
            self.z_range = z_range
        else:
            self.z_range = units.PositionRange(**z_range)
        
        if isinstance(z_step, units.Position):
            self.z_step = z_step
        else:
            self.z_step = units.Position(z_step)

        if not (0 <= strip_overlap < 1):
            raise ValueError(f"`overlap` must be a float between 0 and 1")
        self.strip_overlap = strip_overlap

    @cached_property
    def z_steps(self) -> int:
        return math.ceil( abs(self.z_range.range / self.z_step) )
      

class StitchedAcquisition(Acquisition, ABC):
    """
    Base class for stitched multi-strip scan acquisition worker.

    Uses coordinated line acquisition and stage movement to capture a large 
    field. Define 'scan axis' as the axis parallel to the acquired lines. Define
    'web axis' as perpendicular to scan axis. (Terms borrowed from industrial
    machine vision 'web inspection')

    Concrete subclasses need to define required_resources, spec_location, 
    Spec, and provide a method for setup_strip_acquisition()
    """
    Spec = StitchedAcquisitionSpec
    spec_location = io.config_path() / "acquisition/stitched"
    
    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec)
        self.spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec # to refine type hints

        # set up internal strip acquisition        
        self._strip_acquisition = self.setup_strip_acquisition()
        self._strip_acquisition.spec.buffers_per_acquisition = -1 # codes for unlimited buffers

        # define functional axes (web & scan axes terminology from industrial web inspection)
        if self._strip_acquisition.axis == 'x':
            self._scan_axis_stage = self.hw.stages.x
            self._web_axis_stage = self.hw.stages.y # web axis = perpendicular to the fast raster scanner / line-scan camera axis
            self._web_encoder = self.hw.encoders.y
            n_pixels_scan = round(self.spec.x_range.range / self._strip_acquisition.spec.pixel_size)
            n_pixels_web  = round(self.spec.y_range.range / self._strip_acquisition.spec.pixel_size)
        else:
            self._scan_axis_stage = self.hw.stages.y
            self._web_axis_stage = self.hw.stages.x
            self._web_encoder = self.hw.encoders.x
            n_pixels_scan = round(self.spec.y_range.range / self._strip_acquisition.spec.pixel_size)
            n_pixels_web  = round(self.spec.x_range.range / self._strip_acquisition.spec.pixel_size)

        if isinstance(self._strip_acquisition.data_acquisition_device, Digitizer):  # TODO make dynamic n_channels and axis error part of API
            n_channels = self._strip_acquisition.hw.digitizer.acquire.n_channels_enabled
            axis_error = self.hw.laser_scanning_optics.stage_scanner_angle
        else:
            n_channels = 3 if self.runtime_info.camera_bit_depth == 24 else 1 # for RGB cameras
            axis_error = self.hw.camera_optics.stage_camera_angle

        self.positioner = RectangularFieldStagePositionHelper(
            scan_axis=self._strip_acquisition.axis,
            axis_error=axis_error,
            line_width=self._strip_acquisition.spec.line_width, # TODO, remove line_width since it's already in spec
            spec=spec
        )
        self._final_shape = (self.spec.z_steps, n_pixels_scan, n_pixels_web, n_channels) 

    @property
    def final_shape(self) -> tuple[int,int,int,int]:
        """Shape of the final stitched image with dimensions: (z, scan, web, chan)"""
        return self._final_shape
        
    @abstractmethod
    def setup_strip_acquisition(self) -> "RasterScanStripAcquisition | LineCameraStripAcquisition":
        # Must be implemented in subclass
        pass

    def add_subscriber(self, subscriber):
        """
        Adds a subscriber to subscriber list for internal StripAcquisition worker.

        When workers subscribe to StitchedAcquisition, they are actually 
        subscribing to internal StripAcquisition.
        """
        self._strip_acquisition.add_subscriber(subscriber)

    def run(self):
        original_position = (
            self.hw.stages.x.position, 
            self.hw.stages.y.position,
            self.hw.objective_z_scanner.position
        )
        # move to start (2 axes + objective z scanner)
        self.hw.objective_z_scanner.move_to(self.spec.z_range.min)
        self._scan_axis_stage.move_to(
            self.positioner.scan_center(strip_index=0)
        )
        self._web_axis_stage.move_to(
            self.positioner.web_limits.min - self.spec.pixel_size # a bit extra movement to be sure we trigger enough samples
        )
        #print(f"Are stages moving. Scan {self._scan_axis_stage.moving}")
        #print(f"Are stages moving. Web {self._web_axis_stage.moving}")
        time.sleep(0.050) # to make certain the stages have started moving
        self._scan_axis_stage.wait_until_move_finished()
        self._web_axis_stage.wait_until_move_finished()

        # set strip velocity
        self._original_web_velocity = self._web_axis_stage.max_velocity
        self._web_axis_stage.max_velocity = self._web_velocity
        _ = self._web_period # caches _web_period

        # start line acquisition & hold until it is 'active' (prevents premature movements)
        self._strip_acquisition.start() 
        while not self._strip_acquisition.active.is_set():
            time.sleep(0.001) # wait (active event indicates data is acquiring)

        try:
            for z_index in range(self.spec.z_steps):

                for strip_index in range(self.positioner.n_strips):
                    if self._stop_event.is_set():
                        break # Terminate acquisitions

                    self.reset_encoders("forward" if (strip_index % 2) == 0 else "reverse")
                    print(f"Starting strip {strip_index} of {self.positioner.n_strips}")

                    # start web axis movement
                    if strip_index % 2:
                        strip_end_position = self.positioner.web_limits.min - self.spec.pixel_size
                    else:
                        strip_end_position = self.positioner.web_limits.max

                    self._web_axis_stage.move_to(strip_end_position)

                    if strip_index < (self.positioner.n_strips - 1):
                        # wait until web axis decceleration
                        time.sleep(self._web_period)

                        # begin lateral movement to the next strip
                        self._scan_axis_stage.move_to(
                            self.positioner.scan_center(strip_index=strip_index + 1)
                        )
                    else:
                        time.sleep(0.050) # Wait to be sure the stage is actually moving

                    # wait for web axis movement to come to complete stop
                    self._web_axis_stage.wait_until_move_finished()
                
                if z_index < (self.spec.z_steps - 1): # if not on last z level
                    
                    # Change web axis velocity (usually faster)
                    self._web_axis_stage.max_velocity = self._original_web_velocity

                    # Move Z
                    self.hw.objective_z_scanner.move_to(
                        self.spec.z_range.min + (z_index+1) * self.spec.z_step
                    )

                    # Move back to XY starting point
                    self._scan_axis_stage.move_to(
                        self.positioner.scan_center(strip_index=0)
                    )
                    self._web_axis_stage.move_to(
                        self.positioner.web_limits.min - self.spec.pixel_size
                    )

                    time.sleep(0.050) # to make certain the stages have started moving
                    self._scan_axis_stage.wait_until_move_finished()
                    self._web_axis_stage.wait_until_move_finished()

                    # Set acquisition web axis velocity
                    self._web_axis_stage.max_velocity = self._web_velocity

        finally:
            # Stop the line acquisition Worker
            self._strip_acquisition.stop()

            if self.hw.stages.x.moving:
                self.hw.stages.x.stop()
            if self.hw.stages.y.moving:
                self.hw.stages.y.stop()

            # Revert the web axis velocity
            self._web_axis_stage.max_velocity = self._original_web_velocity

            # Return to original position
            self.hw.stages.x.wait_until_move_finished()
            self.hw.stages.y.wait_until_move_finished()
            self.hw.stages.x.move_to(original_position[0])
            self.hw.stages.y.move_to(original_position[1])
            self.hw.objective_z_scanner.move_to(original_position[2])

    def reset_encoders(self, direction: Literal['forward', 'reverse']):
        """Provide subclass implementation to start a camera trigger"""
        pass

    @property
    def runtime_info(self):
        return self._strip_acquisition.runtime_info

    @cached_property
    def _web_velocity(self) -> units.Velocity:
        """The target velocity for web axis movement."""
        return units.Velocity(
            float(self._strip_acquisition.line_rate) * float(self._strip_acquisition.spec.pixel_size)
        )
    
    @cached_property
    def _web_period(self) -> units.Time:
        """The approximate period of time required to capture 1 strip."""
        x = float(self.positioner.web_limits.range)
        a = float(self._web_axis_stage.acceleration)
        v_max = float(self._web_velocity)
        x_crit = 2 * v_max**2 / a

        if x <= x_crit:
            return units.Time(2 * math.sqrt(x / a))
        else:
            return units.Time((2 * v_max/a) + (x - x_crit)/v_max)

    

# ---------- Stitched acquisitions concrete classes ----------
class RasterScanStitchedAcquisitionSpec(StitchedAcquisitionSpec, RasterScanStripAcquisitionSpec):
    def __init__(self,
                 x_range: units.PositionRange | dict,
                 y_range: units.PositionRange | dict,
                 z_range: units.PositionRange | dict,
                 z_step: units.Position | str,
                 strip_overlap: float,
                 line_width: units.Position | str,
                 pixel_size: units.Position | str,
                 fill_fraction: float,
                 bidirectional_scanning: bool,
                 lines_per_buffer: int,
                 **kwargs) -> None:
        StitchedAcquisitionSpec.__init__(
            self,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            z_step=z_step,
            strip_overlap=strip_overlap
        )
        RasterScanStripAcquisitionSpec.__init__(
            self,
            line_width=line_width,
            pixel_size=pixel_size,
            fill_fraction=fill_fraction,
            bidirectional_scanning=bidirectional_scanning,
            lines_per_buffer=lines_per_buffer
        )


class RasterScanStitchedAcquisition(StitchedAcquisition):
    required_resources = [Digitizer, FastRasterScanner, MultiAxisStage, MultiAxisLinearEncoder]
    SPEC_LOCATION: Path = io.config_path() / "acquisition/point_scan_strip"
    Spec = RasterScanStitchedAcquisitionSpec

    def setup_strip_acquisition(self):
        return RasterScanStripAcquisition(self.hw, self.system_config, self.spec)
    
    @property
    def digitizer_profile(self):
        self._strip_acquisition: RasterScanStripAcquisition
        return self._strip_acquisition.digitizer_profile


class LineCameraStitchedAcquisitionSpec(StitchedAcquisitionSpec, LineCameraStripAcquisitionSpec):
    def __init__(self,
                 x_range: units.PositionRange | dict,
                 y_range: units.PositionRange | dict,
                 z_range: units.PositionRange | dict,
                 z_step: units.Position | str,
                 strip_overlap: float,
                 line_width: units.Position | str,
                 pixel_size: units.Position | str,
                 integration_time: units.Time | str,
                 line_period: units.Time | str,
                 lines_per_buffer: int,
                 **kwargs):
        StitchedAcquisitionSpec.__init__(
            self,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            z_step=z_step,
            strip_overlap=strip_overlap
        )
        LineCameraStripAcquisitionSpec.__init__(
            self,
            line_width=line_width,
            pixel_size=pixel_size,
            integration_time=integration_time,
            line_period=line_period,
            lines_per_buffer=lines_per_buffer,
            **kwargs
        )


class LineCameraStitchedAcquisition(StitchedAcquisition):
    required_resources = [FrameGrabber, LineCamera, Illuminator, MultiAxisStage, MultiAxisLinearEncoder]
    Spec = LineCameraStitchedAcquisitionSpec

    def setup_strip_acquisition(self):
        return LineCameraStripAcquisition(self.hw, self.system_config, self.spec)
    
    @property
    def camera_profile(self):
        self._strip_acquisition: LineCameraStripAcquisition
        return self._strip_acquisition.camera_profile
    
    def reset_encoders(self, direction):
        self.hw.encoders.stop()

        # Wait for stage to settle, so that initial positions remain correct
        self.hw.stages.x.wait_until_move_finished()
        self.hw.stages.y.wait_until_move_finished()
        x_0 = self.hw.stages.x.position # TODO may be better to use positions from Positioner
        y_0 = self.hw.stages.y.position

        self._web_encoder.start_triggering(
            distance_per_trigger    = self.spec.pixel_size, 
            direction               = direction
        )
        self.hw.encoders.start_logging(
            initial_position    = (x_0, y_0),    # type: ignore
            line_rate           = 1 / self.spec.line_period,    # type: ignore
        )  
    


# ---------- Helpers ----------
class RectangularFieldStagePositionHelper:
    """Encapsulates stage runtime position calculations."""
    EPS = units.Position(1e-9)
    def __init__(self, 
                 scan_axis: str, 
                 axis_error: units.Angle, 
                 line_width: units.Position,
                 spec: StitchedAcquisitionSpec):
        self._scan_axis = scan_axis
        self._axis_error = axis_error
        self._line_width = line_width
        self._spec = spec
    
    @cached_property
    def web_limits(self) -> units.PositionRange:
        if self._scan_axis == "x":
            return self._spec.y_range
        else:
            return self._spec.x_range

    def scan_center(self, strip_index: int) -> units.Position:
        """Return the center position (along scan dimension) for `strip_index`"""
        effective_line_width = self._line_width * (1 - self._spec.strip_overlap) # reduced by the overlap
        relative_position = strip_index * effective_line_width + self._line_width / 2

        if self._scan_axis == "x":
            return units.Position(self._spec.x_range.min + relative_position)
        else:
            return units.Position(self._spec.y_range.min + relative_position)
        
    def web_min(self, strip_index: int) -> units.Position:
        shear = (self.scan_center(strip_index) - self.scan_center(self.n_strips//2)) \
            * float(self._axis_error)
        if self._scan_axis == "x":
            return units.Position(self._spec.y_range.min) - shear
        else:
            return units.Position(self._spec.x_range.min) - shear

    def web_max(self, strip_index: int) -> units.Position:
        shear = (self.scan_center(strip_index) - self.scan_center(self.n_strips//2)) \
            * float(self._axis_error)
        if self._scan_axis == "x":
            return units.Position(self._spec.y_range.max) - shear
        else:
            return units.Position(self._spec.x_range.max) - shear

    @cached_property
    def n_strips(self) -> int:
        effective_line_width = self._line_width * (1 - self._spec.strip_overlap)
        if self._scan_axis == "x":
            scan_axis_range = self._spec.x_range.range
        else:
            scan_axis_range = self._spec.y_range.range
        N = (scan_axis_range - self._line_width - self.EPS) / effective_line_width + 1
        return max(math.ceil(N), 1)
    
