from numba import njit, prange
import numpy as np

from dirigo import units
from dirigo.sw_interfaces.worker import Product
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.plugins.processors import RasterFrameProcessor  
from dirigo_strip_acquisition import PointScanStripAcquisition, StripAcquisitionSpec


# line_placement_kernel
@njit(fastmath=True, cache=True)
def _line_placement_kernel(strip: np.ndarray, 
                           lines: np.ndarray, 
                           positions: np.ndarray, 
                           strip_min: float, 
                           pixel_height: float) -> None:
    n_height, n_width, n_chan = strip.shape

    for line_idx in prange(lines.shape[0]):
        # compute closest row of strip
        strip_row = round((positions[line_idx,0] - strip_min) / pixel_height)

        if (strip_row < 0) or (strip_row >= n_height):
            continue # line is out of strip area

        # TODO, shift for lateral movement

        strip[strip_row,:,:] = lines[line_idx,:,:] # compound/average?
        #interp_lines[strip_index,strip_row_idx] = 0



class StripProcessor(Processor):
    """  """
    def __init__(self, upstream: RasterFrameProcessor):
        super().__init__(upstream)

        self._spec: StripAcquisitionSpec
        self._acq: PointScanStripAcquisition
        self._data_range = upstream.data_range

        self._prev_position = np.array([0, 0], dtype=np.float64)
        if self._acq._web_axis_stage.axis == "x": 
            self._strip_height = self._spec.x_range.range
            self._prev_position[0] = self._spec.x_range.min               # x
            self._prev_position[1] = self._acq._positioner.scan_center(0) # y
            self._web_min = self._spec.x_range.min
        elif self._acq._web_axis_stage.axis == "y":
            self._strip_height = self._spec.y_range.range
            self._prev_position[0] = self._acq._positioner.scan_center(0) # x
            self._prev_position[1] = self._spec.y_range.min               # y
            self._web_min = self._spec.y_range.min
        else:
            raise RuntimeError

        self._strip_shape = (
            round(self._strip_height / self._spec.pixel_height),
            self._spec.pixels_per_line,
            self._acq.hw.digitizer.acquire.n_channels_enabled
        )

        self.init_product_pool(n=3, shape=self._strip_shape, dtype=np.int16)
        
        self._strip_idx = 0

    def run(self):
        try:
            strip_product = self.get_free_product()

            while True:
                frame_product: ProcessorProduct = self.inbox.get()
                if frame_product is None: return

                with frame_product:
                    # interpolate (bidi-scanning expect 1:2 positions:lines)
                    p = np.array(frame_product.positions)
                    # positions: actually the 2nd position of the current frame
                    # up to (and including) the first position of the NEXT frame

                    if self._spec.bidirectional_scanning:
                        p1 = np.concatenate(
                            (self._prev_position[np.newaxis,:], p[:-1,:]),
                            axis=0
                        )
                        p2 = p[:,:]

                        positions = np.zeros(
                            shape=(frame_product.data.shape[0],2), 
                            dtype=np.float64
                        )
                        positions[0::2,:] = p1
                        positions[1::2,:] = (p1 + p2) / 2 # interpolated positions
                    else:
                        positions = np.concatenate(
                            (self._prev_position[np.newaxis,:], p[:-1,:]),
                            axis=0
                        )

                    self._prev_position = p[-1,:]
                    
                    _line_placement_kernel(             # add lines to strip
                        strip=strip_product.data, 
                        lines=frame_product.data,
                        positions=positions,
                        strip_min=self._web_min,
                        pixel_height=self._spec.pixel_height
                    )

                    # Check whether this is the end of strip
                    axis = 0 if self._acq._web_axis_stage.axis == "y" else 1
                    width = self._spec.line_width * (1 - self._spec.strip_overlap)

                    latest_strip_idx = round(
                        (self._prev_position[axis] - self._acq._positioner.scan_center(0)) / width
                    )

                    if latest_strip_idx != self._strip_idx:
                        # Moving to next strip, publish
                        print(f"Publishing strip {self._strip_idx} with size: {strip_product.data.shape}")
                        self.publish(strip_product)

                        strip_product = self.get_free_product()

                        # add any leftover lines to new strip
                        _line_placement_kernel( 
                            strip=strip_product.data, 
                            lines=frame_product.data,
                            positions=positions,
                            strip_min=self._web_min,
                            pixel_height=self._spec.pixel_height
                        )
                        self._strip_idx = latest_strip_idx

        finally:
            # publish final strip (completed or not)
            print(f"Publishing FINAL strip {self._strip_idx} with size: {strip_product.data.shape}")
            self.publish(strip_product)

            # send shutdown sentinel
            self.publish(None)
    
    @property
    def data_range(self) -> units.IntRange:
        return self._data_range
    

class TileProduct(Product):
    __slots__ = ("data", "tile_coords")
    def __init__(self, 
                 pool, 
                 data: np.ndarray,
                 tile_coords: tuple = None):
        super().__init__(pool)
        self.data = data
        self.tile_coords = tile_coords


class StripStitcher(Processor):
    """  """
    def __init__(self, upstream: StripProcessor, tile_shape=(512,512)):
        super().__init__(upstream)
        self._data_range = upstream.data_range
        n_channels = self._acq.hw.digitizer.acquire.n_channels_enabled
        self._tile_shape = tile_shape + (n_channels,)
        
        self._spec: StripAcquisitionSpec
        self._acq: PointScanStripAcquisition

        if self._acq._web_axis_stage.axis == "x":
            Nx = round(self._spec.x_range.range / self._spec.pixel_height)
            Ny = round(self._spec.y_range.range / self._spec.pixel_size)
            self._tiles_web  = Nx // self._tile_shape[0]
            self._tiles_scan = Ny // self._tile_shape[0]
        elif self._acq._web_axis_stage.axis == "y":
            Ny = round(self._spec.y_range.range / self._spec.pixel_height)
            Nx = round(self._spec.x_range.range / self._spec.pixel_size)
            self._tiles_web  = Ny // self._tile_shape[0]
            self._tiles_scan = Nx // self._tile_shape[0]
        self._full_shape = (Nx, Ny, n_channels)

        self.init_product_pool(n=10, shape=self._tile_shape, dtype=np.int16)
 
    def run(self):
        t_i = 0 # tile index 

        while True:
            strip_product: ProcessorProduct = self.inbox.get()
            if strip_product is None: 
                self.publish(None) # send sentinel None
                return

            with strip_product:
                while True:
                    # What range does this cover?
                    t_w = t_i %  self._tiles_web # tile coordinate web dim
                    t_s = t_i // self._tiles_web # tile coordinate scan dim
                    if ((t_s+1) * self._tile_shape[1]) > strip_product.data.shape[1]:
                        t_i = 0
                        break

                    tile_product = self.get_free_product()
                    tile_product.tile_coords = (t_s, t_w)

                    tile_product.data[...] = strip_product.data[
                        (t_w * self._tile_shape[0]):((t_w+1) * self._tile_shape[0]),
                        (t_s * self._tile_shape[1]):((t_s+1) * self._tile_shape[1]),
                        :
                    ] # TODO, transpose

                    print(f"Publishing tile {t_i}: ({t_s},{t_w})")
                    self.publish(tile_product)
                    t_i += 1


    def init_product_pool(self, n, shape, dtype):
        for _ in range(n):
            aq_buf = TileProduct(
                pool=self._product_pool,
                data=np.empty(shape, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(aq_buf)

    def get_free_product(self) -> TileProduct:
        return self._product_pool.get()

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range