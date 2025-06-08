import math
from typing import Optional

from numba import njit, prange, int16, uint16, float64, int64, boolean
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from dirigo import units, io
from dirigo.sw_interfaces.worker import Product, EndOfStream
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.plugins.processors import RasterFrameProcessor

from dirigo_strip_acquisition.acquisitions import (
    RasterScanStitchedAcquisitionSpec, LineCameraStitchedAcquisitionSpec,
    RasterScanStitchedAcquisition, LineCameraStitchedAcquisition

)

"""
Expected limitations:
- won't work when tiff tile size >= strip width
"""

sig = [
#         strip         lines         positions     pixel_size  prev_row  flip_line
    int64(int16[:,:,:],  int16[:,:,:],  float64[:,:], float64,    int64,    boolean),
    int64(uint16[:,:,:], uint16[:,:,:], float64[:,:], float64,    int64,    boolean)
]
@njit(sig, parallel=True, fastmath=True, cache=True)
def _line_placement_kernel(strip: np.ndarray,     # dim order (web, scan, chan)
                           lines: np.ndarray,     # dim order (web, scan, chan)
                           positions: np.ndarray, # dim order (web, scan)
                           pixel_size: float,
                           prev_row: int,
                           flip_line: bool) -> int:
    n_height, n_width, n_chan = strip.shape

    for idx in prange(lines.shape[0]):
        # compute where this line should go
        if idx > 0:
            prev_row = int(round(positions[idx-1, 0] / pixel_size))
        cur_row   = int(round(positions[idx, 0] / pixel_size))
        cur_shift = int(round(positions[idx, 1] / pixel_size))

        # skip if the line is out of web dim bounds
        if (cur_row < 0) or (cur_row >= n_height):
            continue

        # for each source-pixel index j, compute its destination-index and copy
        for j in range(n_width):
            # pick the source column (mirrored if requested)
            src_j = n_width - 1 - j if flip_line else j
            dst_j = j + cur_shift

            if dst_j < 0 or dst_j >= n_width:
                continue        # skip if out of bounds laterally

            strip[cur_row, dst_j, :] = lines[idx, src_j, :]

        # Nearest neighbor interpolation
        if abs(cur_row - prev_row) == 2:
            for j in range(n_width):
                strip[(cur_row + prev_row)//2, j, :] = strip[cur_row, j, :]

    # returns the last placed row
    return int(round(positions[lines.shape[0]-1, 0] / pixel_size))


class StripProcessor(Processor[RasterFrameProcessor]):
    """  """
    def __init__(self, upstream: RasterFrameProcessor):
        super().__init__(upstream)

        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self._acq: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        self._system_config = self._acq.system_config
        self._data_range = upstream.data_range
        self._positioner = self._acq.positioner 

        # positions are stored in order (web, scan)
        prev_position = (self._positioner.web_min(0), self._positioner.scan_center(0))
        self._prev_position = np.array(prev_position, dtype=np.float64)

        self._strip_shape = ( # strips are assembled in dim order: (web, scan, chan)
            self._acq.final_shape[1], # final_shape[1]: pixels in web dimension
            self._spec.pixels_per_line,
            self._acq.final_shape[2]
        )
        if self._acq.data_acquisition_device.data_range.recommended_dtype in {np.uint8, np.uint16}:
            dt = np.uint16
        else:
            dt = np.int16
        self.init_product_pool(n=4, shape=self._strip_shape, dtype=dt)
        
        self._strip_idx = 0
        self._prev_row = -1 # increments as _line_placement_kernel is called

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
    
    def run(self):
        try:
            strip = self._get_free_product()
            strip.data[...] = 0 # TODO, check performance impact of this

            while True:
                with self._receive_product() as frame:
                    positions = np.array(frame.positions)

                    strip_center_min = np.array([
                        self._positioner.web_min(self._strip_idx),        
                        self._positioner.scan_center(self._strip_idx)                        
                    ])
                    strip_positions = positions - strip_center_min[np.newaxis,:] # (web, scan)

                    # Check whether this frame includes move to next strip
                    translation = self._positioner.scan_center(self._strip_idx + 1) - self._positioner.scan_center(self._strip_idx)
                    next_strip = round(strip_positions[-1, 1] / translation) > 0

                    # blank out lines moving in opposite direction
                    direction = 1 if (self._strip_idx % 2 == 0) else -1
                    b = np.diff(strip_positions[:,0]) * direction < 0 # bool array of lines with opposite web velo sign
                    b = np.insert(b, 0, False)
                    strip_positions[b] = 0

                    self._prev_row = _line_placement_kernel( # add lines to strip
                        strip=strip.data, 
                        lines=frame.data,
                        positions=strip_positions,
                        pixel_size=self._spec.pixel_size,
                        prev_row=self._prev_row,
                        flip_line=False
                    )

                    if next_strip:
                        # Moving to next strip, publish
                        print(f"Publishing strip {self._strip_idx} with size: {strip.data.shape}")
                        self._publish(strip)
                        self._strip_idx += 1

                        strip = self._get_free_product()
                        strip.data[...] = 0 # TODO, remove this if we can ensure complete overwrite of previous data

                        # add any leftover lines to new strip
                        strip_center_min = np.array([
                            self._positioner.web_min(self._strip_idx),
                            self._positioner.scan_center(self._strip_idx)                            
                        ])
                        strip_positions = positions - strip_center_min[np.newaxis,:]
                        
                        self._prev_row = _line_placement_kernel( 
                            strip=strip.data, 
                            lines=frame.data,
                            positions=strip_positions,
                            pixel_size=self._spec.pixel_size,
                            prev_row=self._prev_row,
                            flip_line=False
                        )
                        
        except EndOfStream:
            # publish final strip (completed or not)
            print(f"Publishing FINAL strip {self._strip_idx} with size: {strip.data.shape}")
            self._publish(strip)

            # send shutdown sentinel
            self._publish(None)
    
    @property
    def data_range(self) -> units.IntRange:
        return self._data_range


def _linear_blend(strip_a: np.ndarray, strip_b: np.ndarray, overlap_px: int):
    """Blend the right `overlap_px` columns of `strip_a`
       with the left `overlap_px` columns of `strip_b`."""
    if overlap_px > 0:
        w = overlap_px
        alpha = np.linspace(0, 1, w, dtype=np.float32)[np.newaxis, :, np.newaxis]  # (1,w,1)

        strip_a_end = strip_a[:, -w:, :].astype(np.float32)
        strip_b_start = strip_b[:, :w, :].astype(np.float32)

        blended = ((1 - alpha) * strip_a_end + alpha * strip_b_start).astype(strip_a.dtype)

        strip_a[:, -w:, :] = blended
        strip_b[:, :w,  :] = blended


class StripStitcher(Processor[StripProcessor]):
    def __init__(self, upstream: StripProcessor):
        super().__init__(upstream)
        self._data_range = upstream.data_range

        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self._overlap_pixels = round(self._spec.strip_overlap * self._spec.pixels_per_line)
        self._prev_strip = None

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        try:
            while True:
                with self._receive_product() as strip_product:

                    if self._prev_strip is None:
                        # we need one more strip to start blending
                        self._prev_strip = strip_product
                        continue

                    _linear_blend(self._prev_strip.data,
                                strip_product.data,
                                self._overlap_pixels)
                    
                    with self._prev_strip:
                        self._publish(self._prev_strip)
                        # publish increments product._remaining
                        # exiting context manager decrements product._remaining
                        # for 1 subscriber, net = 0; product not returned to pool

                    self._prev_strip = strip_product

        finally:
            if self._prev_strip is not None:
                self._publish(self._prev_strip)
            self._publish(None)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range


# @njit(parallel=True, cache=True)
# def _rotate90_inplace(a: np.ndarray, cw: bool = True):
#     """
#     In-place clockwise 90° rotation of a square array with channels.
#     a.shape == (n, n, c)
#     """
#     n, _, c = a.shape
#     # only iterate over the “top-left quadrant” of spatial indices:
#     for i in prange(n//2):
#         for j in range(n//2):
#             for k in range(c):
#                 tmp = a[i, j, k]
#                 if cw:
#                     # clockwise cycle
#                     a[i, j, k]               = a[n-1-j, i, k]
#                     a[n-1-j, i, k]           = a[n-1-i, n-1-j, k]
#                     a[n-1-i, n-1-j, k]       = a[j, n-1-i, k]
#                     a[j, n-1-i, k]           = tmp
#                 else:
#                     # counter-clockwise cycle
#                     a[i, j, k]               = a[j, n-1-i, k]
#                     a[j, n-1-i, k]           = a[n-1-i, n-1-j, k]
#                     a[n-1-i, n-1-j, k]       = a[n-1-j, i, k]
#                     a[n-1-j, i, k]           = tmp


@njit(parallel=True)
def _transpose_inplace(a: np.ndarray):
    """
    In-place transpose of the spatial dims of a (n, n, c) array.
    Channels are left in place.
    """
    n, _, c = a.shape
    for i in prange(n):
        for j in range(i+1, n):
            for k in range(c):
                tmp         = a[i, j, k]
                a[i, j, k]  = a[j, i, k]
                a[j, i, k]  = tmp


class TileProduct(Product):
    """Simplified Product class for a tiff image 'tile'"""
    __slots__ = ("data", "coords")
    def __init__(self, 
                 pool, 
                 data: np.ndarray,
                 tile_coords: Optional[tuple] = None):
        super().__init__(pool)
        self.data = data
        self.coords = tile_coords


class TileBuilder(Processor[StripStitcher]):
    """Parcels up tiles to send to file logger."""
    def __init__(self, upstream: StripStitcher, tile_shape=(512,512)):
        super().__init__(upstream)
        self._acq: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec

        if tile_shape[0] != tile_shape[1]:
            raise ValueError("Tile shape must be square")

        self._data_range = upstream.data_range
        self._tile_shape = tile_shape
        self._n_channels = self._acq.final_shape[2]

        self.init_product_pool(n=4, shape=(*self._tile_shape, self._n_channels), dtype=np.int16)

        self._tiles_web  = math.ceil(self._acq.final_shape[1] / tile_shape[0]) # tiles along the web dimension (strip long axis)
        self._tiles_scan = math.ceil(self._acq.final_shape[0] / tile_shape[0]) # tiles along the scan dimension (strip short axis)
        
        self._leftovers: Optional[np.ndarray] = None

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
        
    def run(self):
        tile_idx = 0
        strip_idx = 0

        tile_shape = self._tile_shape
        effective_pixels_per_line = int(
            self._spec.pixels_per_line * (1-self._spec.strip_overlap)
        )

        try:
            while True:
                with self._receive_product() as strip:

                    while True:
                        t_w = tile_idx %  self._tiles_web # tile coordinate web dim
                        t_s = tile_idx // self._tiles_web # tile coordinate scan dim

                        if t_s >= self._tiles_scan: 
                            raise EndOfStream
                        
                        w = t_w * tile_shape[0]   # web dim global pixel coordinate
                        s = t_s * tile_shape[1]   # scan dim global pixel coordinate
                        scan_offset = strip_idx * effective_pixels_per_line 

                        s_o = s - scan_offset   # scan pixel coordinate relative to the current strip
                        
                        if (s_o + tile_shape[1]) > strip.data.shape[1]:
                            # If the next tile exceeds current strip dimensions, 
                            # store leftovers and break to move on to next strip
                            self._leftovers = strip.data[:, s_o:, :].copy()
                            strip_idx += 1
                            break

                        tile = self.get_free_product()
                        tile.coords = (t_s, t_w)
                        tile.data[...] = 0      # clear old tile data                   

                        if s_o >= 0:
                            # tile fully within current strip, copy into tile object
                            data = strip.data[
                                w : min(w + self._tile_shape[0], strip.data.shape[0]),
                                s_o : min(s_o + self._tile_shape[1], strip.data.shape[1]),
                                :
                            ]
                            tile.data[:data.shape[0], :data.shape[1], :] = data
                        else:   # tile stradles previous strip and the current strip
                            # copy data from leftovers
                            if self._leftovers is None:
                                raise RuntimeError("Leftovers not initialized")
                            data1 = self._leftovers[
                                w : min(w + self._tile_shape[0], self._leftovers.shape[0]),
                                :, :
                            ]
                            tile.data[:data1.shape[0], :data1.shape[1], :] = data1

                            # copy data from current strip
                            data2 = strip.data[
                                w : min(w + self._tile_shape[0], self._leftovers.shape[0]),
                                :(self._tile_shape[1] + s_o),
                                :
                            ]
                            tile.data[:data2.shape[0], -data2.shape[1]:, :] = data2
                        
                        if self._acq.system_config.fast_raster_scanner['axis'] == "x":
                            _transpose_inplace(tile.data)
                            #_rotate90_inplace(tile.data, False)

                        #print(f"Publishing tile {tile_idx}: ({t_s},{t_w}), shape: {tile.data.shape}, dtype: {tile.data.dtype}")
                        self._publish(tile)
                        tile_idx += 1
        except EndOfStream:
            self._publish(None)

        finally:
            self._publish(None)

    def init_product_pool(self, n, shape, dtype):
        for _ in range(n):
            aq_buf = TileProduct(
                pool=self._product_pool,
                data=np.empty(shape, dtype) 
            )
            self._product_pool.put(aq_buf)

    def get_free_product(self, timeout: units.Time | None = None) -> TileProduct:
        return self._product_pool.get(timeout=timeout)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range

