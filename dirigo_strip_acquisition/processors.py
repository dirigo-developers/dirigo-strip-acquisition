import math, time
from typing import Optional

from numba import njit, prange, int16, uint8, uint16, float32, float64, int64, boolean
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from dirigo import units, io
from dirigo.sw_interfaces.worker import Product, EndOfStream, Worker
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
    #int64(uint16[:,:,:], uint16[:,:,:], float64[:,:], float64,    int64,    boolean)
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


class StripProcessor(Processor[RasterFrameProcessor]): # TODO this can also be used with a LineCamera Processor (not limited to raster)
    """Receives position-encoded line data and places lines into strip."""
    def __init__(self, upstream: RasterFrameProcessor):
        super().__init__(upstream)

        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self._acquisition: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        if isinstance(self._spec, RasterScanStitchedAcquisitionSpec):
            self._scan_axis_label = self._acquisition.system_config.fast_raster_scanner['axis']
        else:
            self._scan_axis_label = self._acquisition.system_config.line_camera['axis']
        self._system_config = self._acquisition.system_config
        self._data_range = upstream.data_range
        self._positioner = self._acquisition.positioner

        # positions are stored in order (web, scan)
        prev_position = (self._positioner.web_min(0), self._positioner.scan_center(0))
        self._prev_position = np.array(prev_position, dtype=np.float64)

        # _acquisition.final_shape: (z, scan, web, channel)
        self._strip_shape = ( # WARNING strips are assembled in dim order: (web, scan, chan)
            self._acquisition.final_shape[2],
            self._spec.pixels_per_line,
            self._acquisition.final_shape[3]
        )

        self._init_product_pool(n=4, shape=self._strip_shape, dtype=np.int16)
        
        self._prev_row = -1 # increments as _line_placement_kernel is called

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
    
    def run(self):
        scan_transl = self._positioner.scan_center(1) - self._positioner.scan_center(0)  
        try:
            strip = self._get_free_product()
            strip.data[...] = 0         # TODO, check performance impact of this

            for z_index in range(self._spec.z_steps):
                for strip_index in range(self._positioner.n_strips):
                    
                    while True:
                        # Collect a number of frames and put lines into strip;
                        # sense movement to the next strip -> break
                        with self._receive_product() as frame:
                            if frame.positions is None:
                                raise RuntimeError("Incoming frame missing encoder positions")
                            if self._scan_axis_label == "x":
                                positions = np.array(frame.positions[:,::-1]) # flip so order is (web[y], scan[x])
                            else:
                                positions = np.array(frame.positions)

                            strip_center_min = np.array([[
                                self._positioner.web_min(strip_index),        
                                self._positioner.scan_center(strip_index)                        
                            ]])
                            strip_positions = positions - strip_center_min # (web, scan)

                            # Check if frame includes move to next z step or next strip
                            next_z = round(strip_positions[-1, 1] / scan_transl) < 0    # TODO, this would fail for 1-strip wide acquisitions
                            next_strip = round(strip_positions[-1, 1] / scan_transl) > 0

                            if next_z or next_strip:
                                # we know there is next strip data in this product,
                                # blank out lines moving in opposite direction
                                direction = 1 if (strip_index % 2 == 0) else -1
                                b = np.diff(strip_positions[:,0]) * direction < 0 # bool array of lines with opposite web velo sign
                                b = np.insert(b, 0, False)
                                strip_positions[b] = 0

                            # add lines to current strip
                            self._prev_row = _line_placement_kernel( 
                                strip=strip.data, 
                                lines=frame.data,
                                positions=strip_positions,
                                pixel_size=self._spec.pixel_size,
                                prev_row=self._prev_row,
                                flip_line=False
                            )

                            if next_z or next_strip:
                                # Moving to next strip, publish
                                if next_z:
                                    # Next z movement may take a number of frames to reach XY starting point
                                    scan_pos_relative_strip_0 = positions[-1, 1] - self._positioner.scan_center(0)
                                    if round(scan_pos_relative_strip_0 / scan_transl) != 0:
                                        # haven't made it back to XY starting point
                                        continue
                                    else:
                                        # or we have made it back to XY starting point
                                        new_strip_index = 0 # starting over from strip 0
                                else:
                                    # next strip
                                    new_strip_index = strip_index + 1

                                # publish the completed strip
                                print(f"Publishing strip {z_index, strip_index} with size: {strip.data.shape}")
                                strip.indices = (z_index, strip_index)
                                self._publish(strip)

                                strip = self._get_free_product()
                                strip.data[...] = 0 # TODO, remove this if we can ensure complete overwrite of previous data

                                # add any leftover lines to new strip
                                strip_center_min = np.array([
                                    self._positioner.web_min(new_strip_index),
                                    self._positioner.scan_center(new_strip_index)                            
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
                                break

        except EndOfStream:
            # publish final strip (completed or not)
            print(f"Publishing final strip {z_index, strip_index} with size: {strip.data.shape}")
            strip.indices = (z_index, strip_index)
            self._publish(strip)

        finally:
            # send shutdown sentinel
            self._publish(None)
    
    @property
    def data_range(self) -> units.IntRange:
        return self._data_range
    


class StripStitcher(Processor[StripProcessor]):
    """
    Blends edges of consecutive strips. 
    
    Note that this works in-place on the strip data (not copied).
    """
    def __init__(self, upstream: StripProcessor):
        super().__init__(upstream)
        self._data_range = upstream.data_range

        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self._n_strips = upstream._positioner.n_strips
        self._strip_height = upstream.product_shape[0]
        self._strip_dtype = upstream.product_dtype
        self._overlap_pixels = round(self._spec.strip_overlap * self._spec.pixels_per_line)

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def run(self):
        w = self._overlap_pixels
        prev_correction = 1
        try:
            while True:
                with self._receive_product() as strip:
                    if strip.indices is None:
                        raise RuntimeError("Strip products must include indices.")
                    strip.hold_once() # product won't be released until it is opened again

                    if strip.indices[1] == 0:
                        # we need one more strip to start blending
                        prev_strip = strip
                        continue

                    a, b = prev_strip.data, strip.data

                    # Field flattening
                    a_end   = np.average(a[:, -w:-1, :], axis=(1,2))
                    b_start = np.average(b[:, 1:w, :], axis=(1,2))

                    seam_avg = (a_end + b_start) / 2

                    a_correction = seam_avg / a_end
                    a_correction = a_correction[~np.isnan(a_correction) & (a_end > 60)] # TODO: set cutoff programatically
                    a_correction = np.median(a_correction)

                    b_correction = seam_avg / b_start
                    b_correction = b_correction[~np.isnan(b_correction) & (b_start > 60)]
                    b_correction = np.median(b_correction[~np.isnan(b_correction)])

                    correction = np.linspace(prev_correction, a_correction, a.shape[1])
                    a[...] = (a * correction[None,:,None]).astype(np.int16)

                    prev_correction = b_correction

                    # Blend the edges
                    if w > 0:
                        alpha = np.linspace(0, 1, w, dtype=np.float32)[np.newaxis, :, np.newaxis]  # (1,w,1)
                        alpha = np.clip(2 * alpha, a_min=0, a_max=1) # blend only the inner part of overlap area

                        strip_a_end     = a[:, -w:, :].astype(np.float32)
                        strip_b_start   = b[:, :w,  :].astype(np.float32) * b_correction

                        blended = ((1-alpha)*strip_a_end + alpha*strip_b_start).astype(np.int16)

                        a[:, -w:, :] = blended  # only correct A since B (edge) will not be used for tiles
                    
                    with prev_strip: # type: ignore
                        # this is the 2nd time this Product is entered, so it will release after this
                        print(f"Republishing strip {prev_strip.indices}")
                        self._publish(prev_strip)
                        # publish increments product._remaining
                        # exiting context manager decrements product._remaining
                        # for 1 subscriber, net = 0; product not returned to pool

                    if strip.indices[1] == self._n_strips - 1:
                        # on last strip of the z opt. section, publish last strip
                        with strip:     # make sure to release the product
                            correction = np.linspace(prev_correction, 1, b.shape[1])
                            b[...] = (b * correction[None,:,None]).astype(np.int16)

                            print(f"Republishing strip {strip.indices}")
                            self._publish(strip) 
                        prev_correction = 1
                        prev_strip = None
                    else:
                        prev_strip = strip

        except EndOfStream:
            if prev_strip:
                with prev_strip: # type: ignore
                    print(f"Republishing strip {prev_strip.indices}")
                    self._publish(prev_strip)

            self._publish(None)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range


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
        super().__init__(pool, data)
        self.data = data
        self.coords = tile_coords


class TileBuilder(Processor[StripStitcher]):
    """Parcels up tiles to send to file logger."""
    Product = TileProduct

    def __init__(self, upstream: StripStitcher, tile_shape=(512,512)):
        super().__init__(upstream)
        self._acquisition: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec

        if tile_shape[0] != tile_shape[1]:
            raise ValueError("Tile shape must be square")

        self._data_range = upstream.data_range
        self._tile_shape = tile_shape
        self._n_channels = self._acquisition.final_shape[-1] # (z, scan, web, chan)

        self._init_product_pool(n=4, shape=(*self._tile_shape, self._n_channels), dtype=np.int16)

        self._n_strips = self._acquisition.positioner.n_strips
        self._tiles_web  = math.ceil(self._acquisition.final_shape[-2] / tile_shape[0]) # tiles along the web dimension (strip long axis)
        self._tiles_scan = math.ceil(self._acquisition.final_shape[-3] / tile_shape[0]) # tiles along the scan dimension (strip short axis)
        self._tiles_image = self._tiles_web * self._tiles_scan

        self._leftovers: Optional[np.ndarray] = None

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
        
    def run(self):
        tiles_scan = self._tiles_scan
        tiles_web  = self._tiles_web
        tile_idx = 0   # tile XY coordinate
        t_z = 0        # tile Z coordinate

        tile_shape = self._tile_shape
        effective_pixels_per_line = int(
            self._spec.pixels_per_line * (1-self._spec.strip_overlap)
        )

        try:
            while True: # Looping in strips
                with self._receive_product() as strip:
                    
                    if strip.indices is None:
                        raise RuntimeError("Strip must include indices")

                    while True: # Looping in tiles scan, tiles web
                        t_s = tile_idx // tiles_web   # tile coordinate scan dim (parallel to acquired line)
                        t_w = tile_idx %  tiles_web   # tile coordinate web dim

                        if t_s >= tiles_scan:
                            # If t_s exceeds expected number tiles in scan dim, reset and break
                            self._leftovers = None
                            tile_idx = 0
                            t_z += 1
                            break
                        
                        p_s = t_s * tile_shape[0]   # scan dim global pixel coordinate
                        p_w = t_w * tile_shape[1]   # web dim global pixel coordinate
                        scan_offset = strip.indices[1] * effective_pixels_per_line 

                        p_so = p_s - scan_offset   # scan pixel coordinate relative to the current strip
                        
                        # If start of next tile will exceed current strip, store leftovers
                        if (p_so + tile_shape[0]) > strip.data.shape[1]:
                            self._leftovers = strip.data[:, p_so:, :].copy()
                            # and not last strip of z level:
                            if strip.indices[1] < (self._n_strips-1):
                                break # go on to recieve a new strip to complete the tile

                        tile = self.get_free_product()
                        tile.coords = (t_z, t_s, t_w)
                        tile.data[...] = 0      # clear old tile data                   

                        if p_so >= 0:    # Situation 1: tile in current strip, copy into tile object
                            data = strip.data[
                                p_w : min(p_w + self._tile_shape[1], strip.data.shape[0]),
                                p_so : min(p_so + self._tile_shape[0], strip.data.shape[1]),
                                :
                            ]
                            tile.data[:data.shape[0], :data.shape[1], :] = data
                        else:           # Situation 2: tile stradles previous and current strips
                            # copy data from leftovers
                            if self._leftovers is None:
                                raise RuntimeError("Leftovers not initialized")
                            data1 = self._leftovers[
                                p_w : min(p_w + self._tile_shape[0], self._leftovers.shape[0]),
                                :, :
                            ]
                            tile.data[:data1.shape[0], :data1.shape[1], :] = data1

                            # copy data from current strip
                            data2 = strip.data[
                                p_w : min(p_w + self._tile_shape[0], self._leftovers.shape[0]),
                                :(self._tile_shape[1] + p_so),
                                :
                            ]
                            tile.data[:data2.shape[0], -data2.shape[1]:, :] = data2
                        
                        _transpose_inplace(tile.data) # go from strips in dimensions (web, scan, chan) to tiles in (scan, web, chan)

                        print(f"Publishing tile {t_z,t_s,t_w}")
                        self._publish(tile)
                        tile_idx += 1

        except EndOfStream:
            self._publish(None)

    def get_free_product(self, timeout: units.Time | None = None) -> TileProduct:
        return self._product_pool.get(timeout=timeout)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range



sigs = [
     uint8[:,:,:]( uint8[:,:,:], int64),
    uint16[:,:,:](uint16[:,:,:], int64),
     int16[:,:,:]( int16[:,:,:], int64)
]
@njit(sigs, parallel=True, fastmath=True, cache=True)
def downsample_kernel(tile: np.ndarray, f: int) -> np.ndarray:
    h, w, n_channels = tile.shape
    ds_h, ds_w = h//f, w//f
    area = f * f

    downsampled_tile = np.empty((ds_h, ds_w, n_channels), tile.dtype)

    for i in prange(ds_h):
        for j in range(ds_w):
            for k in range(n_channels):

                tmp = np.int32(0)
                for di in range(f):
                    for dj in range(f):

                        tmp += tile[i*f + di, j*f + dj, k]

                downsampled_tile[i, j, k] = tmp // area
    
    return downsampled_tile



class StitchedPreview(Processor):
    """
    Creates a downsampled preview of stitched image from tiles.
    """
    def __init__(self, 
                 upstream: TileBuilder, 
                 downsample: int = 16,
                 **kwargs):
        super().__init__(upstream, **kwargs)
        self._hold = True # effectively holds off starting run loop until a subscriber is added
        self._acquisition: RasterScanStitchedAcquisition
        self._data_range = upstream.data_range
        self._downsample = downsample

        # Product is a downsampled version of the full stitched image       
        self._downsampled_tile_length = upstream.product_shape[0] // downsample
        # _acquisition.final_shape (z, scan, web, channel)
        self._tiles_scan = math.ceil(
            self._acquisition.final_shape[1] / upstream.product_shape[0]
        )
        self._tiles_web  = math.ceil(
            self._acquisition.final_shape[2] / upstream.product_shape[0]
        )
        self._z_levels = self._acquisition.final_shape[0]
        preview_shape = (self._tiles_scan * self._downsampled_tile_length, 
                         self._tiles_web  * self._downsampled_tile_length,
                         self._acquisition.final_shape[3]) # the preview does not have a Z dimension

        self._init_product_pool(
            n       = 1,
            shape   = preview_shape, 
            dtype   = self.data_range.recommended_dtype,
        )

    def _receive_product(self) -> TileProduct:
        return super()._receive_product() # type: ignore
    
    def run(self):
        while self._hold:
            # trick to avoid deadlock: getting stuck at _receive_product()
            time.sleep(0.05)
        try:
            for tz in range(self._z_levels):
                for ts in range(self._tiles_scan):
                    # Get the preview product from the pool
                    preview = self._get_free_product()

                    # work through all the tiles along the web direction, then publish
                    for tw in range(self._tiles_web):

                        with self._receive_product() as tile:
                            #print(f"Tile's indices {tile.coords}, Our indices {tz,ts,tw}")
                            i0 = ts * self._downsampled_tile_length
                            j0 = tw * self._downsampled_tile_length
                            i1 = i0 + self._downsampled_tile_length
                            j1 = j0 + self._downsampled_tile_length
                            # downsample and place in array
                            preview.data[i0:i1, j0:j1, :] = \
                                downsample_kernel(tile.data, self._downsample)

                    self._publish(preview)
                    # print(f"Published preview(s) on scan row {ts} of {self._tiles_scan}")

            self._publish(None) # forward sentinel None

        except EndOfStream:
            self._publish(None) # forward sentinel None

    def add_subscriber(self, subscriber: Worker):
        """Adds the subscriber and publishes a blank product."""
        super().add_subscriber(subscriber)
        self._publish(self._get_free_product())
        self._hold = False

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range

