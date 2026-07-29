import math, time
from typing import Optional

from numba import njit, prange, int16, uint8, uint16, float32, float64, int64, boolean, void, types
import numpy as np
from numpy.polynomial.polynomial import Polynomial

from dirigo import units, io
from dirigo.sw_interfaces.worker import Product, EndOfStream, Worker
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.plugins.acquisitions import LineAcquisitionRuntimeInfo, CameraAcquisitionRuntimeInfo
from dirigo.plugins.processors import RasterFrameProcessor

from dirigo_strip_acquisition.acquisitions import (
    RasterScanStitchedAcquisitionSpec, LineCameraStitchedAcquisitionSpec,
    RasterScanStitchedAcquisition, LineCameraStitchedAcquisition,
    RectangularFieldStagePositionHelper,
)


uint8_3d_readonly  = types.Array(types.uint8, 3, 'C', readonly=True)
int16_3d_readonly = types.Array(types.int16, 3, "C", readonly=True)
float64_1d_readonly = types.Array(types.float64, 1, "C", readonly=True)
int64_1d_readonly = types.Array(types.int64, 1, "C", readonly=True)

sig_resample = [
    void(
        types.int16[:, :, :],       # strip
        types.int64,                # strip_index
        types.int64,                # nlines_copied
        int16_3d_readonly,          # lines
        float64_1d_readonly,        # source_web_px, sorted ascending
        float64_1d_readonly,        # source_scan_px, same order
        int64_1d_readonly           # channel_row_offsets
    )
]


@njit(sig_resample, parallel=True, fastmath=True, cache=True)
def _resample_strip_nearest_kernel(
    strip: np.ndarray,                 # (web, scan, chan)
    strip_index: int,
    nlines_copied: int,
    lines: np.ndarray,                 # (line_idx, scan, chan), sorted by web position
    source_web_px: np.ndarray,         # line_idx -> web position in pixels
    source_scan_px: np.ndarray,        # line_idx -> scan shift in pixels
    channel_row_offsets: np.ndarray):  # chan -> row offset in pixels

    n_height, n_width, n_chan = strip.shape

    # Compute interpolation source indices (this can't be done parallel)
    src_indices = np.zeros((n_height,), np.int64)

    if strip_index % 2 == 0:
        # Forward strips
        i = 0
        for trgt_row in range(n_height):
            while abs(source_web_px[i+1]-trgt_row) < abs(source_web_px[i]-trgt_row):
                i += 1
            src_indices[trgt_row] = i
    else:
        i = 0
        for trgt_row in range(n_height-1, -1, -1): # fill backwards
            while abs(source_web_px[i+1]-trgt_row) < abs(source_web_px[i]-trgt_row):
                i += 1
            src_indices[trgt_row] = i

    # Perform NN interpolation
    for dst_row in prange(n_height):
        for ch in range(n_chan):
            target_web_px = dst_row - channel_row_offsets[ch]

            src_idx = src_indices[target_web_px]

            if src_idx < 0:
                continue

            scan_shift = int(round(source_scan_px[src_idx]))

            for dst_j in range(n_width):
                src_j = dst_j - scan_shift

                if 0 <= src_j < n_width:
                    strip[dst_row, dst_j, ch] = lines[src_idx, src_j, ch]


class StripProcessor(Processor[RasterFrameProcessor]): # TODO this can also be used with a LineCamera Processor (not limited to raster)
    """Receives position-encoded line data and places lines into strip."""
    def __init__(self, 
                 upstream: RasterFrameProcessor,
                 channel_shear: np.ndarray | None = None):
        
        super().__init__(upstream, name="StripProcessor")
        
        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self._acquisition: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        if isinstance(self._spec, RasterScanStitchedAcquisitionSpec):
            self._scan_axis_label = self._acquisition.system_config.fast_raster_scanner['axis']
            axis_error = self._acquisition.runtime_info.stage_scanner_angle
        else:
            self._scan_axis_label = self._acquisition.system_config.line_camera['axis']
            axis_error = self._acquisition.runtime_info.stage_camera_angle

        self._system_config = self._acquisition.system_config
        self._data_range = upstream.data_range
        self._positioner = RectangularFieldStagePositionHelper(
            scan_axis   = self._scan_axis_label,
            axis_error  = axis_error,
            line_width  = self._spec.line_width, # TODO, remove line_width since it's already in spec
            spec        = self._spec
        )

        if self._scan_axis_label == "x":
            n_pixels_web = round(self._spec.y_range.range / self._spec.pixel_size)
        else:
            n_pixels_web = round(self._spec.x_range.range / self._spec.pixel_size)

        n_channels = self._acquisition.runtime_info.n_channels
        self._strip_shape = ( # strips are assembled in dim order: (web, scan, chan)
            n_pixels_web,
            self._spec.pixels_per_line,
            n_channels
        )
        if channel_shear is None:
            self._forward_shear = np.zeros(shape=(n_channels,), dtype=np.int64)
            self._reverse_shear = np.zeros(shape=(n_channels,), dtype=np.int64)
        else:
            if not isinstance(channel_shear, np.ndarray) or channel_shear.shape != (2, n_channels):
                raise ValueError("`channel_shear` must be a numpy array")
            self._forward_shear = channel_shear[0,:].astype(np.int64)
            self._reverse_shear = channel_shear[1,:].astype(np.int64)

        self._init_product_pool(
            n =     2, 
            shape = self._strip_shape, 
            dtype = np.int16
        )

        # Make large buffer to contain incoming line data
        # TODO allocate smarter, currently it's just 2X the strip web length
        buffer_shape = (2*self._strip_shape[0], self._strip_shape[1], self._strip_shape[2])
        self._buffer = np.zeros(shape=buffer_shape, dtype=np.int16)
        self._positions = np.zeros(shape=(2*self._strip_shape[0], 2), dtype=np.float64)
        
    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
    
    def _work(self):
        scan_trans_thresh = 0.25 * self._spec.line_width
        
        try:
            for z_index in range(self._spec.z_steps):
                for strip_index in range(self._positioner.n_strips):
                    self._nlines_copied = 0

                    web_min = self._positioner.web_min(strip_index)
                    web_max = self._positioner.web_max(strip_index)
                    web_length = web_max - web_min
                    scan_center = self._positioner.scan_center(strip_index) 
                    strip_center_min = np.array([[web_min, scan_center]])
                
                    while True:
                        with self._receive_product() as frame:
                            
                            if self._scan_axis_label == "x":
                                positions = np.array(frame.positions[:,::-1]) # flip so order is (web[y], scan[x])
                            else:
                                positions = np.array(frame.positions)
                            strip_positions = positions - strip_center_min # relative positions

                            web_valid = (strip_positions[:, 0] >= 0) & (strip_positions[:, 0] <= web_length)
                            # TODO, would ascending/monotonic be better condition for 'web valid'?

                            if self._positioner.n_strips > 1:
                                scan_valid = (strip_positions[:, 1] > -scan_trans_thresh) & (strip_positions[:, 1] < scan_trans_thresh)
                            else:
                                scan_valid = np.ones(shape=(len(positions),), dtype=np.bool_)

                            valid = web_valid & scan_valid
                            nvalid = int(sum(valid))

                            nlc = self._nlines_copied # for brevity below
                            if nlc+nvalid > self._buffer.shape[0]:
                                print("problem")
                            self._buffer[nlc:(nlc+nvalid),:,:] = frame.data[valid]
                            self._positions[nlc:(nlc+nvalid),:] = strip_positions[valid]
                            self._nlines_copied += nvalid

                            # TODO store carry over lines for next strip

                            if (self._nlines_copied > 0) and (not valid[-1]):
                                break # break out of while loop

                    self._flush_strip(z_index, strip_index)

        except EndOfStream:
            self._flush_strip(z_index, strip_index)

        finally:
            self._publish(None)

    def _flush_strip(self, z_index: int, strip_index: int):

        source_web_px = self._positions[:, 0] / self._spec.pixel_size
        source_scan_px = self._positions[:, 1] / self._spec.pixel_size

        strip = self._get_free_product()
        strip.data[...] = 0 # TODO, is this needed with new resampling strategy?

        _resample_strip_nearest_kernel(
            strip               = strip.data, # final strip data
            strip_index         = strip_index,
            nlines_copied       = self._nlines_copied,
            lines               = self._buffer,
            source_web_px       = source_web_px,
            source_scan_px      = source_scan_px,
            channel_row_offsets = self.channel_shear(strip_index),
        )

        strip.indices = (z_index, strip_index)
        print(f"Publishing strip with indices {strip.indices} (n lines copied: {self._nlines_copied})")
        self._publish(strip)
    
    @property
    def data_range(self) -> units.IntRange:
        return self._data_range
    
    def channel_shear(self, strip_index: int) -> np.ndarray:
        """Given the strip index, selects the appropriate channel shearing vector"""
        if (strip_index % 2) == 0:
            # Forward / even strips
            return self._forward_shear
        else:
            # Reverse / odd strips
            return self._reverse_shear
    


class StripStitcher(Processor[StripProcessor]):
    """
    Blends edges of consecutive strips. 
    """
    INTENSITY_THRESH = 200

    def __init__(self, upstream: StripProcessor):
        super().__init__(upstream, name="StripStitcher")
        self._data_range = upstream.data_range

        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        self.n_strips = upstream._positioner.n_strips

        self._overlap_pixels = round(self._spec.strip_overlap * self._spec.pixels_per_line)

        self._init_product_pool(
            n =     2, 
            shape = upstream._strip_shape, 
            dtype = np.int16
        )

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore

    def _work(self):
        w = self._overlap_pixels
        prev_correction = 1
        try:
            stitched_strip = self._get_free_product()
            while True:
                with self._receive_product() as in_strip:
                    if in_strip.indices is None: # (z, strip)
                        raise RuntimeError("Strip products must include indices.")

                    if in_strip.indices[1] == 0:
                        stitched_strip.data[...] = in_strip.data
                        stitched_strip.indices = tuple(in_strip.indices)
                        
                        if self.n_strips == 1:
                            self._publish(stitched_strip)

                        continue

                    a, b = stitched_strip.data, in_strip.data

                    # Field flattening
                    a_end   = np.average(a[:, -w:-1, :], axis=1, keepdims=True) # skip the very last pixel b/c can be 0
                    a_end[a_end == 0] = 1e-9
                    b_start = np.average(b[:, 1:w, :], axis=1, keepdims=True)
                    b_start[b_start == 0] = 1e-9

                    seam_avg = (a_end + b_start) / 2

                    ac = seam_avg / a_end
                    a_correction = [
                        np.median(ac[...,c][a_end[...,c] > self.INTENSITY_THRESH], axis=0)
                            for c in range(a.shape[-1])
                    ]

                    bc = seam_avg / b_start
                    b_correction = [
                        np.median(bc[...,c][b_start[...,c] > self.INTENSITY_THRESH], axis=0)
                            for c in range(a.shape[-1])
                    ]

                    correction = np.linspace(prev_correction, a_correction, a.shape[1])
                    a[...] = (a * correction[None,:,:]).astype(np.int16)

                    prev_correction = b_correction

                    # Blend the edges
                    if w > 0:
                        alpha = np.linspace(0, 1, w, dtype=np.float32)[np.newaxis, :, np.newaxis]  # (1,w,1)
                        alpha = np.clip(2*(alpha-0.5) + 0.5, a_min=0, a_max=1) # blend only the inner part of overlap area

                        strip_a_end     = a[:, -w:, :].astype(np.float32)
                        strip_b_start   = b[:, :w,  :].astype(np.float32) * b_correction

                        blended = ((1-alpha)*strip_a_end + alpha*strip_b_start).astype(np.int16)

                        a[:, -w:, :] = blended  # only correct A since B (edge) will not be used for tiles
                    
                    print(f"Publishing stitched strip {stitched_strip.indices}")
                    self._publish(stitched_strip)
                    stitched_strip = self._get_free_product()
                    stitched_strip.data[...] = in_strip.data
                    stitched_strip.indices = tuple(in_strip.indices)

                    if in_strip.indices[1] == self.n_strips - 1:
                        # on last strip of the z opt. section, publish last strip
                        correction = np.linspace(prev_correction, 1, b.shape[1])
                        stitched_strip.data[...] = (b * correction[None,:,:]).astype(np.int16)

                        print(f"Publishing stitched strip {in_strip.indices}")
                        self._publish(stitched_strip)
                        stitched_strip = self._get_free_product()

                        prev_correction = 1

        except EndOfStream:
            self._publish(None)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
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
    """Parcels up tiles to send to file writer."""
    Product = TileProduct

    def __init__(self, 
                 upstream: StripStitcher, 
                 tile_shape=(512,512)):
        super().__init__(upstream, name="TileBuilder")
        self._acquisition: RasterScanStitchedAcquisition | LineCameraStitchedAcquisition
        self._spec: RasterScanStitchedAcquisitionSpec | LineCameraStitchedAcquisitionSpec
        
        if isinstance(self._acquisition.runtime_info, LineAcquisitionRuntimeInfo):
            scan_axis = self._acquisition.system_config.fast_raster_scanner['axis']
        elif isinstance(self._acquisition.runtime_info, CameraAcquisitionRuntimeInfo):
            scan_axis = self._acquisition.system_config.line_camera['axis']
        else:
            raise RuntimeError(f"Acquistion runtime_info is unexpected type: "
                               f"{type(self._acquisition.runtime_info)}")

        if tile_shape[0] != tile_shape[1]:
            raise ValueError("Tile shape must be square")

        self._data_range = upstream.data_range
        self._tile_shape = tile_shape
        self._n_channels = self._acquisition.runtime_info.n_channels

        self._init_product_pool(
            n =     10,     # TODO how should this be set?
            shape = (*self._tile_shape, self._n_channels), 
            dtype = np.int16
        )

        self._n_strips = upstream.n_strips

        if scan_axis == 'x':
            self.n_pixels_scan = round(self._spec.x_range.range / self._spec.pixel_size)
            self.n_pixels_web  = round(self._spec.y_range.range / self._spec.pixel_size)
        else:
            self.n_pixels_scan = round(self._spec.y_range.range / self._spec.pixel_size)
            self.n_pixels_web  = round(self._spec.x_range.range / self._spec.pixel_size)
        self._tiles_web  = math.ceil(self.n_pixels_web / tile_shape[0])  # tiles along the web dimension (strip long axis)
        self._tiles_scan = math.ceil(self.n_pixels_scan / tile_shape[0]) # tiles along the scan dimension (strip short axis)
        self._tiles_image = self._tiles_web * self._tiles_scan

        self._leftovers: Optional[np.ndarray] = None

    def _receive_product(self) -> ProcessorProduct:
        return super()._receive_product() # type: ignore
        
    def _work(self):
        tiles_scan = self._tiles_scan
        tiles_web  = self._tiles_web
        tile_idx = 0   # tile XY coordinate
        t_z = 0        # tile Z coordinate

        tile_shape = self._tile_shape
        effective_pixels_per_line = int(
            self._spec.pixels_per_line * (1-self._spec.strip_overlap)
        )
        overlap_pixels = self._spec.pixels_per_line - effective_pixels_per_line

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
                        
                        p_s = t_s * tile_shape[0]   # scan dim global pixel tile start
                        p_w = t_w * tile_shape[1]   # web dim global pixel tile start
                        scan_offset = strip.indices[1] * effective_pixels_per_line # scan dim global pixel strip start

                        p_so = p_s - scan_offset   # scan pixel coordinate relative to the current strip
                        # p_so < 0 means that the tile "starts" in the previous strip

                        # If start of next tile will exceed current strip, then we need another strip to complete it; store leftovers
                        if (p_so + tile_shape[0]) > strip.data.shape[1]:
                            self._leftovers = strip.data[:, p_so:, :].copy()
                            # and not last strip of z level:
                            if strip.indices[1] < (self._n_strips-1):
                                break # go on to recieve a new strip to complete the tile

                        tile = self._get_free_product()
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
                                overlap_pixels:(self._tile_shape[1] + p_so),
                                :
                            ]
                            tile.data[:data2.shape[0], -data2.shape[1]:, :] = data2
                        
                        _transpose_inplace(tile.data) # go from strips in dimensions (web, scan, chan) to tiles in (scan, web, chan)

                        self._publish(tile)
                        tile_idx += 1

        except EndOfStream:
            self._publish(None)

    def _get_free_product(self) -> TileProduct:
        return super()._get_free_product() # type: ignore

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range



sigs = [
     uint8[:,:,:](uint8[:,:,:],      int64),
     uint8[:,:,:](uint8_3d_readonly, int64),
    #uint16[:,:,:](uint16[:,:,:], int64),
     int16[:,:,:](int16_3d_readonly, int64)
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
                 show_progress: bool = True,
                 **kwargs):
        super().__init__(upstream, name="StitchedPreview", **kwargs)
        self._acquisition: RasterScanStitchedAcquisition
        self._data_range = upstream.data_range
        self._downsample = downsample
        self._show_progress = show_progress

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
        preview_shape = (self._acquisition.final_shape[1] // self._downsample, 
                         self._acquisition.final_shape[2] // self._downsample,
                         self._acquisition.final_shape[3]) # the preview does not have a Z dimension

        self._init_product_pool(
            n       = 1,
            shape   = preview_shape, 
            dtype   = self.data_range.recommended_dtype,
        )

    def _receive_product(self) -> TileProduct:
        return super()._receive_product() # type: ignore
    
    def _work(self):
        if self._show_progress:
            # Hold until blank is published
            while self._n_published == 0:
                time.sleep(0.05)

        preview = self._get_free_product()

        try:
            for tz in range(self._z_levels):
                for ts in range(self._tiles_scan):
                    # work through all the tiles along the web direction, then publish
                    for tw in range(self._tiles_web):

                        with self._receive_product() as tile:
                            i0 = ts * self._downsampled_tile_length
                            j0 = tw * self._downsampled_tile_length
                            i1 = min(i0 + self._downsampled_tile_length, self.product_shape[0])
                            j1 = min(j0 + self._downsampled_tile_length, self.product_shape[1])
                            # downsample and place in array
                            preview.data[i0:i1, j0:j1, :] = \
                                downsample_kernel(tile.data, self._downsample)[:(i1-i0), :(j1-j0), :]

                    if self._show_progress:
                        self._publish(preview)
                        preview = self._get_free_product()
                    elif ts == (self._tiles_scan - 1):
                        self._publish(preview)

            self._publish(None) # forward sentinel None

        except EndOfStream:
            self._publish(None) # forward sentinel None        
    
    def publish_blank(self):
        self._publish(self._get_free_product())

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range

