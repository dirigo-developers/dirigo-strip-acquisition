from pathlib import Path
import struct, json

import numpy as np
import tifffile

from dirigo.components.io import SystemConfig
from dirigo.sw_interfaces.acquisition import AcquisitionProduct, Loader
from dirigo.hw_interfaces.digitizer import DigitizerProfile
from dirigo.plugins.acquisitions import LineAcquisitionRuntimeInfo
from dirigo.plugins.loggers import TiffLogger
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo_strip_acquisition.acquisitions import (
    StripAcquisitionSpec, RectangularFieldStagePositionHelper
)
from dirigo_strip_acquisition.processors import StripProcessor, StripStitcher, TileBuilder
from dirigo_strip_acquisition.loggers import PyramidLogger


def deserialize_float64_list(blob: bytes):
    """
    Reverse of `serialize_float64_list` (full shape in header).
    Returns a list of np.ndarray, all copies (writable).
    """
    ndims, = struct.unpack_from("<Q", blob, 0)

    fmt          = f"<Q{ndims}Q"
    header_size  = struct.calcsize(fmt)
    shape        = struct.unpack_from(fmt, blob, 0)[1:]   # skip ndims

    items_per_frame = np.prod(shape)
    bytes_per_frame = items_per_frame * 8                    # float64 → 8 B
    n_frames        = (len(blob) - header_size) // bytes_per_frame

    data   = np.frombuffer(blob, dtype=np.float64, offset=header_size)
    stack  = data.reshape((n_frames, *shape))
    return [stack[i].copy() for i in range(n_frames)]


class StripAcquisitionLoader(Loader): 
    def __init__(self, file_path: str | Path):
        super().__init__(file_path, thread_name="Strip acquisition data loader")

        with tifffile.TiffFile(self._file_path) as tif:   

            self.init_product_pool(
                n=4, 
                shape=tif.pages[0].shape, 
                dtype=tif.pages[0].dtype
            )

            tags = tif.pages[0].tags

            cfg_dict = json.loads(tags[TiffLogger.SYSTEM_CONFIG_TAG].value)
            self.system_config = SystemConfig(**cfg_dict)

            runtime_dict = json.loads(tags[TiffLogger.RUNTIME_INFO_TAG].value)
            self.runtime_info = LineAcquisitionRuntimeInfo.from_dict(runtime_dict)

            spec_dict = json.loads(tags[TiffLogger.ACQUISITION_SPEC_TAG].value)
            self.spec = StripAcquisitionSpec(**spec_dict)

            digi_dict = json.loads(tags[TiffLogger.DIGITIZER_PROFILE_TAG].value)
            self.digitizer_profile = DigitizerProfile.from_dict(digi_dict)

        self.positioner = RectangularFieldStagePositionHelper(
            scan_axis=self.system_config.fast_raster_scanner['axis'],
            spec=self.spec
        )
        
        if self.system_config.fast_raster_scanner['axis'] == 'x':
            n_pixels_scan = round(self.spec.x_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.y_range.range / self.spec.pixel_size)
        else:
            n_pixels_scan = round(self.spec.y_range.range / self.spec.pixel_size)
            n_pixels_web  = round(self.spec.x_range.range / self.spec.pixel_size)
        n_channels = sum([c.enabled for c in self.digitizer_profile.channels])

        self.final_shape = (
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
                    self.publish(frame)

                    frames_read += 1
        finally:
            self.publish(None) # sentinel coding finished
        

    def init_product_pool(self, n, shape, dtype):
        for _ in range(n):
            aq_buf = AcquisitionProduct(
                pool=self._product_pool,
                data=np.empty(shape, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(aq_buf)

    def get_free_product(self) -> AcquisitionProduct:
        return self._product_pool.get()
    



if __name__ == "__main__":
    fn = r"C:\Users\MIT\Documents\Dirigo\experiment.tif"

    loader = StripAcquisitionLoader(fn)
    processor = RasterFrameProcessor(upstream=loader)
    strip_processor = StripProcessor(upstream=processor)
    strip_stitcher = StripStitcher(upstream=strip_processor)
    tile_builder = TileBuilder(upstream=strip_stitcher)
    logger = PyramidLogger(upstream=tile_builder)

    loader.add_subscriber(processor)
    processor.add_subscriber(strip_processor)
    strip_processor.add_subscriber(strip_stitcher)
    strip_stitcher.add_subscriber(tile_builder)
    tile_builder.add_subscriber(logger)

    processor.start()
    strip_processor.start()
    strip_stitcher.start()
    tile_builder.start()
    logger.start()

    loader.start()
