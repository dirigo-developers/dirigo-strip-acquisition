
import numpy as np

from dirigo.sw_interfaces.worker import Product, EndOfStream
from dirigo.sw_interfaces.display import Display, DisplayProduct, DisplayChannel
from dirigo.plugins.displays import default_colormap_lists, ColorVector

from dirigo_strip_acquisition.acquisitions import StitchedAcquisition
from dirigo_strip_acquisition.processors import TileBuilder, TileProduct
from dirigo_strip_acquisition.loggers import downsample_kernel



class StitchedPreviewDisplay(Display):
    """Creates a downsampled preview of stitched image from tiles."""
    def __init__(self, 
                 upstream: TileBuilder, 
                 downsample: int = 16,
                 **kwargs):
        super().__init__(upstream, **kwargs)
        self._acquisition: StitchedAcquisition

        self._prev_data = None # None indicates that no data has been acquired yet
        self._downsample = downsample

        n_channels = upstream.product_shape[2]
        bpp = self.bits_per_pixel
        self._luts = np.zeros(
            shape = (n_channels, self.data_range.range + 1, bpp), 
            dtype = np.uint16
        )

        colormap_list = default_colormap_lists(n_channels)
        
        self.display_channels: list[DisplayChannel] = []
        for ci, colormap_name in enumerate(colormap_list):
            dc = DisplayChannel(
                lut_slice=self._luts[ci],
                color_vector=ColorVector[colormap_name.upper()],
                display_range=self.data_range,
                pixel_format=self._pixel_format,
                update_method=self.update_display,
                gamma_lut_length=self.gamma_lut_length
            )
            self.display_channels.append(dc)

        # Product is a downsampled version of the full stitched image       
        self._tile_length = upstream._tile_shape[0] // downsample
        self._tiles_scan = upstream._tiles_scan
        self._tiles_web  = upstream._tiles_web
        preview_shape = (self._tiles_scan * self._tile_length, 
                         self._tiles_web  * self._tile_length)     # (scan, web)
        self._init_product_pool(n=1, shape=(*preview_shape, bpp), dtype=np.uint8)

    def _receive_product(self) -> TileProduct:
        return super()._receive_product() # type: ignore
    
    def run(self):
        try:
            for ts in range(self._tiles_scan):
                # Get the preview product from the pool
                preview = self._get_free_product()

                # work through all the tiles along the web direction, then publish
                for tw in range(self._tiles_web):

                    with self._receive_product() as tile:
                        assert tile.coords == (ts, tw), "Tile coordinate mismatch"

                        i0 = ts * self._tile_length
                        j0 = tw * self._tile_length
                        i1 = i0 + self._tile_length
                        j1 = j0 + self._tile_length
                        # downsample and place in array
                        preview.data[i0:i1, j0:j1, :2] = downsample_kernel(tile.data, self._downsample)
                        
                        # self._apply_display_kernel(
                        #     image           = tile.data, 
                        #     luts            = self._luts, 
                        #     display_image   = preview.data
                        # )

                self._publish(preview)
                print("Published preview on scan row", ts)

        except EndOfStream:
            self._publish(None) # forward sentinel None


    def _apply_display_kernel(self, image, luts, display_image):
        #additive_display_kernel(image, luts, self.gamma_lut, display_image)
        pass

    @property
    def gamma(self) -> float:
        return self._gamma
    
    @gamma.setter
    def gamma(self, new_gamma: float):
        if not isinstance(new_gamma, float):
            raise ValueError("Gamma must be set with a float value")
        if not (0 < new_gamma <= 10):
            raise ValueError("Gamma must be between 0.0 and 10.0")
        self._gamma = new_gamma

        # Generate gamma correction LUT
        x = np.arange(self.gamma_lut_length) \
            / (self.gamma_lut_length - 1) # TODO, not sure about the -1
        
        gamma_lut = (2**self._monitor_bit_depth - 1) * x**(self._gamma)
        if self._monitor_bit_depth > 8:
            self.gamma_lut = np.round(gamma_lut).astype(np.uint16)
        else:
            self.gamma_lut = np.round(gamma_lut).astype(np.uint8)

    def update_display(self, skip_when_acquisition_in_progress: bool = True):
        """
        On demand reprocessing of the last acquired frame for display.
        
        Used when the acquisition is stopped and need to update the appearance  
        of the last acquired frame.
        """
        pass

    @property
    def n_frame_average(self) -> int:       # TODO, cut this out
        return 1

    @n_frame_average.setter
    def n_frame_average(self, frames: int):
        pass
