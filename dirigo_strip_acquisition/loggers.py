from typing import Iterator, Optional
import math, time

import tifffile
import numpy as np

from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces import Logger
from dirigo_strip_acquisition.processors import (
    TileBuilder, TileProduct, downsample_kernel
)
from dirigo_strip_acquisition.acquisitions import StitchedAcquisition



class PyramidLogger(Logger):

    def __init__(self, 
                 upstream: TileBuilder, 
                 levels: tuple = (1, 2, 8),
                 compression: Optional[None] = None):
        super().__init__(upstream)
        self._acquisition: StitchedAcquisition
        
        self._n_channels = upstream.product_shape[2]
        self._shape      = (*self._acquisition.final_shape[:3], self._n_channels)
        self._dtype      = upstream.product_dtype
        self._tile_shape = upstream.product_shape[:2]
        self._levels     = levels
        if self._dtype == np.uint8 and self._n_channels == 3:
            photometric = 'rgb'
        else:
            photometric = 'minisblack'
        self._options    = dict(
            tile            = self._tile_shape, 
            dtype           = self._dtype,
            compression     = compression,
            photometric     = photometric,
            planarconfig    = 'contig',
        )
        self._metadata={
            'axes': 'ZYXS',
            'PhysicalSizeX': float(self._acquisition.spec.pixel_size),
            'PhysicalSizeXUnit': 'm',
            'PhysicalSizeY': float(self._acquisition.spec.pixel_size),
            'PhysicalSizeYUnit': 'm',
            'Channel': {'Name': ['Channel 1', 'Channel 2']}, # TODO, rename these out of system config
        }
        
        # Pre-allocate for downsampled data (saved all at once after full-res)
        self._ds_tiles = []
        for d in self._levels[1:]:
            # self._shape: (z, scan, web, channels)
            shape = (
                self._shape[0],
                math.ceil(self._shape[1] / d / self._tile_shape[0]),
                math.ceil(self._shape[2] / d / self._tile_shape[1]),
                self._tile_shape[0],
                self._tile_shape[1],
                self._n_channels
            )
            tiles = np.zeros(shape, dtype=self._dtype)
            self._ds_tiles.append(tiles)

    def _receive_product(self) -> TileProduct:
        return super()._receive_product() # type: ignore

    def _tiles_gen(self) -> Iterator[np.ndarray]:
        """Yield tile data, blocking on queue."""
        n_z         = self._shape[0]
        ntiles_scan = math.ceil(self._shape[1] / self._tile_shape[0])
        ntiles_web  = math.ceil(self._shape[2] / self._tile_shape[1])
        try:
            for z in range(n_z):
                for ti in range(ntiles_scan):
                    for tj in range(ntiles_web):
                        with self._receive_product() as tile: # incoming tiles are 2D (+channel dim)
                            
                            # While we have the full-res tile here, do downsampling
                            prev_f = 1
                            prev_data = tile.data
                            for lvl_idx, f in enumerate(self._levels[1:]):
                                # calculate tile index in downsampled image
                                di = ti // f
                                dj = tj // f

                                i0 = int( (ti / f - di) * self._tile_shape[0] )
                                j0 = int( (tj / f - dj) * self._tile_shape[1] )
                                i1 = i0 + self._tile_shape[0] // f
                                j1 = j0 + self._tile_shape[1] // f

                                df = f // prev_f
                                self._ds_tiles[lvl_idx][z, di, dj, i0:i1, j0:j1 , :] = \
                                    downsample_kernel(prev_data, df)

                                prev_f = f
                                prev_data = self._ds_tiles[lvl_idx][z, di, dj, i0:i1, j0:j1 , :]
                                
                            # yield full resolution data 
                            yield tile.data  

        except EndOfStream:
            self._publish(None)

    def _downsampled_tiles_gen(self, level_idx) -> Iterator[np.ndarray]:
        # Get the tiles corresponding to a particular downsampled level
        ds_tiles = self._ds_tiles[level_idx]
        n_z, n_rows, n_cols = ds_tiles.shape[:3]

        try:
            for z in range(n_z):
                for ti in range(n_rows):
                    for tj in range(n_cols):
                        yield ds_tiles[z, ti, tj, ...]
        except GeneratorExit:
            pass

    def run(self):
        try:
            self.save_data()
        
        finally:
            self._publish(None)
            print("Image write complete")

    def save_data(self):
        while not self._acquisition.is_alive() or self._stop_event.is_set():
            # Spin while waiting for acquisition to start
            time.sleep(0.01)

        fp = self.save_path / f"{self.basename}.ome.tif"
        with tifffile.TiffWriter(fp, bigtiff=True) as tif:
            tif.write(
                self._tiles_gen(),
                shape=self._shape,
                subifds=len(self._levels) - 1,
                metadata=self._metadata,
                **self._options # type: ignore
            )

            # write downsampled levels
            for level_idx, f in enumerate(self._levels[1:]):
                n_z, d_h, d_w = self._shape[0], math.ceil(self._shape[1] / f), math.ceil(self._shape[2] / f)
                tif.write(
                    self._downsampled_tiles_gen(level_idx),
                    shape=(n_z, d_h, d_w, self._n_channels),
                    subfiletype=1,
                    **self._options # type: ignore
                )
        
        self.last_saved_file_path = fp
