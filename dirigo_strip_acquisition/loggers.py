from typing import Iterator
import math

import tifffile
import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces import Logger
from dirigo_strip_acquisition.processors import TileBuilder, TileProduct
from dirigo_strip_acquisition.acquisitions import StitchedAcquisition



sig = [types.int16[:,:,:](types.int16[:,:,:], types.int64)]
@njit(sig, parallel=True, fastmath=True, cache=True)
def _downsample(tile: np.ndarray, f: int) -> np.ndarray:
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
    

class PyramidLogger(Logger):

    def __init__(self, upstream: TileBuilder, levels: tuple = (1, 2, 8)):
        super().__init__(upstream)
        self._acq: StitchedStripAcquisition
        
        self._file       = self.save_path / f"{self.basename}.ome.tif"
        self._n_channels = upstream._n_channels
        self._shape      = self._acq.final_shape
        self._dtype      = upstream.data_range.recommended_dtype
        self._tile_shape = upstream._tile_shape
        self._levels     = levels
        self._options    = dict(
            tile=self._tile_shape, 
            dtype=self._dtype,
            compression=None,
            photometric='minisblack',
            planarconfig='contig',
        )
        self._metadata={
            'PhysicalSizeX': float(self._acq.spec.pixel_size),
            'PhysicalSizeXUnit': 'm',
            'PhysicalSizeY': float(self._acq.spec.pixel_size),
            'PhysicalSizeYUnit': 'm',
            'Channel': {'Name': ['Channel 1', 'Channel 2']}, # TODO, rename these out of system config
        }
        
        self._ds_tiles = []
        for d in self._levels[1:]:
            shape = (
                math.ceil(self._shape[0] / d / self._tile_shape[0]),
                math.ceil(self._shape[1] / d / self._tile_shape[1]),
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
        try:
            while True:
                with self._receive_product() as tile:
                    # downsample & store into levels
                    if tile.coords is None:
                        raise RuntimeError("Tile coordinates not initialized")
                    for level_idx, f in enumerate(self._levels[1:]):
                        di = tile.coords[0] // f
                        dj = tile.coords[1] // f
                        i0 = int( (tile.coords[0] / f - di) * self._tile_shape[0] )
                        j0 = int( (tile.coords[1] / f - dj) * self._tile_shape[1] )
                        i1 = i0 + self._tile_shape[0] // f
                        j1 = j0 + self._tile_shape[1] // f
                        self._ds_tiles[level_idx][di, dj, i0:i1, j0:j1 , :] = \
                            _downsample(tile.data, f) # TODO branch here to use prev downsampled data

                    # yield full resolution data 
                    yield tile.data  

        finally:
            self._publish(None)

    def _downsampled_tiles_gen(self, level_idx) -> Iterator[np.ndarray]:
        ds_tiles: np.ndarray = self._ds_tiles[level_idx]
        tile_idx = 0

        while True:
            ti = tile_idx // ds_tiles.shape[1]
            tj = tile_idx %  ds_tiles.shape[1]
            if ti >= ds_tiles.shape[0]: 
                break
            yield ds_tiles[ti, tj, ...]
            tile_idx += 1
                               
    def run(self):
        try:
            self.save_data()
        
        finally:
            self._publish(None)
            print("Image write complete")

    def save_data(self):

        with tifffile.TiffWriter(self._file, bigtiff=True) as tif:
            tif.write(
                self._tiles_gen(),
                shape=self._shape,
                subifds=len(self._levels) - 1,
                metadata=self._metadata,
                **self._options # type: ignore
            )

            # write downsampled levels
            for level_idx, f in enumerate(self._levels[1:]):
                d_h, d_w = math.ceil(self._shape[0] / f), math.ceil(self._shape[1] / f)
                tif.write(
                    self._downsampled_tiles_gen(level_idx),
                    shape=(d_h, d_w, self._n_channels),
                    subfiletype=1,
                    **self._options # type: ignore
                )
