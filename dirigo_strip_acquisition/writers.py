from typing import Iterator
import math, time
from functools import cached_property

import tifffile
import numpy as np

from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces import Writer
from dirigo_strip_acquisition.processors import (
    TileBuilder, TileProduct, downsample_kernel
)
from dirigo_strip_acquisition.acquisitions import StitchedAcquisition


class PyramidWriter(Writer):

    def __init__(
            self, 
            upstream: TileBuilder, 
            levels: tuple = (1, 2, 8),
            basename: str = "experiment",
            compression: str | None = None
        ):
        super().__init__(upstream, basename)
        self._acquisition: StitchedAcquisition

        self.file_ext = "ome.tif"
        
        self._n_channels = upstream.product_shape[2]
        spec = self._acquisition.spec
        self._shape = (                     # (z, y, x, c)
            self._acquisition.spec.z_steps,
            round(spec.y_range.range / spec.pixel_size),
            round(spec.x_range.range / spec.pixel_size),
            self._n_channels
        )
        self._dtype      = upstream.product_dtype
        self._tile_shape = upstream.product_shape[:2]
        self._levels     = levels

        self._options    = dict(
            tile            = self._tile_shape, 
            dtype           = self._dtype,
            compression     = compression,
            photometric     = 'rgb',
            planarconfig    = 'contig',
        )
        self._metadata={
            'axes': 'ZYXS',
            'PhysicalSizeX': float(self._acquisition.spec.pixel_size),
            'PhysicalSizeXUnit': 'm',
            'PhysicalSizeY': float(self._acquisition.spec.pixel_size),
            'PhysicalSizeYUnit': 'm',
            'PhysicalSizeZ': float(self._acquisition.spec.z_step),
            'PhysicalSizeZUnit': 'm',
        }
        
        # Pre-allocate for downsampled data (saved all at once after full-res)
        self._ds_tiles = []
        for d in self._levels[1:]:
            downsampled_shape = (
                self._shape[1] / d, # y
                self._shape[2] / d, # x
            )
            shape = (
                self._shape[0],
                math.ceil(downsampled_shape[0] / self._tile_shape[0]),
                math.ceil(downsampled_shape[1] / self._tile_shape[1]),
                self._tile_shape[0],
                self._tile_shape[1],
                self._n_channels
            )
            tiles = np.zeros(shape, dtype=self._dtype)
            self._ds_tiles.append(tiles)

    def _receive_product(self) -> TileProduct:
        return super()._receive_product() # type: ignore

    @cached_property
    def is_stream_row_major(self) -> bool:
        """True if strips are along horizontal (X) direction, False otherwise"""
        if self._acquisition.system_config.fast_raster_scanner['axis'] == "y":
            return True
        else:
            return False

    def _tiles_gen(self) -> Iterator[np.ndarray]:
        """Yield tile data, blocking on queue."""
        n_z = self._shape[0]
        ntiles_y = math.ceil(self._shape[1] / self._tile_shape[0])
        ntiles_x = math.ceil(self._shape[2] / self._tile_shape[1])
        ntiles_per_level = ntiles_y * ntiles_x

        i = 0
        try:
            while True:
                with self._receive_product() as tile:
                    # compute tile indices
                    z = i // ntiles_per_level
                    ii = i % ntiles_per_level
                    if self.is_stream_row_major:
                        ty = ii // ntiles_x
                        tx = ii % ntiles_x
                    else:
                        ty = ii % ntiles_y
                        tx = ii // ntiles_y

                    # While we have the full-res tile here, do downsampling
                    prev_f = 1
                    prev_data = tile.data
                    for lvl_idx, f in enumerate(self._levels[1:]):
                        # calculate tile index in downsampled image
                        dy = ty // f
                        dx = tx // f
                        
                        i0 = int( (ty / f - dy) * self._tile_shape[0] )
                        j0 = int( (tx / f - dx) * self._tile_shape[1] )
                        i1 = i0 + self._tile_shape[0] // f
                        j1 = j0 + self._tile_shape[1] // f

                        df = f // prev_f
                        self._ds_tiles[lvl_idx][z, dy, dx, i0:i1, j0:j1 , :] = \
                            downsample_kernel(prev_data, df)

                        prev_f = f
                        prev_data = self._ds_tiles[lvl_idx][z, dy, dx, i0:i1, j0:j1 , :]

                    # yield full resolution data 
                    yield tile.data 

                i += 1 

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

    def _work(self):
        try:
            self.save_data()
        
        finally:
            self._publish(None)
            print("Image write complete")

    def save_data(self):
        while not self._acquisition.is_alive() or self._stop_event.is_set():
            # Spin while waiting for acquisition to start
            time.sleep(0.01)

        fp = self._file_path()
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

        # Patch tile order
        if self.is_stream_row_major == False:
            with tifffile.TiffFile(fp, mode='r+') as tif:
                native_level = tif.series[0].levels[0]

                for page in native_level.pages:
                    if not hasattr(page, 'tags'):
                        # tifffile sometimes returns type TiffFrame (lacking tags) rather than TiffPage 
                        page = page.aspage()
                    self._patch_page_column_major(page)

        self.last_saved_file_path = fp

    @staticmethod
    def _patch_page_column_major(page: tifffile.TiffPage) -> None:

        tile_height = page.tilelength
        tile_width = page.tilewidth

        n_tiles_y = math.ceil(page.imagelength / tile_height)
        n_tiles_x = math.ceil(page.imagewidth / tile_width)
        n_tiles = n_tiles_y * n_tiles_x

        offsets_tag = page.tags['TileOffsets']
        bytecounts_tag = page.tags['TileByteCounts']

        old_offsets = np.asarray(offsets_tag.value, dtype=np.uint64)
        old_bytecounts = np.asarray(bytecounts_tag.value, dtype=np.uint64)

        permutation = np.fromiter(
            (
                tile_x * n_tiles_y + tile_y
                for tile_y in range(n_tiles_y)
                for tile_x in range(n_tiles_x)
            ),
            dtype=np.intp,
            count=n_tiles,
        )

        new_offsets = old_offsets[permutation]
        new_bytecounts = old_bytecounts[permutation]

        offsets_tag.overwrite(
            tuple(int(value) for value in new_offsets)
        )
        bytecounts_tag.overwrite(
            tuple(int(value) for value in new_bytecounts)
        )
