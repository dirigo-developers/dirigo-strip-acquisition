from typing import Iterator

import tifffile
import numpy as np

from dirigo.sw_interfaces import Logger
from dirigo_strip_acquisition.processors import StripStitcher, TileProduct



class PyramidLogger(Logger):

    def __init__(self, upstream: StripStitcher):
        super().__init__(upstream)
        
        self._file       = self.save_path / f"{self.basename}.tif"
        self._shape      = upstream._full_shape
        self._dtype      = np.int16
        self._tile       = upstream._tile_shape[:2]  # only need XY tile dims
        #self._levels     = levels
        self._options    = dict(tile=self._tile, dtype=self._dtype,
                                compression=None)

    def _tiles_gen(self) -> Iterator[np.ndarray]:
        """Yield tile data, blocking on queue."""
        while True:
            tile: TileProduct = self.inbox.get()
            if tile is None: return     # finished sentinel
            with tile: 
                yield tile.data  # yield image data                   
                               
    def run(self):
        try:
            with tifffile.TiffWriter(self._file, bigtiff=True) as tif:
                tif.write(
                    self._tiles_gen(),
                    shape=self._shape,
                    photometric='minisblack',
                    planarconfig='contig',
                    #subifds=self._levels - 1,
                    **self._options
                )
        
        finally:
            self.publish(None)

    def save_data(self, data):
        pass