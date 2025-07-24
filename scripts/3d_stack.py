import numpy as np
from tifffile import TiffWriter


file_path = r"D:\dirigo-data\sample_z_stack.ome.tif"
Nz = 4
Ny = 2048
Nx = 3072
Nc = 3
data = np.random.randint(0, 2**8-1, (Nz, Ny, Nx, Nc), 'uint8')
tile_shape = (128, 128)
levels = (1,2,8)


def tiles(data):
    for z in range(Nz):
        for y in range(0, Ny, tile_shape[0]):
            for x in range(0, Nx, tile_shape[1]):
                yield data[z, y : y + tile_shape[0], x : x + tile_shape[1]]


with TiffWriter(file_path, bigtiff=True) as tif:
    options = dict(
        photometric='rgb',
        tile=tile_shape,
    )
    metadata = {
        'axes': 'ZYXS',
    }
    
    tif.write(
        #tiles(data),
        data,
        metadata    = metadata,
        shape       = data.shape,
        dtype       = np.uint8,
        subifds     = len(levels) - 1,
        **options
    )

    for level in levels[1:]:
        tif.write(
            data[:, ::level, ::level, :],
            subfiletype=1,
            **options,
        )

