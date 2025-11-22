import subprocess, pathlib, os

from dirigo.main import Dirigo
from dirigo.sw_interfaces.display import DisplayPixelFormat


diri = Dirigo()

acquisition     = diri.make_acquisition("raster_scan_stitched", spec="raster_scan")

line_processor  = diri.make_processor("raster_frame", upstream=acquisition)
strip_processor = diri.make_processor("strip", upstream=line_processor)
strip_stitcher  = diri.make_processor("strip_stitcher", upstream=strip_processor)
# tile_builder    = diri.make_processor("tile_builder", upstream=strip_stitcher)

# preview         = diri.make_processor(
#     name            = "stitch_preview",
#     upstream        = tile_builder,
#     downsample      = 8
# )

# preview_display = diri.make_display_processor(
#     name                    = "frame",
#     upstream                = preview,
#     pixel_format            = DisplayPixelFormat.BGRX32,
#     color_vector_names      = ["hematoxylin", "eosin"],
#     transfer_function_name  = "negative_exponential"
# )

# colorizer       = diri.make_display_processor(
#     name                    = "frame",
#     upstream                = tile_builder,
#     color_vector_names      = ["hematoxylin", "eosin"],
#     transfer_function_name  = "negative_exponential"
# )
# for channel in colorizer.display_channels:
#     channel.display_min = 0
#     channel.display_max = 4000

# pyramid_writer  = diri.make_writer(
#     name            = "pyramid", 
#     upstream        = colorizer,
#     levels          = (1, 2, 4, 8, 16, 32),
#     compression     = 'jpeg'
# )

strip_writer = diri.make_writer(
    name        = "tiff",
    upstream    = strip_stitcher
)


acquisition.start()
strip_writer.join()

