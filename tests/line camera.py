from dirigo.main import Dirigo
from dirigo.sw_interfaces.display import DisplayPixelFormat


diri = Dirigo()
    
acquisition     = diri.make_acquisition("line_camera_stitched", spec="line_camera")
line_processor  = diri.make_processor(
    name        = "line_camera_line", 
    upstream    = acquisition
)
strip_processor = diri.make_processor(
    name        = "strip", 
    upstream    = line_processor
)
strip_stitcher = diri.make_processor(
    name        = "strip_stitcher",
    upstream    = strip_processor
)
tile_builder    = diri.make_processor(
    name        = "tile_builder", # type: ignore
    upstream    = strip_stitcher
)
stitch_preview  = diri.make_processor(
    name          = "stitch_preview",
    upstream      = tile_builder,
    downsample    = 1, # type: ignore
    show_progress = False # type: ignore
)

disp_processor  = diri.make_display_processor(
    name            = "frame",
    upstream        = stitch_preview, 
    pixel_format    = DisplayPixelFormat.RGB24
)
disp_processor.display_channels[0].display_min = 0
disp_processor.display_channels[0].display_max = 32000

writer = diri.make_writer("tiff", upstream=disp_processor)


acquisition.start()
acquisition.join(timeout=100.0)

print("Acquisition complete")
