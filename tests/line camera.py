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
disp_processor  = diri.make_display_processor(
    name            = "frame",
    upstream        = strip_processor, 
    pixel_format    = DisplayPixelFormat.RGB24
)
disp_processor.display_channels[0].display_min = 0
disp_processor.display_channels[0].display_max = 32000

writer          = diri.make_writer("tiff", upstream=disp_processor)


acquisition.start()
acquisition.join(timeout=100.0)

print("Acquisition complete")
