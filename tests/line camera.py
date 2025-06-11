from dirigo.main import Dirigo
from dirigo.sw_interfaces.display import DisplayPixelFormat


diri = Dirigo()
    
acquisition     = diri.make_acquisition("line_camera_stitched", spec="line_camera")
line_processor  = diri.make_processor("line_camera_line", upstream=acquisition)
strip_processor = diri.make_processor("strip", upstream=line_processor)
disp_processor  = diri.make_display_processor(
    name            = 'rgb_frame',
    upstream        = strip_processor, 
    pixel_format    = DisplayPixelFormat.BGRX32
)
# TODO, put the following settings in some sort of toml file
disp_processor.gamma = 1.0
disp_processor.display_channels[0].display_min = 0
disp_processor.display_channels[0].display_max = 8000


acquisition.start()
acquisition.join(timeout=100.0)

print("Acquisition complete")
