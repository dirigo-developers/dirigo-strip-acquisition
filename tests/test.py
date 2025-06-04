from dirigo.main import Dirigo


diri = Dirigo()
    
acquisition     = diri.make_acquisition("line_camera_stitched", spec="line_camera")
line_processor  = diri.make_processor("line_camera_line", upstream=acquisition)
strip_processor = diri.make_processor("strip", upstream=line_processor)
# raw_logger  = diri.make_logger("tiff", upstream=acquisition)
# raw_logger.frames_per_file = 100

acquisition.start()
acquisition.join(timeout=100.0)

print("Acquisition complete")
