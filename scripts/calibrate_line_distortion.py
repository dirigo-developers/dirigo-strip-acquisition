

from dirigo.main import Dirigo
from dirigo import units

from dirigo_strip_acquisition.acquisitions import (
    LineCameraStitchedAcquisition, LineCameraStitchedAcquisitionSpec,
    RasterScanStitchedAcquisition, RasterScanStitchedAcquisitionSpec
)


DEVICE = "line camera"
TRANSLATION = units.Position("1 mm") # make divisible by target pixel size
N_MOVES = 1

# Calibration script
diri = Dirigo()

# Use default StitchedAcquisition spec as basis for line width, pixel size, etc.
if DEVICE == "line camera":
    spec = LineCameraStitchedAcquisition.get_specification("line_camera") # type: ignore
    line_axis = diri.hw.line_camera.axis
elif DEVICE == "raster scan":
    spec = RasterScanStitchedAcquisition.get_specification("raster_scan") # type: ignore
    line_axis = diri.hw.fast_raster_scanner.axis
else: 
    raise Exception
spec: LineCameraStitchedAcquisitionSpec | RasterScanStitchedAcquisitionSpec

if line_axis == "x":
    scan_range = spec.x_range
    web_range = spec.y_range
elif line_axis == "y":
    scan_range = spec.y_range
    web_range = spec.x_range
else:
    raise Exception

# modify spec to relut in N_MOVES + 1 strips with heavy overlap
scan_center = (scan_range.max + scan_range.min) / 2
extra = N_MOVES * TRANSLATION
scan_range._min = units.Position(scan_center - spec.line_width / 2 - extra/2)
scan_range._max = units.Position(scan_center + spec.line_width / 2 + extra/2)

spec.strip_overlap = 1 - TRANSLATION / spec.line_width

# Set up pipeline
acquisition     = diri.make_acquisition("line_camera_stitched", spec=spec)
line_processor  = diri.make_processor("line_camera_line", upstream=acquisition)
strip_processor = diri.make_processor("strip", upstream=line_processor)
writer          = diri.make_writer("strip_translation_calibration", upstream=strip_processor)

# raw_writer      = diri.make_writer("tiff", upstream=strip_processor)
# raw_writer.frames_per_file = 100


acquisition.start()
strip_processor.join()