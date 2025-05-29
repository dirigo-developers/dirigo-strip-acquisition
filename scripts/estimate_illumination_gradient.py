from dirigo.plugins.processors import RasterFrameProcessor

from dirigo_strip_acquisition.processors import StripProcessor
from dirigo_strip_acquisition.analysis import StripAcquisitionLoader, SignalGradientLogger




fn = r"C:\Users\MIT\Documents\Dirigo\experiment.tif"

loader = StripAcquisitionLoader(fn)
frame_processor = RasterFrameProcessor(upstream=loader)
strip_processor = StripProcessor(upstream=frame_processor)
logger = SignalGradientLogger(upstream=strip_processor)

# Connect and start child threads
loader.add_subscriber(frame_processor)
frame_processor.add_subscriber(strip_processor)
strip_processor.add_subscriber(logger)

frame_processor.start()
strip_processor.start()
logger.start()

# start by starting (data) Loader thread
loader.start()
logger.join(30)