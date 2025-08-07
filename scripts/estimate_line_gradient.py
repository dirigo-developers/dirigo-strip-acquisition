"""
Use an existing dataset to estimate the line gradient (vignetting) function.

Tips: works best on relatively homogenous bland tissue samples (e.g. a large 
chunk of endometrium)
"""

from dirigo.plugins.processors import RasterFrameProcessor
from dirigo_strip_acquisition.analysis import StripAcquisitionLoader, SignalGradientLogger



if __name__ == "__main__":
    fn = r"D:\dirigo test data\gyn3_scan_raw_0.tif"

    loader          = StripAcquisitionLoader(fn)
    processor       = RasterFrameProcessor(upstream=loader)
    gradient_logger = SignalGradientLogger(upstream=processor)
    gradient_logger.show_results = True

    loader.add_subscriber(processor)
    processor.add_subscriber(gradient_logger)

    gradient_logger.start()
    processor.start()
    loader.start()  # <-- kicks everything off

    gradient_logger.join(50)