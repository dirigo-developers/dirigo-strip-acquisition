"""
Use an existing dataset to estimate the line gradient (vignetting) function.

Tips: works best on relatively homogenous bland tissue samples (e.g. a large 
chunk of endometrium)
"""

from dirigo.plugins.loaders import RawRasterFrameLoader
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo_strip_acquisition.acquisitions import RasterScanStitchedAcquisitionSpec
from dirigo_strip_acquisition.analysis import SignalGradientLogger



if __name__ == "__main__":
    fn = r"D:\dirigo-data\2001-P-001104\SD082721-1\SD082721-1_scan_4_raw.tif"

    loader          = RawRasterFrameLoader(
        file_path   = fn,
        spec_class  = RasterScanStitchedAcquisitionSpec
    )
    processor       = RasterFrameProcessor(upstream=loader)
    gradient_logger = SignalGradientLogger(upstream=processor)
    gradient_logger.show_results = True

    loader.add_subscriber(processor)
    processor.add_subscriber(gradient_logger)

    gradient_logger.start()
    processor.start()
    loader.start()  # <-- kicks everything off

    gradient_logger.join()

    gradient_logger.plot_results()