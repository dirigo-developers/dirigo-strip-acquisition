from dirigo.main import Dirigo



diri = Dirigo()

acquisition     = diri.make_acquisition("raster_scan_stitched", spec="raster_scan")

line_processor  = diri.make_processor("raster_frame", upstream=acquisition)
strip_processor = diri.make_processor("strip", upstream=line_processor)
strip_stitcher  = diri.make_processor("strip_stitcher", upstream=strip_processor)
tile_builder    = diri.make_processor("tile_builder", upstream=strip_stitcher)

pyramid_logger  = diri.make_logger("pyramid", 
                                   upstream=tile_builder,
                                   levels=(1,2,4,8,16,32))


acquisition.start()
pyramid_logger.join()