from dirigo.main import Dirigo


diri = Dirigo()
    
acquisition = diri.acquisition_factory('line_scan_camera_strip')
# processor = diri.processor_factory(acquisition)
# display = diri.display_factory(processor)
# logging = diri.logger_factory(processor)

# Connect threads
# acquisition.add_subscriber(processor)
# processor.add_subscriber(display)
# processor.add_subscriber(logging)

# processor.start()
# display.start()
# logging.start()
acquisition.start()

acquisition.join(timeout=100.0)
# processor.stop()

print("Acquisition complete")


# Do it again!
acquisition = diri.acquisition_factory('line_scan_camera_strip')
acquisition.start()
acquisition.join(timeout=100.0)