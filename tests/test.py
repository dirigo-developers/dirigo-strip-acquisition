from dirigo.main import Dirigo


diri = Dirigo()
    
acquisition = diri.acquisition_factory('line_scan_camera_strip')
display = diri.display_factory(acquisition=acquisition)
logging = diri.logger_factory(acquisition=acquisition)

# Connect workers
acquisition.add_subscriber(display)
acquisition.add_subscriber(logging)

# start workers
display.start()
logging.start()
acquisition.start()

acquisition.join(timeout=100.0)
display.stop()

print("Acquisition complete")
