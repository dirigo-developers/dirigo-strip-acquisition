from dirigo.sw_interfaces.acquisition import Acquisition

from dirigo_strip_acquisition.acquisitions import LineCameraStripAcquisition


class StageTranslationCalibration(Acquisition):
    """
    Translates the stage between capturing frames. Can be used with small 
    translations to estimate distortion field or large displacements to estimate
    frame size / angle relative the stage
    """
    Spec: Type[StageTranslationCalibrationSpec] = StageTranslationCalibrationSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec, 
                         thread_name="Line distortion calibration")

        if isinstance(spec, LineScanCameraAcquisition)

        self._strip_acquisition = LineCameraStripAcquisition(self.hw, 
                                                                 system_config, 
                                                                 self.spec)
        self._strip_acquisition.add_subscriber(self)

        self.runtime_info = self._strip_acquisition.runtime_info



        if "digitizer" in self.system_config:
        self.digitizer_profile = self._strip_acquisition.digitizer_profile
        

        assert self.system_config.fast_raster_scanner is not None
        fast_axis = self.system_config.fast_raster_scanner['axis']
        if fast_axis == "x":
            self._fast_stage = self.hw.stages.x
        else:
            self._fast_stage = self.hw.stages.y


        self._original_position = self._fast_stage.position