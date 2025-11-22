import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy import fft

from dirigo import io, units
from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.writer import Writer
from dirigo.sw_interfaces.processor import ProcessorProduct

from dirigo_strip_acquisition import LineCameraStitchedAcquisition



class StripTranslationCalibrationWriter(Writer):
    UPSAMPLE_X     = 20
    UPSAMPLE_Y     = 1
    EPS            = 1e-1
    PATCH          = 96
    STRIDE         = 16

    def __init__(self, upstream):
        super().__init__(upstream)
        self._acq: LineCameraStitchedAcquisition
        #self._processor: RasterFrameProcessor

        self.basename = "line_distortion_calibration"
        self.filepath = io.config_path() / "optics" / (self.basename + ".csv")
        self.data_filepath = io.data_path() / (self.basename + "_data.csv")

    def _receive_product(self) -> ProcessorProduct: 
        return super()._receive_product() # type: ignore

    def _work(self):
        # Collects strips
        self._strips =  [] 
        try:
            while True:
                with self._receive_product() as product:
                    self._strips.append(product.data[:,:,0].copy())

        except EndOfStream:
            self._publish(None) # forward sentinel
            self.save_data()

    def save_data(self):
        spec = self._acq.spec # for brevity

        n_s = len(self._strips)
        n_comparisons = n_s - 1
        n_y, n_x = self._strips[0].shape
        yc = n_y//2
        yr = n_y//4

        # Create: dx_true/dx_observed
        translation = (1-spec.strip_overlap) * spec.line_width
        dx_true = round(translation / spec.pixel_size)
        dx_observed = np.zeros(shape=(n_x // self.STRIDE, n_comparisons))
        field_position = np.zeros(n_x // self.STRIDE)

        ref_strip = self._strips[0]
        for s_idx, strip in enumerate(self._strips[1:]): # strip index

            for p_idx in range(n_x // self.STRIDE): # patch index
                p0 = (p_idx * self.STRIDE) # ref patch start pixel index
                ref_patch = ref_strip[(yc-yr):(yc+yr), p0:(p0 + self.PATCH)]

                m0 = p0 - dx_true # mov patch start pixel index
                if (m0 < 0) or (p0+self.PATCH >= n_x) or (m0+self.PATCH) >= n_x: 
                    dx_observed[p_idx, s_idx] = np.nan
                    field_position[p_idx] = np.nan
                    continue
                mov_patch = strip[(yc-yr):(yc+yr), m0:(m0 + self.PATCH)]

                _, j = self.x_corr(ref_patch, mov_patch)
                print(units.Position(j * spec.pixel_size))

                dx_observed[p_idx, s_idx] = j + dx_true
                ref_patch_center = p0 + self.PATCH//2
                mov_patch_center = m0 + self.PATCH//2
                comp_center = (ref_patch_center + mov_patch_center) / 2
                field_position[p_idx] = comp_center

            ref_strip = strip


        # Fit error
        field_positions = np.tile(field_position[:,np.newaxis], (1, n_comparisons)) 

        nan_mask = np.isnan(dx_observed)
        pfit: Polynomial = Polynomial.fit(
            x=field_positions[~nan_mask].ravel(),
            y=(dx_true/dx_observed)[~nan_mask].ravel(),
            deg=2
        )
        c0, c1, c2 = pfit.convert().coef
        a=1

        
    @classmethod
    def x_corr(cls, ref_frame: np.ndarray, moving_frame: np.ndarray):
        n_y, n_x = ref_frame.shape

        R = fft.rfft2(ref_frame,    workers=-1)
        M = fft.rfft2(moving_frame, workers=-1)
        xps = R * np.conj(M)
        s = (n_y * cls.UPSAMPLE_Y, n_x * cls.UPSAMPLE_X)
        corr = fft.irfft2(xps / (np.abs(xps) + cls.EPS), s, workers=-1)
        arg_max = np.argmax(corr)
        i = int(arg_max // corr.shape[1])
        j = int(arg_max %  corr.shape[1])

        if i > (s[0] // 2):  # Handle wrap-around for negative shifts
            i -= s[0]
        if j > (s[1] // 2): 
            j -= s[1]

        return i / cls.UPSAMPLE_Y, j / cls.UPSAMPLE_X