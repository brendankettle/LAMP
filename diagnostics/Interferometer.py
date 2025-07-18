import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LAMP.diagnostic import Diagnostic
from LAMP.utils.image_proc import ImageProc
from scipy.fft import fft2, fftshift
from skimage.measure import profile_line

class Interferometer(Diagnostic):
    """Interferometer
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return
    
    def fft_amp_spec(self, shot_dict):
        """"""
        img,x,y = self.get_proc_shot(shot_dict)
        if img is None:
            print(f'No image data for {shot_dict}')
            return None
        
        # The 2D Fourier transform of an image converts the image from the spatial domain (coordinates x,y, pixel intensities) 
        # into the frequency domain (u,v, representing spatial frequencies). 
        # [a representation of the image as a sum of sine and cosine waves at different spatial frequencies].
        # Each point in the 2D Fourier transform encodes a particular spatial frequency — 
        # essentially, how rapidly intensity varies in different directions across the image.
        # The 2D Fourier transform is complex-valued.
        # The magnitude of the Fourier transform (amplitude spectrum) tells you the strength of each frequency present
        # Low frequencies (near the center of the transform) represent large-scale structures and smooth variations.
        # High frequencies (towards the edges) represent fine details, edges, and noise
        # Images with repeating patterns produce sharp peaks at corresponding frequencies. 
        # The phase contains information about the position and structure of features in the image.
        fourier_img = fft2(img)
        # Shift the zero-frequency component to the center of the spectrum
        # and then get absolute value (from complex)
        fourier_amp = np.abs(fftshift(fourier_img))

        # Haidinger fringes?

        return fourier_amp

    def fringe_level(self, shot_dict, mask=None, line_pts=None, line_width=10, percentile_cut=99.9, debug=False):
        """"""

        fourier_amp = self.fft_amp_spec(shot_dict)
        if fourier_amp is None:
            return None

        # get rid of the very strong components? 
        if percentile_cut:
            high_cut = np.percentile(fourier_amp,percentile_cut)
            fourier_amp[fourier_amp>high_cut] = 0
        else:
            percentile_cut = 100

        # mask?
        if mask is not None:
            fourier_amp = fourier_amp * mask

        # histogram
        # plt.figure()
        # plt.hist(fourier_amp.flatten(),bins=1000, log=True)
        # plt.show(block=False)

        if debug:
            plt.figure()
            im = plt.imshow(fourier_amp, vmin=np.min(fourier_amp), vmax=np.percentile(fourier_amp,percentile_cut))
            cb = plt.colorbar(im)
            plt.tight_layout()

        # lineout?
        if line_pts is not None:
            x1 = line_pts[0][0] 
            y1 = line_pts[0][1] 
            x2 = line_pts[1][0] 
            y2 = line_pts[1][1] 
            lineout = profile_line(fourier_amp,[y1,x1],[y2,x2],linewidth=line_width)
            if debug:
                plt.plot([x1, x2], [y1, y2], color="red", linewidth=line_width, alpha=0.5) # this seems only approximate? not same "width" as above?
                plt.show(block=False)
                plt.figure()
                plt.plot(lineout)
                plt.show(block=False)
            return fourier_amp, np.mean(fourier_amp), lineout
        else:
            if debug:
                plt.show(block=False)
            return fourier_amp, np.mean(fourier_amp)