import numpy as np
import matplotlib.pyplot as plt 
from .diagnostic import Diagnostic
from ..utils.image_proc import ImageProc

# TODO: look at other previous code for filter calibration / spectrum fitting etc.
# TODO: Masking etc. should be in ImageProc
# TODO: And background correction!

class XrayFilterArray(Diagnostic):
    """X-ray filter array analysis. I.e. fitting an incident spectrum for E.g. betatron diagnostic
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_shot_img(self, shot_dict, subtract_dark=True, calib_id=None):
        """Return a shot image, by default with darks removed. Background?"""
        # get calibration dictionary?
        self.calib_input = self.get_calib_input(calib_id, shot_dict=shot_dict)
        raw_img = self.get_shot_data(shot_dict)
        img = ImageProc(raw_img)
        if subtract_dark:
            dark_img = self.make_dark(self.calib_input['darks'])
            img.subtract(dark_img)
        return img.get_img()
    
    def get_shot_counts(self, shot_dict, calib_id=None):
        """Return sum counts of a a shot
        """
        return np.sum(self.get_shot_img(shot_dict))
    
    # Could this be in base diagnostic class?
    def make_dark(self, shot_dict):
        # TODO: Could be single filepath rather than shot dictionarys. Do usual is dict check
        dark_date = shot_dict['date']
        dark_run = shot_dict['run']
        # single shot or whole run?
        if 'shotnum' in shot_dict:
            shot_dicts = [shot_dict]
        else:
            shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, {'date': dark_date, 'run': dark_run})
        # loop through all shots, and build average dark
        num_shots = 0
        for shot_dict in shot_dicts:
            if 'img' in locals():
                img += self.DAQ.get_shot_data(self.diag_name, shot_dict)
            else:
                img = self.DAQ.get_shot_data(self.diag_name, shot_dict)
            num_shots += 1
        # return average
        return img / num_shots
    
    # TODO: Save dark in calibration pickle?
    def make_calib(self):

        return

    def load_filter_specs(self):
        specs_file = self.config['filter_specs']
        filter_specs = self.load_calib_file(specs_file)
        # filter_label, filter_keys, filter_names, filter_widths, mass_density, filter_k_edges, uncertainty_in_filter_width, background_sub_filter, filter_backing_name, filter_backing_widths, filter_backing_mass_density
        # TODO: currently csv of old format; should update to JSON?
        
        return

    