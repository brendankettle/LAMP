import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib

from .diagnostic import Diagnostic
from ..lib.image_proc import ImageProc

class ESpec(Diagnostic):
    """Electron (charged particle?) Spectrometer
    This can potentially expand to cover a lot more actions;
        - Montage creation (lib function?)
        - Two screen processing
        - Click tools for making spatial calibrations
        - Charge calibration?
        - Even maybe calculating trajectories and dispersions?
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    tform_dict = None

    def __init__(self, exp_obj, config_filepath):
        # pass in experiment object
        self.ex = exp_obj
        self.DAQ = self.ex.DAQ # shortcut to DAQ
        # Initiate parent base Diagnostic class to get all shared attributes and funcs
        super().__init__(config_filepath)
        return
    
    def get_shot_data(self, shot_dict, processed=False):
        """Wrapper for getting shot data through DAQ"""
        if processed:
            print('Need to implement applying calibrations...')
            # Need to call load calibration here... and will use shot_dict to determine?
        return self.DAQ.get_shot_data(self.config['setup']['name'], shot_dict)

    def make_transform(self, raw_img, points, img_size_t, img_size_px, offsets=[0,0], notes='', description='', save_path=None, view=False):

        # points are (by convention) passed in a list of [X,Y], where the first is in the pixel point, 
        # the next is the corresponding transform point, and repeat
        # so here we pick out every other value for the appropriate seperate arrays
        points = np.array(points)
        p_px, p_t =  points[::2], points[1::2]

        # if raw_img is passed as a shot dictionary, grab the actual image
        if isinstance(raw_img, dict):
            raw_img = self.get_shot_data(raw_img)

        if view:
            # plot raw image and points if checking
            plt.figure()
            im = plt.imshow(raw_img)
            plt.plot(p_px[:,0],p_px[:,1],'r+')
            cb = plt.colorbar(im)
            cb.set_label('Counts on CCD', rotation=270, labelpad=20)
            plt.title(description)
            plt.xlabel('pixels')
            plt.ylabel('pixels')
            plt.tight_layout()
            plt.show(block=False)

        # Use image processing library to generate a transform dictionary 
        img = ImageProc(raw_img)
        tform_dict = img.make_transform(p_px, p_t, img_size_t, img_size_px, offsets, notes, description)
        self.tform_dict = tform_dict

        if view:
            # if checking, perform transform and plot
            timg, tx, ty = self.transform(raw_img)
            plt.figure()
            im = plt.imshow(timg, extent= (np.min(tx), np.max(tx), np.max(ty), np.min(ty)))
            plt.plot(p_t[:,0],p_t[:,1],'r+')
            cb = plt.colorbar(im)
            cb.set_label('Counts on CCD', rotation=270, labelpad=20)
            plt.title(description)
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.tight_layout()
            plt.show(block=False)

        # TO DO: Save file
        if save_path:
            print('TO DO')

        # store with object
        self.tform_dict = tform_dict

        return tform_dict

    def transform(self, img_data, tform_dict=None):
        # if not passed, use stored tform_dict, or complain
        if tform_dict is None:
            if self.tform_dict is None:
                print('ESpec Error, transform dictionary needs to be passed or loaded')
                return
        else:
            self.tform_dict = tform_dict
        # if img_data is passed as a shot dictionary, grab the actual image
        if isinstance(img_data, dict):
            img_data = self.get_shot_data(img_data)
        img = ImageProc(img_data)
        timg, tx, ty = img.transform(self.tform_dict)
        return timg, tx, ty
    
    def load_transform(self, filepath):

        tform_dict = ''

        return tform_dict
    
    def load_calib_input(self, calib_id):
        """Wrapper for returning a calibraiton input"""
        if not self.config['setup']['calib_input']:
            print('ESpec Error, calibration input file not set in config')
            return
        calib_file = f"{self.ex.config['paths']['calibs_folder'].strip('./')}.{self.config['setup']['calib_folder'].strip('./')}.{self.config['setup']['calib_input'].strip('./')}"
        try:
            calib_lib = importlib.import_module(calib_file)
        except ImportError:
            raise Exception(f'Could not find calibration input file: {calib_file}')
        
        if calib_id in calib_lib.calib_input:
            return calib_lib.calib_input[calib_id]
        else:
            raise IndexError(f"ESpec calibration input error, could not find id {calib_id}")

    def load_calib(self, calib_filename):
        # TO DO: Pass some time frame and work out calibration file? 
        #print(f'Loading some calibration from file: {timestamp}')
        #print(self.ex.config['paths']['calib_folder'])
        return
