import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from scipy import ndimage

from ..diagnostic import Diagnostic
from ..utils.image_proc import ImageProc
from ..utils.general import dict_update
from ..utils.plotting import *

class ScintillatorArray(Diagnostic):
    """
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return
    
    def get_proc_shot(self, shot_dict, calib_id=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        """
        # TO DO: SHOULD ALOT OF THIS BE IN DIAGNOSTIC CLASS? ALOT COULD APPLY TO ANY IMAGE PROCESSING

        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            self.calib_dict = self.get_calib(shot_dict)

        img_data = self.get_shot_data(shot_dict)

        # spatial transform?
        if 'transform' in self.calib_dict:
            img_data, x, y = self.transform(img_data, self.calib_dict['transform'])
            self.curr_img = img_data
            self.x_mm = x
            self.y_mm = y

        # OBVIOUSLY ROIS (AND FOR BKGS ETC.) ARE DEPENDENT UPON TRANSFORM OR NOT?

        if 'bkg_type' in self.calib_dict:
            if self.calib_dict['bkg_type'] == 'flat':
                if 'bkg_roi' in self.calib_dict:
                    bkg_roi = self.calib_dict['bkg_roi']
                    bkg_value = np.mean(img_data[bkg_roi[0][1]:bkg_roi[1][1],bkg_roi[0][0]:bkg_roi[1][0]])
                    img_data = img_data - bkg_value
                else:
                    print(f"{self.config['name']}: No bkg_roi provided")
            else:
                print(f"{self.config['name']}: Unknown background correction type '{self.calib_dict['bkg_type']}'")

        if 'roi' in self.calib_dict:
            roi = self.calib_dict['roi']
            img_data = img_data[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]

        if 'img_rot' in self.calib_dict:
            img_data = ndimage.rotate(img_data, self.calib_dict['img_rot'], reshape=False)

        rows, cols = np.shape(img_data)
        self.x = np.arange(cols)
        self.y = np.arange(rows)

        if debug:
            scint_sigs = self.get_scint_sigs(img_data)
            plt.figure()
            im = plt.imshow(scint_sigs)
            cb = plt.colorbar(im)
    
        if 'transform' in self.calib_dict:
            return img_data, self.x_mm, self.y_mm
        else:
            return img_data, self.x, self.y
    
    #  # This could be in diagnostics class?
    # def make_calib(self, calib_id=None, save=False, view=True):
    #     """Master function for generating procssed portion of calibration file
    #         E.g transform, dispersion, etc. 
    #     """

    #     # Get calibration input
    #     self.calib_dict = self.get_calib(calib_id, no_proc=True)

    #     # Make a spatial transform (required)
    #     self.make_transform(self.calib_dict['transform'], view=view)

    #     # transfer other values
    #     # NOTE: we save the whole existing dictionary anyway...
    #     # save_vars = ['roi_mm','roi_MeV','roi_mrad']
    #     # for var in save_vars:
    #     #     if var in self.calib_dict:
    #     #         self.calib_dict[var] = self.calib_dict[var]

    #     # save the full calibration?
    #     if save:
    #         self.save_calib_file(self.calib_dict['proc_file'], self.calib_dict)

    #     return self.get_calib()
    
    #  # This is pretty much the same as ESpec function; could be in diagnostics class?
    # def make_transform(self, tcalib_input, view=False):
    #     """Generate a transform dictionary for use with spatially transforming raw shot images.
    #         This is a wrapper for ImageProc make_transform()

    #         tcalib_input: A dictionary containing the required information for the transform, or calibration file/id for loading...
    #                     Required dictionary keys; 
    #                         - tpoints; list of [X,Y], where the first pair is raw pixel, the next is the corresponding transform point, and repeat...
    #                         - raw_img; shot dictionary or filepath to raw untransformed calibration image 
    #                         - img_size_t; [X,Y] size of plane being transformed, in it's coords (mm?)
    #                         - img_size_px; [X,Y] new size of transformed image in pixels (can upsample)
    #                         - offsets; [X,Y] offset of plane being transformed, in it's coords (mm?)
    #                         - e_offsets; [X,Y] shift of transformed plane from electron beam axis
    #                     Optional dictionary keys; description, notes
    #         save_path:
    #         view:
    #     """

    #     # points are (by convention) passed in a list of [X,Y], where the first is in the pixel point, 
    #     # the next is the corresponding transform point, and repeat
    #     # so here we pick out every other value for the appropriate seperate arrays
    #     points = np.array(tcalib_input['tpoints'])
    #     p_px, p_t =  points[::2], points[1::2]

    #     # get raw image using shot dictionary or filepath
    #     raw_img = self.get_shot_data(tcalib_input['raw_img'])

    #     # optionals?
    #     if 'description' in tcalib_input:
    #         description = tcalib_input['description']
    #     else:
    #         description = ''
    #     if 'notes' in tcalib_input:
    #         notes = tcalib_input['notes']
    #     else:
    #         notes = ''

    #     # Use image processing library to generate a transform dictionary 
    #     img = ImageProc(raw_img)

    #     if self.calib_dict is None:
    #         self.calib_dict = {}
    #     if 'transform' not in self.calib_dict:
    #         self.calib_dict['transform'] = {}

    #     # update dictionary with new dictionary values
    #     dict_update(self.calib_dict['transform'], img.make_transform(p_px, p_t, tcalib_input['img_size_t'], tcalib_input['img_size_px'], 
    #                                     tcalib_input['offsets'], notes=notes, description=description))
        
    #     # ?? Add beam axis offset??
    #     self.calib_dict['transform']['offsets'] = tcalib_input['offsets']

    #     # perform transform to check
    #     timg, tx, ty = self.transform(raw_img)

    #     # save current processed image to object along with x and y values
    #     self.curr_img = timg
    #     self.x_mm = tx
    #     self.y_mm = ty

    #     if view:
    #         # if viewing, plot raw image
    #         plt.figure()
    #         im = plt.imshow(raw_img)
    #         plt.plot(p_px[:,0],p_px[:,1],'r+')
    #         cb = plt.colorbar(im)
    #         cb.set_label('Counts on CCD', rotation=270, labelpad=20)
    #         plt.title(description)
    #         plt.xlabel('pixels')
    #         plt.ylabel('pixels')
    #         plt.tight_layout()
    #         plt.show(block=False)
    #         # then plot transformed
    #         plt.figure()
    #         im = plt.imshow(timg, extent= (np.min(self.x_mm), np.max(self.x_mm), np.max(self.y_mm), np.min(self.y_mm)))
    #         plt.plot(p_t[:,0]-tcalib_input['e_offsets'][0],p_t[:,1]-tcalib_input['e_offsets'][1],'r+')
    #         cb = plt.colorbar(im)
    #         cb.set_label('Counts on CCD', rotation=270, labelpad=20)
    #         plt.title(description)
    #         plt.xlabel('mm')
    #         plt.ylabel('mm')
    #         plt.tight_layout()
    #         plt.show(block=False)

    #     return self.calib_dict['transform']
    
    # def transform(self, img_data, tform_dict=None):
    #     """"""
    #     # This is pretty much the same as ESpec function; could be in diagnostics class?

    #     # if not passed, use stored tform_dict, or complain
    #     if tform_dict is None:
    #         if self.calib_dict['transform'] is None:
    #             print('Scintillator Array Error, transform dictionary needs to be passed or loaded')
    #             return
    #     else:
    #         self.calib_dict['transform'] = tform_dict

    #     # if img_data is passed as a shot dictionary or filepath, grab the actual image
    #     if isinstance(img_data, dict) or isinstance(img_data, str):
    #         img_data = self.get_shot_data(img_data)

    #     img = ImageProc(img_data)
    #     timg, tx, ty = img.transform(self.calib_dict['transform'])

    #     # offset? shifts the xy cords on transformed screen
    #     x = tx - self.calib_dict['transform']['offsets'][0]
    #     y = ty - self.calib_dict['transform']['offsets'][1]

    #     return timg, x, y
    
    def get_scint_sigs(self, img_data, debug=False):

        scint_sigs = np.zeros([self.calib_dict['num_cols'],self.calib_dict['num_rows']])

        scint_centres_x = self.calib_dict['ext_start_x'] + self.calib_dict['ext_spacing_x'] * np.array(range(self.calib_dict['num_cols'])) 
        scint_centres_y = self.calib_dict['ext_start_y'] + self.calib_dict['ext_spacing_y'] * np.array(range(self.calib_dict['num_rows'])) 

        wr = int(self.calib_dict['ext_sample_width']/2)
        hr = int(self.calib_dict['ext_sample_height']/2)

        # TODO: Would be better to integrate signal in crystal masked area around each of these points
        # i.e. get all the signal from a crystal, then give average counts per pixel?
        if(debug):
            plt.figure()
            im = plt.imshow(img_data)
            ax = plt.gca()

        for ci in range(len(scint_centres_x)):
            for ri in range(len(scint_centres_y)):
                x = scint_centres_x[ci]
                y = scint_centres_y[ri]
                scint_sigs[ci,ri] = np.mean(img_data[y-hr:y+hr,x-wr:x+wr])
                if(debug):
                    rect = patches.Rectangle((x-wr, y-hr), 2*wr, 2*hr, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        if(debug): cb = plt.colorbar(im)

        mm2_per_px = self.calib_dict['scale_x'] * self.calib_dict['scale_y']

        # returning average counts per mm2 of scintalator?
        return scint_sigs.T / mm2_per_px

    # ------------------------------------------------------ #
    # PLOTTING FUNCTIONS
    # TODO: Move some of this to shared plotting class
    # ------------------------------------------------------ #

    def montage(self, timeframe, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=True, num_rows=1):

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:
            img, x, y = self.get_proc_shot(shot_dict)

            if 'images' in locals():
                images = np.concatenate((images, np.atleast_3d(img)), axis=2)
            else:
                images = np.atleast_3d(img)

            if 'burst' in shot_dict:
                m = re.search(r'\d+$', str(shot_dict['burst'])) # gets last numbers
                burst = int(m.group())
                burst_str = str(burst) + '|'
            else:
                burst_str = ''
            if 'shotnum' in shot_dict:
                shot_str = str(shot_dict['shotnum'])
            else:
                shot_str = ''
            shot_labels.append(burst_str + shot_str)

        # or y_MeV?
        fig, ax = plot_montage(images, x_roi=x_roi, y_roi=y_roi, x_downsample=x_downsample, y_downsample=y_downsample, 
                               title=self.shot_string(timeframe), vmin=vmin, vmax=vmax, 
                               transpose=transpose, num_rows=num_rows, shot_labels=shot_labels)
        #ax.set_ylabel(r'$E$ [MeV]')

        return fig, ax
    
        # def plot_histogram(self, timeframe, num_bins=100):

    #     shot_dicts = self.DAQ.get_shot_dicts(self.diag_name, timeframe)

    #     raw_data = []
    #     for shot_dict in shot_dicts:
    #         print(shot_dict)
    #         raw_data.append(self.get_shot_data(shot_dict))

    #     # plt.figure()
    #     # plt.imshow(raw_img)
    #     # plt.show(block=False)

    #     #print(len(np.array(raw_data).flatten()))

    #     plt.figure()
    #     plt.hist(np.array(raw_data).flatten(), bins=num_bins, log=True)
    #     plt.show(block=False)

    #    return