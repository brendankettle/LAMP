import os
#from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from skimage.morphology import reconstruction
import re
from LAMP.utils.io import *
from LAMP.utils.general import dict_update
from LAMP.utils.image_proc import ImageProc
from LAMP.utils.plotting import plot_montage
from LAMP.utils.plotting import get_colormap


class Diagnostic():
    """Base class for Diagnostics. 
    Currently this mostly handles loading/saving calibrations.
    """

    calib_id = None
    calib_dict = {} # filled when calibrations are loaded
    calib_dict_fixed = {} # overwrites / persists when new calibration are loaded
    calib_start = 0 # for keeping track of calibration in shot timeline
    calib_end = 0 # for keeping track of calibration in shot timeline

    def __init__(self, exp_obj, config):
        self.ex = exp_obj # pass in experiment object
        self.DAQ = self.ex.DAQ # shortcut to DAQ
        # ToDo: if diagnostic config filepath passed as opposed to array, go get config
        # self.load_config(config_filepath)
        self.config = config
        self.config['data_type'] = self.data_type # need to define data type in child class. I.e image or text?
        return

    def __repr__(self):
        return f"{self.config['type']}(name='{self.config['name']}')"

    def get_shot_data(self, shot_dict):
        """Wrapper for getting shot data through DAQ"""
        return self.DAQ.get_shot_data(self.config['name'], shot_dict)
    
    def set_calib(self, calib_id):
        self.calib_dict = self.get_calib(calib_id)
        return

    def get_calib(self, calib_id=None, no_proc=False):
        """Take a calibration id of some form, and return calibration dictionary.
            - None: Try and use pre-saved calibration dict within object
            - Dictionary: Assume a shot dictionary, and look for configuration using dates
            - (string) ID String:  Look for a dictionary within the master calibration file, with this ID
            - (string) Filepath: Look for a dictionary within a calibration file?
        """
        # If none passed, try use pre-saved calib 
        if calib_id is None:
            if self.calib_dict:
                calib_dict = self.calib_dict
                return calib_dict # returning now, not trying to load processed file
            else:
                print('get_calib() error; None passed and no calibration loaded yet')
                return None

        # passing dictionary? assume shot dictionary and try load using dates
        if isinstance(calib_id, dict):
            shot_dict = calib_id
            shot_time = self.DAQ.build_time_point(shot_dict)
            # are we already using the correct calibration? Then return dict so we don't keep loading
            if shot_time >= self.calib_start and shot_time <= self.calib_end:
                calib_dict = self.calib_dict
            else:
                # load all calibrations in master file
                all_calibs = self.load_calib_file(self.config['calib_file'])
                # loop through calibrations and 
                calib_dict = None
                for this_calib_id in all_calibs:
                    if 'start' in all_calibs[this_calib_id]:
                        start_shot_dict  = all_calibs[this_calib_id]['start']
                        self.calib_start  = self.DAQ.build_time_point(start_shot_dict)
                        end_shot_dict = all_calibs[this_calib_id]['end']
                        self.calib_end  = self.DAQ.build_time_point(end_shot_dict)
                        if shot_time >= self.calib_start and shot_time <= self.calib_end:
                            calib_dict = all_calibs[this_calib_id]
                            self.calib_id = this_calib_id
                            #print(f'Using Calibration: {this_calib_id}') # for debugging...
                            break
                if not calib_dict:
                    print("get_calib() error; Could not place shot in calibration timeline")        
                    return None

        # passing string?
        if isinstance(calib_id, str):
            # let's look for a file first
            if os.path.exists(self.build_calib_filepath(calib_id)):
                calib_dict = self.load_calib_file(calib_id)
            # no file, so let's look for ID key within the master calibration input file (if set in config)
            elif 'calib_file' in self.config:
                all_calibs = self.load_calib_file(self.config['calib_file'])
                if calib_id in all_calibs:
                    calib_dict = all_calibs[calib_id]
                else:
                    # default calibration id set?
                    if 'calib_default' in self.config:
                        print(f"Using default calibration: {self.config['calib_default']}")
                        calib_dict = all_calibs[self.config['calib_default']]
                    else:
                        print(f"get_calib() error; No calibration input ID found for {calib_id} in master calib input file")
                        return None
            else:
                print(f"get_calib() error; Unknown calibration found for {calib_id} - No calib_file set in diagnostics.toml?")
                return None
            self.calib_id = calib_id

        # before returning, if processed file is set, try load it and return contents with dictionary
        if not no_proc:
            if 'proc_file' in calib_dict:
                if os.path.exists(self.build_calib_filepath(calib_dict['proc_file'])):
                    proc_calib_dict = self.load_calib_file(calib_dict['proc_file'])
                    #dict_update(calib_dict, proc_calib_dict)
                    # actually, treat calibration file details as highest priorities, so they overwrite anything in processed dictionary
                    calib_dict = dict_update(proc_calib_dict, calib_dict)
                # might not be processed yet, just print a warning and move on
                else:
                    print(f"get_calib() warning; no processed file found for '{calib_dict['proc_file']}'")

        # load persistent calibs
        for param in self.calib_dict_fixed:
            # if self.calib_dict_fixed[param] is dict:
            #     # setting a whole subset of values?
            #     for sub_param in self.calib_dict_fixed[param]:
            #         self.calib_dict[param][sub_param] = self.calib_dict_fixed[param][sub_param]
            # else:
            # This should still work if param is a dictionary?
            calib_dict[param] = self.calib_dict_fixed[param]

        return calib_dict
    
    def fix_calib(self, param, value, remove=False):
        """This is to add to or overwrite a param in the calibration dictionary, keeping it persistent.
        An important difference here is that this will stay if a new calibration is loaded.
        Setting remove=True will no longer keep the calibration value as persistent. 
        You can either set one last time (using param and value), or it will if value=False, it will be removed completely."""

        if not remove:
            # update current dict
            # this should still work if "value" is a dictionary of sub params?
            self.calib_dict[param] = value 
            # and keep track of it as fixed param
            self.calib_dict_fixed[param] = value
        # removing?
        else:
            del self.calib_dict_fixed[param] # no longer persistent
            if value is False:
                del self.calib_dict[param] # delete param completely
            else:
                self.calib_dict[param] = value # set one last time
        return

    def build_calib_filepath(self, filename):
        if 'calib_subfolder' in self.config:
            calib_subfolder = self.config['calib_subfolder']
        else:
            calib_subfolder = ''
        return Path(os.path.join(self.ex.config['paths']['root'],self.ex.config['paths']['calibs_folder'], calib_subfolder, filename))

    def file_exists(self, shot_dict):
        return self.DAQ.file_exists(self.config['name'], shot_dict)

    def load_calib_file(self, filename, file_type=None, options=None):
        calibs  = load_file(self.build_calib_filepath(filename), file_type=file_type, options=options)
        return calibs

    def save_calib_file(self, filename, calib_data, file_type=None, options=None):
        save_file(self.build_calib_filepath(filename), calib_data, file_type=file_type, options=options)
        return
    
    def list_calibs(self):
        all_calibs = self.load_calib_file(self.config['calib_file'])
        return list(all_calibs.keys())

    def make_calib(self, calib_id=None, save=False, debug=True):
        """Master function for generating procssed portion of calibration file
            E.g transform, dispersion, etc. 
        """

        # Get calibration input
        self.calib_dict = self.get_calib(calib_id, no_proc=True)

        # Apply spatial transform?
        if 'transform' in self.calib_dict:
            self.make_transform(self.calib_dict['transform'], debug=debug)

        # Apply dispersion?
        # the function would be defined in the actual diagnostic module.
        if 'dispersion' in self.calib_dict:
            self.make_dispersion(self.calib_dict['dispersion'], debug=debug)

        # Apply divergence?
        # the function would be defined in the actual diagnostic module.
        if 'divergence' in self.calib_dict:
            self.make_divergence(self.calib_dict['divergence'], debug=debug)

        # save the full calibration?
        if save:
            if 'proc_file' not in self.calib_dict:
                print('Error, proc_file variable required in calibration if saving.')
            self.save_calib_file(self.calib_dict['proc_file'], self.calib_dict)

        return self.get_calib()

    def get_proc_shot(self, shot_dict, calib_id=None, debug=False):
        """Return a processed shot using saved or passed calibrations.
        This can be wrapped by the child function for added functionality.
        """

        # set calibration dictionary
        if calib_id:
            self.calib_dict = self.get_calib(calib_id)
        else:
            # if not self.calib_dict or not len(self.calib_dict):
            #     self.calib_dict = self.get_calib(shot_dict)
            self.calib_dict = self.get_calib(shot_dict)

        # TO DO: check if image type, and if not, skip this... although you probably won't call this function if not image?
        # do standard image calibration. Transforms, background, ROIs etc.
        img, x, y = self.run_img_calib(shot_dict, debug=debug)

        self.curr_img = img
        self.x = x
        self.y = y
        
        return img, x, y
   
    def check_calib(self, shot_dict=None, print_all=False):
        """Wrapper function for checking calibration and img calibration"""
        if shot_dict:
            self.get_proc_shot(shot_dict, debug=True)
        print(f'Using Calibration ID: {self.calib_id}')
        if print_all:
            print(self.calib_dict)
        return 

    def run_img_calib(self, img_data, debug=False):
        """Central wrapper function for processing image data using calibration"""

        # background correction wrapper called at the appropriate stage
        # this could definitely be nicer... use kwargs
        def do_bkg_sub(calib_label='background'):
            if 'type' not in self.calib_dict[calib_label]:
                print(f'run_img_calib() error: {calib_label} error: No type set')
            if 'axis' in self.calib_dict[calib_label]:
                axis = self.calib_dict[calib_label]['axis']
            else:
                axis = None
            if 'order' in self.calib_dict[calib_label]:
                order = self.calib_dict[calib_label]['order']
            else:
                order = None
            if 'options' in self.calib_dict[calib_label]:
                options = self.calib_dict[calib_label]['options']
            else:
                options = None
            if 'data' in self.calib_dict[calib_label]:
                bkg_data = self.calib_dict[calib_label]['data']
            else:
                bkg_data = None
            if 'roi' in self.calib_dict[calib_label]:
                bkg_roi = self.calib_dict[calib_label]['roi']
            else:
                bkg_roi = None
            img.bkg_sub(self.calib_dict[calib_label]['type'], roi=bkg_roi, axis=axis, order=order, data=bkg_data, options=options, debug=debug)


        # if img_data is passed as a shot dictionary or filepath, grab the actual image
        if isinstance(img_data, dict) or isinstance(img_data, str):
            shot_dict = img_data
            img_data = self.get_shot_data(img_data)
            if img_data is None: # no file found
                return None, None, None

        img = ImageProc(data=img_data)

        if 'dark' in self.calib_dict and self.calib_dict['dark'] is not False:
            shot_dicts = self.DAQ.get_shot_dicts(self.config['name'], self.calib_dict['dark']['data']) # darks should be a timeline dict
            # now we have the shot dictionary, check if it's the same as previously loaded, and if so, return saved dark
            dark_img = np.array([])
            if hasattr(self, 'dark_shot_dicts') and hasattr(self, 'dark_img'):
                if shot_dicts == self.dark_shot_dicts:
                    dark_img = self.dark_img
            if not dark_img.any():
                # loop through all shots, and build average dark
                num_shots = 0
                for shot_dict in shot_dicts:
                    if 'sum_img' in locals():
                        sum_img += self.get_shot_data(shot_dict)
                    else:
                        sum_img = self.get_shot_data(shot_dict)
                    num_shots += 1
                dark_img = sum_img / num_shots
                self.dark_img = dark_img
                self.dark_shot_dicts = shot_dicts
            img.subtract(dark_img)

        if 'blob_filter' in self.calib_dict and self.calib_dict['blob_filter'] is not False:
            kwargs = {}
            for param in ['threshold', 'size', 'img_max', 'debug']:
                if param in self.calib_dict['blob_filter']: 
                    kwargs[param] = self.calib_dict['blob_filter'][param]
            _, num_blobs = img.blob_filter(**kwargs)
            if num_blobs> 0 and (debug or ('debug' in self.calib_dict['blob_filter'] and self.calib_dict['blob_filter']['debug'])):
                if 'shot_dict' in locals():
                    shot_dict_str = f'({shot_dict})'
                print(f'{num_blobs} removed. {shot_dict_str}')

        if 'median_filter' in self.calib_dict and self.calib_dict['median_filter'] is not False:
            if 'stage' in self.calib_dict['median_filter'] and self.calib_dict['median_filter']['stage'].lower() == 'original':
                img.median_filter(size=self.calib_dict['median_filter']['size'])

        if 'background' in self.calib_dict and self.calib_dict['background'] is not False:
            if 'stage' in self.calib_dict['background'] and self.calib_dict['background']['stage'].lower() == 'original':
                do_bkg_sub()
        if 'background2' in self.calib_dict and self.calib_dict['background2'] is not False:
            if 'stage' in self.calib_dict['background2'] and self.calib_dict['background2']['stage'].lower() == 'original':
                do_bkg_sub('background2')

        # FIX!
        # currently now transition back to data, not ImageProc object, but this should be fixed!
        img_data = img.get_img()

        # ROIs for original data
        if 'roi' in self.calib_dict and 'stage' in self.calib_dict['roi'] and self.calib_dict['roi']['stage'].lower() == 'original':
            if 'pixels' in self.calib_dict['roi']:
                roi = self.calib_dict['roi']['pixels']
                img_data = img_data[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        #     if 'transformed' in self.calib_dict['roi']:
        #         # this will probably have to go later? (after transform)
        #         print('To Do! ROI processing for transformed Co-ords... etc.')

        if 'transform' in self.calib_dict and self.calib_dict['transform'] is not False:
            img_data, x, y = self.transform(img_data, self.calib_dict['transform'])
            if debug:
                plt.figure()
                im = plt.imshow(img_data, vmin=np.percentile(img_data, 5), vmax=np.percentile(img_data, 99))
                cb = plt.colorbar(im)
                plt.xlabel('new pixels')
                plt.ylabel('new pixels')
                plt.tight_layout()
                plt.show(block=False)
        else:
            # non-transform functions
            if 'img_rotation' in self.calib_dict and self.calib_dict['img_rotation'] is not False:
                img_data = ndimage.rotate(img_data, self.calib_dict['img_rotation'], reshape=False)

            if 'flipud' in self.calib_dict and self.calib_dict['flipud']:
                img_data = np.flipud(img_data)
            if 'fliplr' in self.calib_dict and self.calib_dict['fliplr']:
                img_data = np.fliplr(img_data)

            if 'scale' in self.calib_dict:
                x = np.arange(np.shape(img_data)[1]) * self.calib_dict['scale']['pixel_width']
                y = np.arange(np.shape(img_data)[0]) * self.calib_dict['scale']['pixel_height']
                if 'units' in self.calib_dict['scale']:
                    self.x_units = self.calib_dict['scale']['units']
                    self.y_units = self.calib_dict['scale']['units']
                if 'x_units' in self.calib_dict['scale']:
                    self.x_units = self.calib_dict['scale']['x_units']
                if 'y_units' in self.calib_dict['scale']:
                    self.x_units = self.calib_dict['scale']['y_units']

        # below useful for example when using fibre bundles
        # uses morphology package of skimage. Can also find peaks?
        if 'fill_gaps' in self.calib_dict and self.calib_dict['fill_gaps']:
            seed = np.copy(img_data)
            seed[1:-1, 1:-1] = img_data.max()
            mask = np.copy(img_data)
            img_data = reconstruction(seed, mask, method='erosion')

        # change orientation? landscape or portrait
        if 'orientate' in self.calib_dict and self.calib_dict['orientate'] is not False:
            if self.calib_dict['orientate'].lower() == 'transpose':
                img_data = np.transpose(img_data)
            elif self.calib_dict['orientate'].lower() == 'acw':
                img_data = np.rot90(img_data, 1) # ndimage.rotate(img_data, 90, reshape=False)
            else:
                img_data = np.rot90(img_data, 3) # ndimage.rotate(img_data, 270, reshape=False)
            if 'y' in locals() and 'x' in locals():
                y_tmp = y
                y = x
                x = y_tmp

        # Fix! switching back to img object again...
        img = ImageProc(data=img_data)

        if 'median_filter' in self.calib_dict and self.calib_dict['median_filter'] is not False:
            if 'stage' in self.calib_dict['median_filter'] and self.calib_dict['median_filter']['stage'].lower() == 'transformed':
                img.median_filter(size=self.calib_dict['median_filter']['size'])

        if 'background' in self.calib_dict and self.calib_dict['background'] is not False:
            if 'stage' in self.calib_dict['background'] and self.calib_dict['background']['stage'].lower() == 'transformed':
                do_bkg_sub()
        if 'background2' in self.calib_dict and self.calib_dict['background2'] is not False:
            if 'stage' in self.calib_dict['background2'] and self.calib_dict['background2']['stage'].lower() == 'transformed':
                do_bkg_sub('background2')

        # if 'zero_cut' in self.calib_dict and self.calib_dict['zero_cut']:
        #     img_data = img.get_img()
        #     img_data[img_data<0] = 0
        #     img.set_img(img_data)

        # if x / y not set (i.e. no transforms etc.), use pixel numbers
        if 'x' not in locals():
            x = np.arange(np.shape(img.get_img())[1])
        if 'y' not in locals():
            y = np.arange(np.shape(img.get_img())[0])

        self.x = x
        self.y = y
        self.curr_img = img.get_img()

        return img.get_img(), x, y

    def transform(self, img_data, tform_dict=None):
        """Wrapper function around ImageProc transform"""

        # if not passed, use stored tform_dict, or complain
        if tform_dict is None:
            if self.calib_dict['transform'] is None:
                print('Error, transform dictionary needs to be passed or loaded')
                return
        else:
            self.calib_dict['transform'] = tform_dict

        # if img_data is passed as a shot dictionary or filepath, grab the actual image
        if isinstance(img_data, dict) or isinstance(img_data, str):
            img_data = self.get_shot_data(img_data)

        # just  double check not passing an ImageProc object already
        if isinstance(img_data, ImageProc):
            img = img_data
            #print('Passing ImageProc...')
        else:
            img = ImageProc(img_data)

        timg, tx, ty = img.transform(self.calib_dict['transform'])

        # offset? shifts the xy cords on transformed screen
        if 'e_offsets' in self.calib_dict['transform']: # backwards caompatiable with old ESpec calibrationss...
            self.calib_dict['transform']['zero_offsets'] = self.calib_dict['transform']['e_offsets']

        x = tx - self.calib_dict['transform']['zero_offsets'][0]
        y = ty - self.calib_dict['transform']['zero_offsets'][1]

        return timg, x, y
    
    def make_transform(self, tcalib_input, debug=False):
        """Generate a transform dictionary for use with spatially transforming raw shot images.
            This is a wrapper for ImageProc make_transform()

            tcalib_input: A dictionary containing the required information for the transform, or calibration file/id for loading...
                        Required dictionary keys; 
                            - tpoints; list of [X,Y], where the first pair is raw pixel, the next is the corresponding transform point, and repeat...
                            - raw_img; shot dictionary or filepath to raw untransformed calibration image 
                            - img_size_t; [X,Y] size of plane being transformed (and new transformed image), in it's coords (if real space image, mm?)
                            - img_size_px; [X,Y] new size of transformed image in pixels (can up/down sample)
                            - orig_offsets; [X,Y] offset of plane being transformed (original), in it's coords (mm?)
                            - zero_offsets; [X,Y] cords shift of resulting transformed plane (afterwards), in its coords (mm?)
                        Optional dictionary keys; description, notes
            save_path:
            debug:
        """

        # points are (by convention) passed in a list of [X,Y], where the first is in the pixel point, 
        # the next is the corresponding transform point, and repeat
        # so here we pick out every other value for the appropriate seperate arrays
        points = np.array(tcalib_input['tpoints'])
        p_px, p_t =  points[::2], points[1::2]
        if 'tpoints_shift' in tcalib_input:
            p_px = p_px + tcalib_input['tpoints_shift'][0]
            p_t = p_t + tcalib_input['tpoints_shift'][1]

        # get raw image using shot dictionary or filepath
        raw_img = self.get_shot_data(tcalib_input['raw_img'])

        # optionals?
        if 'description' in tcalib_input:
            description = tcalib_input['description']
        else:
            description = ''
        if 'notes' in tcalib_input:
            notes = tcalib_input['notes']
        else:
            notes = ''

        # Use image processing library to generate a transform dictionary 
        img = ImageProc(raw_img)

        if self.calib_dict is None:
            self.calib_dict = {}
        if 'transform' not in self.calib_dict:
            self.calib_dict['transform'] = {}

        # backwards compatability
        if 'offsets' in tcalib_input:
            tcalib_input['orig_offsets'] = tcalib_input['offsets']

        # no offset define? assume zero
        if 'orig_offsets' not in tcalib_input:
            tcalib_input['orig_offsets'] = [0,0]

        # update dictionary with new dictionary values
        dict_update(self.calib_dict['transform'], img.make_transform(p_px, p_t, tcalib_input['img_size_t'], tcalib_input['img_size_px'], 
                                        tcalib_input['orig_offsets'], notes=notes, description=description))

        # Add zero offset for new image
        if 'zero_offsets' in tcalib_input:
            self.calib_dict['transform']['zero_offsets'] = tcalib_input['zero_offsets']
        else:
            tcalib_input['zero_offsets'] = [0,0]

        # perform transform to check
        timg, tx, ty = self.transform(raw_img)

        # save current processed image to object along with x and y values
        self.curr_img = timg
        self.x_mm = tx
        self.y_mm = ty

        if debug:
            # if debugging, plot raw image
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
            # then plot transformed
            plt.figure()
            im = plt.imshow(timg, extent= (np.min(self.x_mm), np.max(self.x_mm), np.max(self.y_mm), np.min(self.y_mm)))
            plt.plot(p_t[:,0]-tcalib_input['zero_offsets'][0],p_t[:,1]-tcalib_input['zero_offsets'][1],'r+')
            cb = plt.colorbar(im)
            cb.set_label('Counts on CCD', rotation=270, labelpad=20)
            plt.title(description)
            plt.xlabel('mm?')
            plt.ylabel('mm?')
            plt.tight_layout()
            plt.show(block=False)

        return self.calib_dict['transform']
    
    def shot_string(self, shot_dict):
        self.DAQ.shot_string(shot_dict)
        return f"{self.config['name']}, {self.DAQ.shot_string(shot_dict)}"
    
    def get_integrated_signal(self, shot_dict, roi=None):
        imdata, x, y = self.get_proc_shot(shot_dict)
        if not roi:
            height, width = np.shape(imdata)
            roi = [[0,0],[width,height]]

        return np.sum(imdata[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]])
    
    def get_integrated_signals(self, timeframe, roi=None):

        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'], timeframe)

        int_data = []
        for shot_dict in shot_dicts:
            int_data.append(self.get_integrated_signal(shot_dict, roi=roi))
        
        return int_data

    def to_mm(self, value, units):
        if units.lower() == 'mm':
            return value
        elif units.lower() == 'cm':
            return (value * 10)
        elif units.lower() == 'm':
            return (value * 1e3)
        else:
            print(f"to_mm error; unknown spatial units {units}")

    def to_MeV(self, value, units):
        if units == 'MeV':
            return value
        elif units == 'GeV':
            return (value * 1e3)
        elif units == 'eV':
            return (value * 1e-3)
        else:
            print(f"to_MeV error; unknown spectral units {units}")

    def to_mrad(self, value, units):
        if units.lower() == 'mrad':
            return value
        elif units.lower() == 'rad':
            return (value * 1e3)
        elif units.lower() == 'deg':
            return (value * (np.pi() / 180) * 1e3)
        else:
            print(f"to_mrad error; unknown angular units {units}")

    def plot_proc_shot(self, shot_dict, calib_id=None, vmin=None, vmax=None, colormap='plasma', debug=False):

        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

        if vmin is None:
            vmin = np.nanmin(img)
        if vmax is None:
            #vmax = np.nanmax(img)
            vmax = np.percentile(img,99)

        fig = plt.figure()
        im = plt.pcolormesh(x, y, img, vmin=vmin, vmax=vmax, cmap=get_colormap(colormap), shading='auto')
        cb = plt.colorbar(im)
        #cb.set_label(self.make_units(self.img_units), rotation=270, labelpad=20)
        plt.title(self.shot_string(shot_dict))
        # ToDo: Use x_units or y_units var for labels
        plt.tight_layout()
        plt.show(block=False)

        return fig, plt.gca()
    
    def montage(self, timeframe, calib_id=None, x_roi=None, y_roi=None, x_downsample=1, y_downsample=1, exceptions=None, vmin=None, vmax=None, transpose=False, num_rows=1, axis_label = '', cb_label='', colormap='plasma', colormap_option=None, debug=False):
        """Default wrapper function. This can be overwritten by diagnostic with more options
        The diagnostic itslef needs to have a get_proc_shot() defined.
        Also the DAQ has to have get_shot_dicts()"""

        if calib_id:
            self.calib_dict = self.get_calib(calib_id)

        # if not self.calib_dict:
        #     print('Missing Calibration before using Montage...')
        #     return False

        # calling 'universal' DAQ function here, that is probably DAQ specific
        shot_dicts = self.DAQ.get_shot_dicts(self.config['name'],timeframe,exceptions=exceptions)

        shot_labels = []
        for shot_dict in shot_dicts:

            img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

            if 'images' in locals():
                images = np.concatenate((images, np.atleast_3d(img)), axis=2)
            else:
                images = np.atleast_3d(img)

            # try build a shot label
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

        if transpose:
            axis = y
        else:
            axis = x
        # cb_label = self.make_units(self.img_units)

        fig, ax = plot_montage(images, axis=axis, x_downsample=x_downsample, y_downsample=y_downsample, title=self.shot_string(timeframe), 
                               vmin=vmin, vmax=vmax, transpose=transpose, cb_label=cb_label, y_label=axis_label, num_rows=num_rows, shot_labels=shot_labels, colormap=colormap, colormap_option=colormap_option)

        return fig, ax