import numpy as np
import cv2 as cv
from matplotlib.image import imread 
import matplotlib.pyplot as plt 
from pathlib import Path

"""NB: functions/modules should not be coupled into the DAQ or diagnostics; should be independent!"""

class ImageProc():
    """Class for handling the processing of image data. This includes:
        - Background correction
        - Spatial transforms
        - Masking
        - ?
    """

    img_data = None
    tform_dict = None

    def __init__(self, filepath=None, data=None):
        if data is not None:
            self.set_img(data)
        if not isinstance(filepath, str):
            data = filepath
            self.set_img(data) # assume data passed first? (not filepath string)
        elif filepath is not None:
            self.load_img(filepath)
        else:
            print('Error ImageProc: not sure if passing filepath or data')
        return
    
    def set_img(self, data):
        self.img_data = data
        self.width = np.shape(data)[1]
        self.height = np.shape(data)[0]
        return

    def get_img(self):
        return self.img_data
    
    def load_img(self, filepath):
        # Only if not using DAQ...
        #img = imread(Path(filepath)).astype(float)
        img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
        assert img is not None, f"file could not be read, check with os.path.exists(): {filepath}"
        self.set_img(img)
        return self.get_img()

    def subtract(self, img):
        self.img_data = self.img_data - img
        return self.img_data
    
    def resample(self, width=None, height=None, scale=None, interp=cv.INTER_CUBIC):
        # resize image and CONSERVE COUNTS
        # if scale factor, work out
        if scale is not None:
            new_width = int(scale*self.width)
            new_height = int(scale*self.height)
            px_rescale = 1 / (scale * scale)
        elif width is not None and height is not None:
            new_width = width
            new_height = height
            px_rescale = 1 / ((new_width/self.width) * (new_height/self.height))
        else:
            print('resample error: Scale or new Width & Height required')
            return None
        res_img = cv.resize(self.get_img(),(new_width, new_height), interpolation=interp)
        res_img = res_img * px_rescale
        self.set_img(res_img)
        return self.get_img()
    
    def bkg_sub(self, type, roi=None, axis=None, data=None, options=None, debug=False):
        # switch between background type
        if type.lower() == 'img':
            # subtract an image fed into function
            self.bkg = data
            self.set_img(self.get_img() - self.bkg)
        elif type.lower() == 'grad_fit':
            # fit a polynomial to an ROI average along one axis (gradient), and extrapolate across image
            # good for constant gradients in one direction

            # multiples ROIs?
            if len(np.shape(roi)) > 2:
                print('bkg_sub error: List of ROIs not surpported yet')

            if 'y' in options or 'vert' in options:
                # fitting vertical gradient
                print('Average across Y')
            else:
                # default to x axis
                #if isinstance(object, list):
                print('Average across X')

        elif type.lower() == 'surf_fit':
            # feed ROIs or data, and then interpolate surface across this before subtracting
            print('Error in bkg_sub: type surf_fit TO DO!')
            return None
        else:
            print(f'Error in bkg_sub: Unknown type: {type}')
            return None
        # always return current image
        return self.get_img()
    
    # def median_filter?

    def make_transform(self, p_px, p_t, img_size_t, img_size_px, offset = [0,0], notes = '', description='Image transform dictionary'):
        """Function to create a transform dictionary, for use with self.transform()

            p_px: array of [X,Y] pixel values on original image data. [[X1,Y1],[X2,Y2],...]
            p_t: for pixels above, corresponding array of [X,Y] values on plane being transformed, in it's coords (if real space image, mm?)
            img_size_t: [X,Y] size of plane being transformed, in it's coords (if real space image, mm?)
            img_size_px: [X,Y] new size of transformed image in pixels (can upsample)
            offset: offset of plane being transformed, in it's coords (if real space image, mm?)
            notes: For adding details about how the other variables were choosen, dates etc.
            description: Shorthand name
        """

        # resolution of new output image
        dx = img_size_t[0] / img_size_px[0] # mm / px
        dy = img_size_t[1] / img_size_px[1] # mm / px
        new_pixel_area = dx*dy

        # make new pixel coords for transformed image
        x_t = offset[0] + np.linspace(0,img_size_px[0],num=img_size_px[0]) * dx
        y_t = offset[1] + np.linspace(0,img_size_px[1],num=img_size_px[1]) * dy

        # make tranform point pixel values for transformed image (given any upsampling or offsets)
        p_pxt = (p_t - offset) / [dx, dy]

        # perform calculation for transform matrix
        H, status = cv.findHomography(p_px, p_pxt)

        # calculate pixel areas in original image (in terms of transform plane coords; for real space; mm2 per pixel)
        (orig_size_y, orig_size_x) = np.shape(self.get_img())
        retval, H_inv = cv.invert(H)
        (X,Y) = np.meshgrid(x_t,y_t)
        X_raw = cv.warpPerspective(X, H_inv, (orig_size_x, orig_size_y))
        Y_raw = cv.warpPerspective(Y, H_inv, (orig_size_x, orig_size_y))
        orig_pixel_area = np.abs(np.gradient(X_raw,axis=1)*np.gradient(Y_raw,axis=0)) # gradient is like diff, but calculates as average of differences either side
        orig_pixel_area = np.median(orig_pixel_area[np.abs(X_raw**2+Y_raw**2)>0]) # return central value of orig_pixel_area where X_raw and Y_raw > 0

        # build transform dictionary
        tform_dict = {
            'description': description,
            'H': H,
            'new_img_size': (img_size_px[0],img_size_px[1]),
            'x': x_t,
            'y': y_t,
            'orig_pixel_area': orig_pixel_area, # caluclated pixel area of plane to be transformed in calibration image (mm2 per pixel for spatial transform)
            'new_pixel_area': new_pixel_area, # area of pixel in new output imge (mm2 per pixel for spatial transform)
            'p_px': p_px,
            'p_t': p_t,
            'notes': notes,
            'newImgSize': (img_size_px[0],img_size_px[1]), # For backwards capability of old ESpec calibraions
            'x_mm': x_t, # For backwards capability of old ESpec calibraions, where transformed plane is real space (mm)
            'y_mm': y_t, # For backwards capability of old ESpec calibraions, where transformed plane is real space (mm)
            'imgArea0': orig_pixel_area, # For backwards capability of old ESpec calibraions
            'imgArea1': new_pixel_area, # For backwards capability of old ESpec calibraions
        }

        self.tform_dict = tform_dict

        return tform_dict

    def transform(self, tform_dict=None):
        """Calculate transformed image using transform dictionary and cv2 warp perspective
        """ 
        # save raw image data before transforming
        self.img_data_raw = self.get_img()

        # if not passed, use stored tform_dict, or complain
        if tform_dict is None:
            if self.tform_dict is None:
                print('Error, transform dictionary needs to be passed or loaded')
                return
        else:
            self.tform_dict = tform_dict

        # unpack the transform dictionary 
        H = self.tform_dict['H']
        raw_pixel_area_t = self.tform_dict['orig_pixel_area'] # caluclated pixel area of plane to be transformed in calibration image (mm2 per pixel for spatial transform)
        new_pixel_area = self.tform_dict['new_pixel_area'] # area of pixel in new output imge (mm2 per pixel for spatial transform)
        new_img_size = self.tform_dict['new_img_size'] 

        # scale image data by the transform plane pixel area (so it's unit areas)
        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                img_per_area = self.img_data_raw / raw_pixel_area_t
        img_per_area[raw_pixel_area_t==0] = 0
        img_per_area[np.isinf(img_per_area)] = 0
        img_per_area[np.isnan(img_per_area)] = 0

        # do warp and rescale count values for new pixel size (i.e from counts per unit area to counts per bin)
        self.set_img(cv.warpPerspective(img_per_area, H, new_img_size) * new_pixel_area)

        # Save the new X / Y scales from the dictionary to the image object
        self.x = self.tform_dict['x']
        self.y = self.tform_dict['y']

        return self.get_img(), self.x, self.y