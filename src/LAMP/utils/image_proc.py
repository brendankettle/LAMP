import numpy as np
import cv2 as cv
from matplotlib.image import imread 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from pathlib import Path
from scipy.signal import fftconvolve
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
        elif not isinstance(filepath, str):
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
        img = cv.imread(filepath, cv.IMREAD_UNCHANGED).astype(float)
        assert img is not None, f"file could not be read, check with os.path.exists(): {filepath}"
        self.set_img(img)
        return self.get_img()

    def subtract(self, img):
        self.img_data = self.img_data - img
        return self.img_data
    
    def hist(self,roi=None, bins=10,range=None, density=None, weights=None, debug=False):
        if roi:
            data = self.get_img()[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]].flatten()
        else:
            data = self.get_img().flatten()
        hist_data, bin_edges = np.histogram(data,bins=bins, range=range, density=density,weights=weights)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        if debug:
            plt.figure()
            im = plt.imshow(self.get_img(), vmin=np.percentile((self.get_img()), 1), vmax=np.percentile((self.get_img()), 99))
            cb = plt.colorbar(im)
            if roi:
                ax = plt.gca()
                rect = patches.Rectangle((roi[0][0], roi[0][1]), (roi[1][0]-roi[0][0]), (roi[1][1]-roi[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.tight_layout()
            plt.title('Image to histogram')
            plt.show(block=False)

            plt.figure()
            plt.hist(data, log=True, bins=bins, range=range, density=density, weights=weights)
            plt.xlabel('Pixel Value')
            plt.ylabel('Probability') if density else plt.ylabel('Frequency') 
            plt.show(block=False)
            plt.figure()
            plt.hist(data, log=False, bins=bins, range=range, density=density, weights=weights)
            plt.xlabel('Pixel Value')
            plt.ylabel('Probability') if density else plt.ylabel('Frequency') 
            plt.show(block=False)

        return hist_data, bin_centres

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
    
    def surface_fit(self, roi, degree=2, debug=False):
        # feed list of ROIs or data, and then interpolate surface across this before subtracting
        # PROBLEM IF THESE OVERLAP??

        img_data = self.get_img()
        ROIS = roi
        XYS = []
        ZS = []
        for ROI in ROIS:
            X = np.linspace(ROI[0][0],ROI[1][0],ROI[1][0]-ROI[0][0]) 
            Y = np.linspace(ROI[0][1],ROI[1][1],ROI[1][1]-ROI[0][1]) 
            XYS.append(np.reshape(np.meshgrid(X, Y),(2,-1)).T)
            Z = img_data[ROI[0][1]:ROI[1][1], ROI[0][0]:ROI[1][0]]
            ZS.append(np.reshape(Z,(1,-1)).T)

        PXY = np.vstack(XYS)
        ALLX = PXY[:,0]
        ALLY = PXY[:,1]
        ALLZ = np.vstack(ZS)[:,0]

        poly = PolynomialFeatures(degree)
        XY_poly = poly.fit_transform(PXY)

        # Fit the polynomial regression model
        model = LinearRegression()
        model.fit(XY_poly, ALLZ)

        # Predict Z values for the fitted surface
        #Z_pred = model.predict(XY_poly)

        # Plotting the fitted surface
        X_plot = np.linspace(0, np.shape(img_data)[1], np.shape(img_data)[1]) #np.linspace(ALLX.min(), ALLX.max(), 1000)
        Y_plot = np.linspace(0, np.shape(img_data)[0], np.shape(img_data)[0]) #np.linspace(ALLY.min(), ALLY.max(), 1000)
        X_grid, Y_grid = np.meshgrid(X_plot, Y_plot)
        XY_grid = np.vstack((X_grid.ravel(), Y_grid.ravel())).T
        surf_fit_img = model.predict(poly.transform(XY_grid)).reshape(X_grid.shape)

        #
        # TO DO: Definitely could do with more details on error of fitting. How far is fit from data?
        #
        
        if debug:
            plt.figure()
            im = plt.imshow(img_data, vmin=np.percentile(img_data, 5), vmax=np.percentile(img_data, 99))
            cb = plt.colorbar(im)
            ax = plt.gca()
            for ROI in ROIS:
                rect = patches.Rectangle((ROI[0][0], ROI[0][1]), (ROI[1][0]-ROI[0][0]), (ROI[1][1]-ROI[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show(block=False)

            # Plotting the original data points
            # This isn't great; smooth data, subtract fit, then plot 2D image of errors?
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(ALLX, ALLY, ALLZ, color='blue', label='Original Data')
            # ax.plot_surface(X_grid, Y_grid, surf_fit_img, color='red', alpha=0.5, label='Fitted Surface')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.legend()
            # plt.show(block=False)

            fig = plt.figure()
            im = plt.pcolormesh(X_plot, Y_plot, surf_fit_img, shading='auto')
            cb = plt.colorbar(im)
            #cb.set_label(self.make_units(self.img_units), rotation=270, labelpad=20)
            plt.tight_layout()
            plt.title('Calculated surface fit image')
            plt.show(block=False)
        
        return surf_fit_img, X_plot, Y_plot


    def bkg_sub(self, type, roi=None, axis=None, data=None, order=None, options=None, debug=True):
        """Could this be its own class?"""

        # switch between background type
        if type.lower() == 'img':
            # subtract an image fed into function
            self.bkg = data
            self.set_img(self.get_img() - self.bkg)

        elif type.lower() == 'flat':
            """Taking a mean count from ROI and subtracting across whole image"""
            if roi is not None:
                img_data = self.get_img()
                self.bkg =  np.mean(img_data[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]])
                self.set_img(img_data - self.bkg)
            else:
                print("bkg_sub  error: No roi provided for type=flat")

            if debug:
                plt.figure()
                im = plt.imshow(img_data, vmin=np.percentile(img_data, 5), vmax=np.percentile(img_data, 50))
                cb = plt.colorbar(im)
                #cb.set_label(self.img_units, rotation=270, labelpad=20)
                ax = plt.gca()
                rect = patches.Rectangle((roi[0][0], roi[0][1]), (roi[1][0]-roi[0][0]), (roi[1][1]-roi[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.tight_layout()
                plt.title('Uncorrected image showing ROI for flat mean')
                plt.show(block=False)
                

        elif type.lower() == 'gradient':
            # fit a polynomial to an ROI average along one axis (gradient), and extrapolate across image
            # good for constant gradients in one direction

            if roi is None:
                print("bkg_sub error: No roi provided for type=gradient")
                return

            # multiples ROIs?
            if len(np.shape(roi)) > 2:
                print('bkg_sub error: List of ROIs not surpported yet')
                return

            if order:
                polyorder = order
            else:
                polyorder = 4

            if axis == 'Y' or axis == 'y' or axis == 'vertical' or axis == 'vert':
                # fitting vertical gradient
                bkg_px = np.arange(roi[0][1],roi[1][1])
                bkg_lin = np.mean(self.get_img()[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]], 1)
                bkg_fit = np.polyfit(bkg_px, bkg_lin, polyorder)
                bkg_func = np.poly1d(bkg_fit)
                all_px = np.arange(0,np.shape(self.get_img())[0])
                all_bkg_func = bkg_func(all_px)
                if (options != 'Ext') and (polyorder > 4):
                    # don't extrapolate over high orders by default?
                    all_bkg_func[0:roi[0][1]] = bkg_func(roi[0][1])
                    all_bkg_func[roi[1][1]:] = bkg_func(roi[1][1])
                bkg_img = np.transpose(np.tile(all_bkg_func, (np.shape(self.get_img())[1],1)))
            elif axis == 'X' or axis == 'x' or axis == 'horizontal' or axis == 'hori':
                # fitting to a horizontal gradient
                bkg_px = np.arange(roi[0][0],roi[1][0])
                bkg_lin = np.mean(self.get_img()[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]], 0)
                bkg_fit = np.polyfit(bkg_px, bkg_lin, polyorder)
                bkg_func = np.poly1d(bkg_fit)
                all_px = np.arange(0,np.shape(self.get_img())[1])
                all_bkg_func = bkg_func(all_px)
                if (options != 'Ext') and (polyorder > 4):
                    # don't extrapolate over high orders by default?
                    all_bkg_func[0:roi[0][0]] = bkg_func(roi[0][0])
                    all_bkg_func[roi[1][0]:] = bkg_func(roi[1][0])
                bkg_img = np.tile(all_bkg_func, (np.shape(self.get_img())[0],1))
            else:
                print('bkg_sub error: No axis provided for type=gradient')

            #
            # TO DO: Definitely could do with more details on error of fitting. How far is fit from data?
            #
            if debug:
                plt.figure()
                im = plt.imshow(self.get_img(), vmin=np.percentile(self.get_img(), 5), vmax=np.percentile(self.get_img(), 99))
                cb = plt.colorbar(im)
                ax = plt.gca()
                rect = patches.Rectangle((roi[0][0], roi[0][1]), (roi[1][0]-roi[0][0]), (roi[1][1]-roi[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.show(block=False)
            
                plt.figure()
                plt.plot(bkg_px, bkg_lin, label='Lineout data')
                plt.plot(all_px,all_bkg_func, label='Fit')
                plt.xlabel('new pixels')
                plt.ylabel('mean counts')
                plt.tight_layout()
                plt.legend()
                plt.title('Background correction')
                plt.show(block=False)

                plt.figure()
                im = plt.imshow(bkg_img)
                cb = plt.colorbar(im)
                #cb.set_label(self.img_units, rotation=270, labelpad=20)
                plt.tight_layout()
                plt.title('Background correction')
                plt.show(block=False)

                plt.figure()
                im = plt.imshow(self.get_img()- bkg_img, vmin=np.percentile((self.get_img()-bkg_img), 10), vmax=np.percentile((self.get_img()- bkg_img), 90))
                cb = plt.colorbar(im)
                #cb.set_label(self.img_units, rotation=270, labelpad=20)
                ax = plt.gca()
                rect = patches.Rectangle((roi[0][0], roi[0][1]), (roi[1][0]-roi[0][0]), (roi[1][1]-roi[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.tight_layout()
                plt.title('Background corrected image')
                plt.show(block=False)

            self.set_img(self.get_img()- bkg_img)

        
        elif type.lower() == 'surface':

            if roi is None:
                print("bkg_sub error: No roi provided for type=surface")
            if not isinstance(roi, list):
                print('bkg_sub error: roi should be list of rois for type=surface')
                return False
        
            if order is not None:
                bkg_img, X, Y = self.surface_fit(roi, degree=order, debug=debug)
            else:
                bkg_img, X, Y = self.surface_fit(roi, debug=debug)

            img_data = self.get_img()

            if debug:
                fig = plt.figure()
                im = plt.pcolormesh(X, Y, img_data - bkg_img, vmin=np.percentile(img_data - bkg_img, 10), vmax=np.percentile(img_data - bkg_img, 90), shading='auto')
                cb = plt.colorbar(im)
                ax = plt.gca()
                #cb.set_label(self.make_units(self.img_units), rotation=270, labelpad=20)
                for ROI in roi:
                    rect = patches.Rectangle((ROI[0][0], ROI[0][1]), (ROI[1][0]-ROI[0][0]), (ROI[1][1]-ROI[0][1]), linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                plt.title('Background corrected image')
                plt.tight_layout()
                plt.show(block=False)

            self.set_img(img_data - bkg_img)

        else:
            print(f'Error in bkg_sub: Unknown type: {type}')
            return None
        # always return current image
        return self.get_img()
    
    # def median_filter?
    def median_filter(self, size):
        self.img_data = ndimage.median_filter(self.img_data, size=size)
        return self.img_data

    def savgol_filter(self, window_size, order, derivative=None):
        """2D version of Savitzky Golay Filtering
        https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay"""

        z = self.get_img()

        # number of terms in the polynomial expression
        n_terms = ( order + 1 ) * ( order + 2)  / 2.0
        
        if  window_size % 2 == 0:
            raise ValueError('window_size must be odd')
        
        if window_size**2 < n_terms:
            raise ValueError('order is too high for the window size')

        half_size = window_size // 2
        
        # exponents of the polynomial. 
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
        # this line gives a list of two item tuple. Each tuple contains 
        # the exponents of the k-th term. First element of tuple is for x
        # second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
        
        # coordinates of points
        ind = np.arange(-half_size, half_size+1, dtype=np.float64)
        dx = np.repeat( ind, window_size )
        dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

        # build matrix of system of equation
        A = np.empty( (window_size**2, len(exps)) )
        for i, exp in enumerate( exps ):
            A[:,i] = (dx**exp[0]) * (dy**exp[1])
            
        # pad input array with appropriate values at the four borders
        new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
        Z = np.zeros( (new_shape) )
        # top band
        band = z[0, :]
        Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
        # bottom band
        band = z[-1, :]
        Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band ) 
        # left band
        band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
        Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
        # right band
        band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
        Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
        # central band
        Z[half_size:-half_size, half_size:-half_size] = z
        
        # top left corner
        band = z[0,0]
        Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
        # bottom right corner
        band = z[-1,-1]
        Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band ) 
        
        # top right corner
        band = Z[half_size,-half_size:]
        Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band ) 
        # bottom left corner
        band = Z[-half_size:,half_size].reshape(-1,1)
        Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band ) 

        # solve system and convolve
        if derivative == None:
            m = np.linalg.pinv(A)[0].reshape((window_size, -1))
            return fftconvolve(Z, m, mode='valid')
        elif derivative == 'col':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            return fftconvolve(Z, -c, mode='valid')        
        elif derivative == 'row':
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return fftconvolve(Z, -r, mode='valid')        
        elif derivative == 'both':
            c = np.linalg.pinv(A)[1].reshape((window_size, -1))
            r = np.linalg.pinv(A)[2].reshape((window_size, -1))
            return fftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid') 

    def make_transform(self, p_px, p_t, img_size_t, img_size_px, offset = [0,0], notes = '', description='Image transform dictionary'):
        """Function to create a transform dictionary, for use with self.transform()

            p_px: array of [X,Y] pixel values on original image data. [[X1,Y1],[X2,Y2],...]
            p_t: for pixels above, corresponding array of [X,Y] values on plane being transformed, in it's coords (if real space image, mm?)
            img_size_t: [X,Y] size of plane being transformed (and new transformed image), in it's coords (if real space image, mm?)
            img_size_px: [X,Y] size of new transformed image in pixels (can down/up sample)
            offset: [X,Y] offset of plane being transformed, in it's coords (if real space image, mm?)
            notes: For adding details about how the other variables were choosen, dates etc.
            description: Shorthand name
        """

        # uses new image size and new number of pixels, to compute resolution of new output image
        dx = img_size_t[0] / img_size_px[0] # mm / px
        dy = img_size_t[1] / img_size_px[1] # mm / px
        new_pixel_area = dx*dy

        # make new axis for transformed image, in its coords
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
            'notes': notes,
            'H': H,
            'new_img_size': (img_size_px[0],img_size_px[1]), # in pixels, for making image
            'x': x_t,
            'y': y_t,
            'orig_pixel_area': orig_pixel_area, # caluclated pixel area of plane to be transformed in calibration image (mm2 per pixel for spatial transform)
            'new_pixel_area': new_pixel_area, # area of pixel in new output imge (mm2 per pixel for spatial transform)
            'p_px': p_px,
            'p_t': p_t,
            # 'newImgSize': (img_size_px[0],img_size_px[1]), # For backwards capability of old ESpec calibraions
            # 'x_mm': x_t, # For backwards capability of old ESpec calibraions, where transformed plane is real space (mm)
            # 'y_mm': y_t, # For backwards capability of old ESpec calibraions, where transformed plane is real space (mm)
            # 'imgArea0': orig_pixel_area, # For backwards capability of old ESpec calibraions
            # 'imgArea1': new_pixel_area, # For backwards capability of old ESpec calibraions
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
    
    def blob_filter(self, threshold = 50, size=1, img_max=pow(2,16), debug=False):
        """Wrapper for OpenCV functions
        TODO: Add the option to pass the other arguments; interia, convexity etc."""

        # need to first conver to 8 bit
        blob_img = self.get_img().copy()
        blob_img[blob_img<0] = 0 # assume no negatives (background corrected)
        blob_img = (blob_img / img_max) * 255 # can't normalise to max, should be normalise to maximum POSSIBLE counts
        blob_img = blob_img.astype(np.uint8) 

        # Setup SimpleBlobDetector parameters
        params = cv.SimpleBlobDetector_Params()
        # Thresholds for binarization
        params.minThreshold = threshold 
        params.maxThreshold = 255
        params.thresholdStep = 1
        # Filter by colour ??
        params.filterByColor = False
        #params.blobColor = 255 # this might just be zero or 255, whether you are looking for dark or bright?
        # Filter by Area
        params.filterByArea = True
        params.minArea = size
        # Filter by Circularity
        params.filterByCircularity = False
        #params.minCircularity = 0.1
        # Filter by Convexity
        params.filterByConvexity = False
        #params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = False
        #params.minInertiaRatio = 0.01
        # Create a detector with the parameters
        detector = cv.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(blob_img)

        if len(keypoints) > 0:
            img = self.get_img()
            # https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
            sizes = []
            blobs = []
            responses = []
            #plt.figure()
            ymax = np.shape(img)[0]
            xmax = np.shape(img)[1]
            for keypoint in keypoints:
                x = keypoint.pt[0]
                y = keypoint.pt[1]
                size = keypoint.size
                #plt.scatter(x,y,size)
                sizes.append(size)
                responses.append(keypoint.response) #??
                y1 = min(ymax,max(0,int(np.floor(y-size)-1)))
                y2 = min(ymax,max(0,int(np.ceil(y+size)+1)))
                x1 = min(xmax,max(0,int(np.floor(x-size)-1)))
                x2 = min(xmax,max(0,int(np.ceil(x+size)+1)))
                blobs.append(img[y1:y2,x1:x2].copy())
                median = int(np.median(img[y1:y2,x1:x2]))
                img[y1:y2,x1:x2] = np.ones(np.shape(img[y1:y2,x1:x2])) * median
                if debug:
                    print(f'Blob replaced: x={x}, y={y}, size={size}, max={np.max(blobs[-1])}. Median: {median}')

            if debug:
                # print(f'{len(keypoints)} blobs removed.')
                # plt.figure()
                # im = plt.imshow(blob_img, vmin=0, vmax=255)
                # plt.colorbar(im)
                # plt.show(block=False)
                for blob in blobs:
                    plt.figure()
                    im = plt.imshow(blob)
                    plt.colorbar(im)
                    plt.show(block=False)

            self.set_img(img)

        return self.get_img(), len(keypoints)