import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os
from scipy.linalg import lstsq
from .diagnostic import Diagnostic
from ..utils.image_proc import ImageProc
from ..utils.filter_transmission import filter_transmission

# TODO: Some of this was bodged quickly, need to make sure it's fully compatiable/flexible
# TODO: Bayesian fitting (see Evas previous code)
# TODO: Masking etc. should be in ImageProc?
# TODO: And background correction!

class XrayFilterArray(Diagnostic):
    """X-ray filter array analysis. I.e. fitting an incident spectrum for E.g. betatron diagnostic
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''

    filter_specs = None
    filter_positions = None
    filter_mask = None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return

    def get_shot_img(self, shot_dict, subtract_dark=True, calib_id=None):
        """Return a shot image, by default with darks removed. Background?"""
        raw_img = self.get_shot_data(shot_dict)
        img = ImageProc(raw_img)
        if subtract_dark:
            if hasattr(self, 'dark_img'):
                img.subtract(self.dark_img)
            elif 'dark_img' in self.calib_dict: 
                img.subtract(self.calib_dict['dark_img'])
            else:
                calib_input = self.get_calib_input(calib_id, shot_dict=shot_dict)
                if 'darks' in calib_input:
                    img.subtract(self.make_dark(calib_input['darks']))
        return img.get_img()
    
    def get_shot_counts(self, shot_dict, calib_id=None):
        """Return sum counts of a a shot
        """
        return np.sum(self.get_shot_img(shot_dict))
    
    def get_img_size(self):
        # TODO: This could be in base class?
        # TODO: Could use calibration file or load test image or dark or something?
        return [int(self.config['camera']['img_width']),int(self.config['camera']['img_height'])]

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

        # now we have the shot dictionary, check if it's the same as previously loaded, and if so, return saved dark
        if hasattr(self, 'dark_shot_dicts') and hasattr(self, 'dark_img'):
            if shot_dicts == self.dark_shot_dicts:
                return self.dark_img

        # loop through all shots, and build average dark
        num_shots = 0
        for shot_dict in shot_dicts:
            if 'img' in locals():
                img += self.DAQ.get_shot_data(self.diag_name, shot_dict)
            else:
                img = self.DAQ.get_shot_data(self.diag_name, shot_dict)
            num_shots += 1
        avg_img = img / num_shots
        self.dark_img = avg_img
        self.dark_shot_dicts = shot_dicts

        return avg_img
    

    def make_calib(self, calib_input, calib_filename=None, view=True):
        """Pass a calibration input and generate a full calibration file
        """

        # Get and set calibration input 
        calib_input = self.set_calib_input(calib_input)

        # start with transferring input data to new calibration dictionary
        for dict_key in calib_input:
            self.calib_dict[dict_key] = calib_input[dict_key]

        # Make dark
        if 'darks' in self.calib_input:
            dark_img = self.make_dark(calib_input['darks'])
            self.calib_dict['dark_img'] = dark_img

        # Make filter mask
        filter_mask = self.make_filter_mask(calib_input)
        self.calib_dict['filter_mask'] = filter_mask

        # Make direct signal mask
        direct_mask = self.make_direct_mask(calib_input)
        self.calib_dict['direct_mask'] = direct_mask

        # TODO: CCD QE?

        # TODO: Combined throughput for filtering and QE etc.?

        # plotting?
        if view:
            plt.figure()
            plt.imshow(filter_mask, vmin=np.min(filter_mask), vmax=np.max(filter_mask))
            plt.colorbar()
            plt.show(block=False)

            plt.figure()
            plt.imshow(direct_mask, vmin=np.min(direct_mask), vmax=np.max(direct_mask))
            plt.colorbar()
            plt.show(block=False)

        #print(self.calib_dict)

        # save the full calibration?
        if calib_filename:
            self.save_calib(calib_filename)

        return self.get_calib()

    def calc_direct_fit(self, shot_dict, calib_id=None, view=False):

        # get calibration dictionary
        calib_dict = self.get_calib(calib_id, shot_dict=shot_dict)

        img = self.get_shot_img(shot_dict)

        if 'direct_mask' in calib_dict:
            direct_mask = calib_dict['direct_mask']
        else:
            direct_mask = self.make_direct_mask(self.get_calib_input(calib_id, shot_dict=shot_dict))

        direct_signal = img * direct_mask
        direct_signal[direct_signal==0] = np.nan

        x = np.arange(np.shape(img)[0])
        y = np.arange(np.shape(img)[1])

        direct_fit = self.polyfit2D(x, y, direct_signal, x, y)

        if view:
            plt.figure()
            plt.imshow(img * direct_mask, vmin=np.min(img * direct_mask), vmax=np.percentile(img * direct_mask, 95))
            plt.title('Direct Signal Mask')
            plt.colorbar()
            plt.show(block=False)

            plt.figure()
            plt.imshow(direct_fit, vmin=np.min(direct_fit), vmax=np.max(direct_fit))
            plt.colorbar()
            plt.title('Direct Fit')
            plt.show(block=False)

        return direct_fit

    def make_filter_mask(self, calib_input=None, method='polygon', fig_ax=None):

        calib_input = self.get_calib_input(calib_input)

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs(calib_input)

        # TODO: check calibration input for method
        if method.lower() == 'polygon':
            if 'filter_positions' not in self.calib_dict:
                self.calib_dict['filter_positions'] = self.load_filter_positions(calib_id=calib_input)

            if 'filter_margin_inner' in calib_input:
                filter_radius = calib_input['filter_margin_inner']
            else:
                filter_radius = 0

            for filter in self.calib_dict['filter_positions']:
                filter_id = filter['spec_id']
                xv, yv, mask = self.create_masked_polygon(filter['points'], filter_radius=filter_radius)
                if filter_id in self.calib_dict['filter_specs']:
                    filter_spec = self.calib_dict['filter_specs'][int(filter_id)]
                    filter_name = filter_spec['name']
                    if fig_ax:
                        fig_ax.add_patch(patches.Polygon(filter['points'], fill=None, edgecolor='r'))
                        fig_ax.text(filter['points'][0][0], filter['points'][0][1], filter_name, color='white')
                    mask[mask==1] = int(filter_id)
                else:
                    print(f"Error, missing filter with id: {filter_id}, in filter specs")

                if 'full_mask' in locals():
                    full_mask += mask
                else:
                    full_mask = mask

        elif method.lower() == 'auto':
            print("TODO: No auto filter mask method yet...")

        self.filter_mask = full_mask

        return full_mask


    def make_direct_mask(self, calib_input, method='inv_filters'):

        calib_input = self.get_calib_input(calib_input)

        # use filter positions... but expand filters outside. 
        if method.lower() == 'inv_filters':
            if 'filter_positions' not in self.calib_dict:
                self.calib_dict['filter_positions'] = self.load_filter_positions(calib_id=calib_input)

            if 'filter_margin_outer' in calib_input:
                filter_radius = -calib_input['filter_margin_outer']
            else:
                filter_radius = 0

            for filter in self.calib_dict['filter_positions']:
                xv, yv, mask = self.create_masked_polygon(filter['points'], filter_radius=filter_radius)
                if 'full_mask' in locals():
                    full_mask += mask
                else:
                    full_mask = mask

            full_mask = 1 - full_mask
            full_mask[full_mask<0] = 0
            
            # roi?
            if 'roi' in calib_input:
                roi = calib_input['roi']
                full_mask[:,:roi[0]] = 0
                full_mask[:roi[1],:] = 0
                full_mask[:,roi[2]:] = 0
                full_mask[roi[3]:,:] = 0

        elif method.lower() == 'auto':
            print("TODO: No auto direct mask method yet... Threshold value of some sort?")

        return full_mask

    def create_masked_polygon(self, coord_array, filter_radius=0):
        """Pass a list of points, and generate a masked element where points inside path are 1, outside are 0
            coord_array should be (N,2) shaped, i.e. [[x0, y0], [x1, y1], ... [xn, yn]]
        """
        # TODO: This could be in image proc class? (pass image size as argument)
        x = np.arange(0,self.get_img_size()[0])
        y = np.arange(0,self.get_img_size()[1])
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
        filter_path = matplotlib.path.Path(coord_array)
        mask = filter_path.contains_points(points, radius=filter_radius).astype(float)
        mask.shape = xv.shape

        return xv, yv, mask

    def load_filter_specs(self, calib_id=None):
        """Load the material/thickness/etc details for each element in filter array
        """
        calib_input = self.get_calib_input(calib_id)
        filter_specs_file = calib_input['filter_specs_file']
        filter_specs = self.load_calib_file(filter_specs_file)
        # old csv format? ... Reformat
        filepath_no_ext, file_ext = os.path.splitext(filter_specs_file)
        if file_ext.lower() == '.csv':
            #  filter_label, filter_keys, filter_names, filter_widths, mass_density, filter_k_edges, uncertainty_in_filter_width, background_sub_filter, filter_backing_name, filter_backing_widths, filter_backing_mass_density
            filters = {}
            for filter in filter_specs:
                id = int(filter[0]) # force an int id (legacy was float?)
                filters[id] = {
                    "name": filter[1].strip(),
                    "material": filter[2].strip(),
                    "thickness": filter[3]*1e4, # this is in cm! convert to um
                    "mass_density": filter[4],
                    "k_edge": filter[5],
                    "thickness_err": filter[6],
                    "bkg_sub": filter[7],
                    "backing_material": filter[8].strip(),
                    "backing_thickness": filter[9],
                    "backing_mass_density": filter[10]
                }
        else:
            # assume we've laoded a JSON file with a dictionary in the correct format...
            filters = filter_specs

        self.calib_dict['filter_specs'] = filters

        return filters

    def load_filter_positions(self, calib_id=None):
        """Load the polygon coordinates of each filter element
        """
        calib_input = self.get_calib_input(calib_id)
        filter_positions_file = calib_input['filter_positions_file']
        filter_positions = self.load_calib_file(filter_positions_file)

        # any shifts? to save rewriting all locations...
        if 'filter_xshift' in calib_input:
            xshift = calib_input['filter_xshift']
        else:
            xshift = 0
        if 'filter_yshift' in calib_input:
            yshift = calib_input['filter_yshift']
        else:
            yshift = 0
        # any flips? filter pack could be in back to front or upside down.
        if 'filter_xflip' in calib_input and calib_input['filter_xflip']:
            if 'camera' in self.config and 'img_width' in self.config['camera']:
                xflip = int(self.config['camera']['img_width'])
            else:
                print(f'Error; Could not flip filter positions without camera.img_width being set in {self.diag_name} config.')
        else:
            xflip = 0
        if 'filter_yflip' in calib_input and calib_input['filter_yflip']:
            if 'camera' in self.config and 'img_height' in self.config['camera']:
                yflip = int(self.config['camera']['img_height'])
            else:
                print(f'Error; Could not flip filter positions without camera.img_height being set in {self.diag_name} config.')
        else:
            yflip = 0
        # any scaling? moving filter pack forward/back will change shadow positions
        if 'filter_xscale' in calib_input and calib_input['filter_xscale']:
            xscale = float(calib_input['filter_xscale'])
        else:
            xscale = 1
        if 'filter_yscale' in calib_input and calib_input['filter_yscale']:
            yscale = float(calib_input['filter_yscale'])
        else:
            yscale = 1

        filters = []
        # old csv format? ... Reformat
        filepath_no_ext, file_ext = os.path.splitext(filter_positions_file)
        if file_ext.lower() == '.csv':
            #  filter number, x0, y0, x1, y1, x2, y2, x3, y3
            for filter in filter_positions:
                filter = np.array(list(filter))
                filter[1::2] = (abs(filter[1::2] - xflip) * xscale) + xshift # x's. Flip first, then scale, then shift
                filter[2::2] = (abs(filter[2::2] - yflip) * yscale) + yshift # y's. Flip first, then scale, then shift
                filters.append({"spec_id": int(filter[0]), "points": filter[1:].reshape((4,2))})
        else:
            # assume we've laoded a JSON file with a dictionary in the correct format...
            # But handle x/y shifts
            print('UNTESTED ADN NEED TO INCORPORATE FLIPS / SCALING')
            for filter in filter_positions:
                filter_pos = filter_positions[filter]
                filter_pos[:,0] = filter_pos[:,0] + xshift
                filter_pos[:,1] = filter_pos[:,1] + yshift
                filters.append({"spec_id": int(filter[0]), "points": filter_pos})

        self.calib_dict['filter_positions'] = filters

        return filters

    def plot_filter_positions(self, shot_dict, calib_id=None):

        calib_input = self.get_calib_input(calib_id, shot_dict=shot_dict)
        
        shot_img = self.get_shot_img(shot_dict)
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(shot_img, vmin=np.min(shot_img), vmax=np.mean(shot_img)*2)
        plt.colorbar(im, ax=ax)
        full_mask = self.make_filter_mask(calib_input, fig_ax=ax)
        plt.show(block=False)

        plt.figure()
        plt.imshow(full_mask, vmin=np.min(full_mask), vmax=np.max(full_mask))
        plt.colorbar()
        plt.show(block=False)

        return

    def plot_filter_transmissions(self, calib_id = None, eV = range(100,int(1e5))):

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs(calib_id)

        calib_dict = self.get_calib(calib_id)

        plt.figure()

        for fid in self.calib_dict['filter_specs']:
            filter = self.calib_dict['filter_specs'][fid]
            trans = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])
            plt.plot(eV, trans, label=filter['name'])
            
        if calib_dict and 'base_filtering' in calib_dict:
            for fname in calib_dict['base_filtering']:
                filter = calib_dict['base_filtering'][fname]
                trans = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])
                plt.plot(eV, trans, label=fname)

        plt.legend()
        plt.xlabel('Photon Energy [eV]')
        plt.ylabel('Transmission')
        plt.show(block=False)

        return
    
    def get_element_transmissions(self, shot_dict, view=False):
        """Return the """

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs()

        if 'filter_mask' not in self.calib_dict:
            self.calib_dict['filter_mask'] = self.make_filter_mask()

        direct_fit = self.calc_direct_fit(shot_dict, view=view)
        shot_img = self.get_shot_img(shot_dict)

        element_transmissions = {}
        for fid in self.calib_dict['filter_specs']:
            element_direct_counts = np.sum(direct_fit[self.calib_dict['filter_mask']==fid])
            element_filtered_counts = np.sum(shot_img[self.calib_dict['filter_mask']==fid])
            element_transmissions[fid] = element_filtered_counts / element_direct_counts

        return element_transmissions

    # TODO: Outside of this class? ImageProc?
    def polyfit2D(self, new_x, new_y, img, x, y, norder=4):
        """Return a fit to the data in img. NaNs are ignored.
        """
        X, Y = np.meshgrid(x, y)
        new_X, new_Y = np.meshgrid(new_x, new_y)
        # delete nans
        img, X, Y = img[~np.isnan(img)], X[~np.isnan(img)], Y[~np.isnan(img)]

        if norder == 4: 
            # best-fit quartic curve
            A = np.c_[np.ones(X.shape[0]), X, Y, X**2, X*Y, Y**2, X**3, X**2*Y, X*Y**2, Y**3, X**4, X**3*Y, X**2*Y**2, X*Y**3, Y**4]
            C,_,_,_ = lstsq(A, img)
            # evaluate it on a grid
            Z = C[0]+C[1]*new_X+C[2]*new_Y+C[3]*new_X**2+C[4]*new_X*new_Y+C[5]*new_Y**2+C[6]*new_X**3+C[7]*new_X**2*new_Y+C[8]*new_X*new_Y**2+C[9]*new_Y**3+C[10]*new_X**4+C[11]*new_X**3*new_Y+C[12]*new_X**2*new_Y**2+C[13]*new_X*new_Y**3+C[14]*new_Y**4
            
        elif norder == 1:
            # best-fit quartic curve
            A = np.c_[np.ones(X.shape[0]), X, Y]
            C,_,_,_ = lstsq(A, img)
            # evaluate it on a grid
            Z = C[0]+C[1]*new_X+C[2]*new_Y
        else:
            print(f"Unknown poltfit_2D() order: {norder}")

        return Z

    # # TODO: Outside of this class? ImageProc?
    # def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    #     '''
    #     Two dimensional polynomial fitting by least squares.
    #     Fits the functional form f(x,y) = z.

    #     Notes
    #     -----
    #     Resultant fit can be plotted with:
    #     np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    #     Parameters
    #     ----------
    #     x, y: array-like, 1d
    #         x and y coordinates.
    #     z: np.ndarray, 2d
    #         Surface to fit.
    #     kx, ky: int, default is 3
    #         Polynomial order in x and y, respectively.
    #     order: int or None, default is None
    #         If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
    #         If int, coefficients up to a maximum of kx+ky <= order are considered.

    #     Returns
    #     -------
    #     Return paramters from np.linalg.lstsq.

    #     soln: np.ndarray
    #         Array of polynomial coefficients.
    #     residuals: np.ndarray
    #     rank: int
    #     s: np.ndarray

    #     '''

    #     # grid coords
    #     x, y = np.meshgrid(x, y)
    #     # coefficient array, up to x^kx, y^ky
    #     coeffs = np.ones((kx+1, ky+1))

    #     # solve array
    #     a = np.zeros((coeffs.size, x.size))

    #     # for each coefficient produce array x^i, y^j
    #     #for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
    #     for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
    #         # do not include powers greater than order
    #         if order is not None and i + j > order:
    #             arr = np.zeros_like(x)
    #         else:
    #             arr = coeffs[i, j] * x**i * y**j
    #         a[index] = arr.ravel()

    #     # do leastsq fitting and return leastsq result
    #     soln, residuals, rank, s = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

    #     # make fitted surface and return
    #     fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1,ky+1)))

    #     return fitted_surf