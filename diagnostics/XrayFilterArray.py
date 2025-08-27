import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import os
from scipy.special import kv
from scipy.linalg import lstsq
from scipy.optimize import least_squares, leastsq
from scipy.interpolate import interp1d
from LAMP.diagnostic import Diagnostic
from LAMP.utils.image_proc import ImageProc
from LAMP.utils.xrays.filter_transmission import filter_transmission
from LAMP.utils.io import load_csv

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
    data_type = 'image'

    filter_specs = None
    filter_positions = None
    filter_mask = None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        return
    
    # not sure this function is needed....
    def get_shot_img(self, shot_dict, debug=False): 
        """Return a shot image, by default with darks removed. Background?"""
        img, x, y = self.get_proc_shot(shot_dict, debug=debug)
        return img
    
    def get_shot_counts(self, shot_dict):
        """Return sum counts of a a shot
        """
        return np.sum(self.get_shot_img(shot_dict))
    
    def get_img_size(self):
        # TODO: This could be in base class?
        # TODO: Could use calibration file or load test image or dark or something?
        return [int(self.calib_dict['camera']['img_width']),int(self.calib_dict['camera']['img_height'])]
    
    def make_calib(self, calib_id, save=False, debug=True):
        """Pass a calibration input and generate a full calibration file
        """
        # Get calibration input
        self.calib_dict = self.get_calib(calib_id, no_proc=True)

        # start with transferring input data to new calibration dictionary
        # for dict_key in calib_input:
        #     self.calib_dict[dict_key] = calib_input[dict_key]

        # Make dark
        # if 'darks' in self.calib_input:
        #     dark_img = self.make_dark(calib_input['darks'])
        #     self.calib_dict['dark_img'] = dark_img

        # Make filter mask
        filter_mask = self.make_filter_mask()
        self.calib_dict['filter_mask'] = filter_mask

        # Make direct signal mask
        direct_mask = self.make_direct_mask()
        self.calib_dict['direct_mask'] = direct_mask

        # plotting?
        if debug:
            plt.figure()
            plt.imshow(filter_mask, vmin=np.min(filter_mask), vmax=np.max(filter_mask))
            plt.colorbar()
            plt.show(block=False)

            plt.figure()
            plt.imshow(direct_mask, vmin=np.min(direct_mask), vmax=np.max(direct_mask))
            plt.colorbar()
            plt.show(block=False)

        # save the full calibration?
        if save:
            if 'proc_file' not in self.calib_dict:
                print('Error, proc_file variable required in calibration if saving.')
            self.save_calib_file(self.calib_dict['proc_file'], self.calib_dict)

        return self.get_calib()

    def normalised_synchrotron_spectrum(self, E_crit, eV):
        """In Energy, for in terms of photon number, divide by eV"""
        var = eV / (2*E_crit)
        K = kv(2/3, var) # modified Bessel function of the second kind
        shape = (var**2) * (K**2) # * constants * gamma ^2
        norm_shape = shape / max(shape)
        return norm_shape

    def load_QE(self, eV):
        """Wrapper function to load camera QE"""
        QE_data = self.load_calib_file(self.config['camera']['QE_file'])
        data_eV = QE_data[:,0]
        data_QE = QE_data[:,1]
        interp_func = interp1d(data_eV, data_QE, fill_value='extrapolate')
        QE = interp_func(eV)
        return QE

    def diff_measured_synchrotron(self, params, eV, shared_throughput, filter_transmissions, element_signals):
        """filter_tranmissions and element_signals are dictionaries with keys of each filter id"""
        E_crit, height = params
        sync_spec = self.normalised_synchrotron_spectrum(E_crit, eV)
        err = []
        for fid in filter_transmissions:
            element_calc = np.trapz(sync_spec * shared_throughput * filter_transmissions[fid], eV) * height
            err.append(element_calc - element_signals[fid])
        return err

    def get_Ecrits(self, timeframe, exceptions=None):

        shot_dicts = self.DAQ.get_shot_dicts(self.diag_name,timeframe,exceptions=exceptions)

        E_crits = []
        for shot_dict in shot_dicts:
            print(shot_dict)
            E_crit, height = self.fit_synchrotron_spectrum(shot_dict)
            E_crits.append(E_crit)
                                                  
        return E_crits

    def fit_synchrotron_spectrum(self, shot_dict, E_crit_guess = 10e3, height_guess = 1, eV = range(100,int(1e5),int(1e2)), debug=False):
        """"""
        # load system throughput (QE, shared filtering)
        QE = self.load_QE(eV)
        # if view:
        #     plt.figure()
        #     plt.plot(eV, QE)
        #     plt.title('QE')
        #     plt.show(block=False)
        shared_throughput = QE.copy()
        filter_transmissions, base_filter_transmissions = self.get_filter_transmissions(eV)
        for filter_name in base_filter_transmissions:
            # if view:
            #     plt.figure()
            #     plt.plot(eV, base_filter_transmissions[filter_name])
            #     plt.title(f"{self.calib_dict['base_filtering'][filter_name]['material']}, {self.calib_dict['base_filtering'][filter_name]['thickness']} um,{self.calib_dict['base_filtering'][filter_name]['mass_density']} g/cc")
            #     plt.show(block=False)
            shared_throughput = shared_throughput * base_filter_transmissions[filter_name]

        # get measured signals through each filter element
        element_signals = self.get_element_signals(shot_dict)


        # IS LEASTSQ CORRECT HERE?? scipys minimize function??
        params, pcov, infodict, errmsg, success = leastsq(self.diff_measured_synchrotron, x0=[E_crit_guess,height_guess], args=(eV, shared_throughput, filter_transmissions, element_signals), full_output=True)
        
        # To obtain the covariance matrix of the parameters x, cov_x must be multiplied by the variance of the residuals – see curve_fit.
        s_sq = (self.diff_measured_synchrotron(params, eV, shared_throughput, filter_transmissions, element_signals)[0]**2).sum()/(len(element_signals)-len(params))
        pcov = np.diag(pcov * s_sq)**0.5
        # NOT SURE ABOUT THE ABOVE

        if success < 1 or success > 4:
            print('Error fitting Synchrotron Spectrum. Error message follows: ')
            print(errmsg)
            print(infodict)

        E_crit_best = params[0]
        height_best = params[1]

        if debug:
            sync_spec_best = self.normalised_synchrotron_spectrum(E_crit_best, eV)
            plt.figure()
            best_fits = {}
            for fid in filter_transmissions:
                plt.plot(eV, sync_spec_best * shared_throughput * filter_transmissions[fid], label=fid)
                best_fits[fid] = np.trapz(sync_spec_best * shared_throughput * filter_transmissions[fid], eV) * height_best
                #print(f'{best_fits[fid]}, {element_signals[fid]}')
            plt.legend()
            plt.show(block=False)

            plt.figure()
            plt.plot(element_signals.keys(), element_signals.values(), label='Measured')
            plt.plot(best_fits.keys(), best_fits.values(), label='Fit')
            plt.xlabel('Filter ID')
            plt.legend()
            plt.show(block=False)

        return E_crit_best, height_best

    def calc_direct_fit(self, shot_dict, debug=False):
        """"""
        #img,x,y = self.get_shot_img(shot_dict)
        img,x,y = self.get_proc_shot(shot_dict)

        if 'direct_mask' in self.calib_dict:
            direct_mask = self.calib_dict['direct_mask']
        else:
            direct_mask = self.make_direct_mask()

        direct_signal = img * direct_mask
        direct_signal[direct_signal==0] = np.nan

        x = np.arange(np.shape(img)[0])
        y = np.arange(np.shape(img)[1])

        direct_fit = self.polyfit2D(x, y, direct_signal, x, y)

        if debug:
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

    def make_filter_mask(self, method='polygon', fig_ax=None):
        """"""

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs()

        # TODO: check calibration input for method
        if method.lower() == 'polygon':
            if 'filter_positions' not in self.calib_dict:
                self.calib_dict['filter_positions'] = self.load_filter_positions()

            if 'filter_margin_inner' in self.calib_dict:
                filter_radius = self.calib_dict['filter_margin_inner']
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


    def make_direct_mask(self, method='inv_filters'):
        """"""

        # use filter positions... but expand filters outside. 
        if method.lower() == 'inv_filters':
            if 'filter_positions' not in self.calib_dict:
                self.calib_dict['filter_positions'] = self.load_filter_positions()

            if 'filter_margin_outer' in self.calib_dict:
                filter_radius = -self.calib_dict['filter_margin_outer']
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
            if 'roi' in self.calib_dict:
                roi = self.calib_dict['roi']
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

    def load_filter_specs(self):
        """Load the material/thickness/etc details for each element in filter array
        """

        filter_specs_file = self.calib_dict['filter_specs_file']
        # old csv format? ... Reformat
        filepath_no_ext, file_ext = os.path.splitext(filter_specs_file)
        if file_ext.lower() == '.csv':
            # below could be better?
            filter_specs = self.load_calib_file(filter_specs_file)
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
            # this is old LAMP.... needs updated
            filters = filter_specs

        self.calib_dict['filter_specs'] = filters

        return filters

    def load_filter_positions(self):
        """Load the polygon coordinates of each filter element
        """

        filter_positions_file = self.calib_dict['filter_positions_file']
        filter_positions = self.load_calib_file(filter_positions_file)

        # any shifts? to save rewriting all locations...
        if 'filter_xshift' in self.calib_dict:
            xshift = self.calib_dict['filter_xshift']
        else:
            xshift = 0
        if 'filter_yshift' in self.calib_dict:
            yshift = self.calib_dict['filter_yshift']
        else:
            yshift = 0
        # any flips? filter pack could be in back to front or upside down.
        if 'filter_xflip' in self.calib_dict and self.calib_dict['filter_xflip']:
            if 'camera' in self.config and 'img_width' in self.config['camera']:
                xflip = int(self.config['camera']['img_width'])
            else:
                print(f"Error; Could not flip filter positions without camera.img_width being set in {self.config['name']} config.")
        else:
            xflip = 0
        if 'filter_yflip' in self.calib_dict and self.calib_dict['filter_yflip']:
            if 'camera' in self.config and 'img_height' in self.config['camera']:
                yflip = int(self.config['camera']['img_height'])
            else:
                print(f"Error; Could not flip filter positions without camera.img_height being set in {self.config['name']} config.")
        else:
            yflip = 0
        # any scaling? moving filter pack forward/back will change shadow positions
        if 'filter_xscale' in self.calib_dict and self.calib_dict['filter_xscale']:
            xscale = float(self.calib_dict['filter_xscale'])
        else:
            xscale = 1
        if 'filter_yscale' in self.calib_dict and self.calib_dict['filter_yscale']:
            yscale = float(self.calib_dict['filter_yscale'])
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

    def plot_filter_positions(self, shot_dict):
        """"""
        # shot_img,x,y = self.get_shot_img(shot_dict)
        shot_img,x,y = self.get_proc_shot(shot_dict)

        plt.figure()
        ax = plt.gca()
        im = ax.imshow(shot_img, vmin=np.min(shot_img), vmax=np.mean(shot_img)*2)
        plt.colorbar(im, ax=ax)
        full_mask = self.make_filter_mask(fig_ax=ax)
        plt.show(block=False)

        plt.figure()
        plt.imshow(full_mask, vmin=np.min(full_mask), vmax=np.max(full_mask))
        plt.colorbar()
        plt.show(block=False)

        return

    def get_filter_transmissions(self, eV, calib_id=None):
        """"""
        if calib_id: # be careful here of overwriting
           self.calib_dict = self.get_calib(calib_id)
        
        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs(calib_id=calib_id)

        filter_transmissions = {}
        for fid in self.calib_dict['filter_specs']:
            filter = self.calib_dict['filter_specs'][fid]
            # filter['name']
            filter_transmissions[fid] = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])
        base_filter_transmissions = {}
        if self.calib_dict and 'base_filtering' in self.calib_dict:
            for fname in self.calib_dict['base_filtering']:
                filter = self.calib_dict['base_filtering'][fname]
                base_filter_transmissions[fname] = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])

        return filter_transmissions, base_filter_transmissions

    def plot_filter_transmissions(self, calib_id = None, eV = range(100,int(1e5))):
        """"""
        if calib_id: # be caareful here of overwriting
           self.calib_dict = self.get_calib(calib_id)

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs()

        plt.figure()

        for fid in self.calib_dict['filter_specs']:
            filter = self.calib_dict['filter_specs'][fid]
            trans = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])
            plt.plot(eV, trans, label=filter['name'])
            
        if 'base_filtering' in self.calib_dict:
            for fname in self.calib_dict['base_filtering']:
                filter = self.calib_dict['base_filtering'][fname]
                trans = filter_transmission(eV, filter['material'], filter['thickness'], filter['mass_density'])
                plt.plot(eV, trans, label=fname)

        plt.legend()
        plt.xlabel('Photon Energy [eV]')
        plt.ylabel('Transmission')
        plt.show(block=False)

        return
    
    def get_element_signals(self, shot_dict, debug=False):
        """Return the """

        shot_img,x,y = self.get_proc_shot(shot_dict) # this will set calib_dict if missing

        if 'filter_specs' not in self.calib_dict:
            self.calib_dict['filter_specs'] = self.load_filter_specs()

        if 'filter_mask' not in self.calib_dict:
            self.calib_dict['filter_mask'] = self.make_filter_mask()

        direct_fit = self.calc_direct_fit(shot_dict, debug=debug)

        # subtract a "background" element?
        bkg = 0
        for fid in self.calib_dict['filter_specs']:
            if self.calib_dict['filter_specs'][fid]['bkg_sub']:
                bkg = np.mean(shot_img[self.calib_dict['filter_mask']==fid])
        # This is still not right for lots of hard hits, as direct fit signal will be skewed

        element_signals = {}
        for fid in self.calib_dict['filter_specs']:
            element_direct_counts = np.mean(direct_fit[self.calib_dict['filter_mask']==fid])
            element_filtered_counts = np.mean(shot_img[self.calib_dict['filter_mask']==fid])
            element_signals[fid] = (element_filtered_counts - bkg) / element_direct_counts

        return element_signals

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