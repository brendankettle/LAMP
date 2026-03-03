from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy import ndimage
import matplotlib.patches as patches
from LAMP.diagnostic import Diagnostic
from LAMP.utils.image_proc import ImageProc
from LAMP.utils.general import mindex

class FocalSpot(Diagnostic):
    """Focal Spot camera
    """

    # To Do - errors!
    # To Do - mroe details on fitting for debug?
    # To Do - LASY integration? or handoff?

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = 'scipy'
    data_type = 'image'

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
        
        # All of these below need to be defined here, rather than above as a class attribute, so they don't change on different instances of diagnostics
        self.curr_img = None
        self.img_units = 'Counts'
        self.x_units, self.y_units = 'px','px'
    
        return
    
    def plot_proc_shot(self, shot_dict, calib_id=None, vmin=None, vmax=None, debug=False):
        """Could definitely be general plot function?"""
        img, x, y = self.get_proc_shot(shot_dict, calib_id=calib_id, debug=debug)

        if vmin is None:
            vmin = np.nanmin(img)
        if vmax is None:
            vmax = np.nanmax(img)

        fig = plt.figure()
        im = plt.pcolormesh(x, y, img, vmin=vmin, vmax=vmax, shading='auto')
        cb = plt.colorbar(im)
        cb.set_label(self.img_units, rotation=270, labelpad=20)
        plt.title(self.shot_string(shot_dict))
        plt.xlabel(self.x_units) 
        plt.ylabel(self.y_units) 
        plt.axis('equal')
        plt.tight_layout()
        plt.show(block=False)

        return fig, plt.gca()
    
    def find_spot(self, img, x=None, y=None, box=None, mask=0.05, debug=False):
        """Use centre of mass to get x,y of spot"""
        # To Do: could make it so "box" could be a mask??

        # if a dictionary or string is passed, get img data (and x/y cords)
        if isinstance(img, dict) or isinstance(img, str) or isinstance(img, Path):
            img, x, y = self.get_proc_shot(img,debug=debug)

        if img is None: # no image data?
            return None, None
        
        # no x/y? use calibration? or pixels
        if x is None:
            if self.x is None:
                x = np.arange(np.shape(img)[1])
            else:
                x = self.x
        if y is None:
            if self.y is None:
                y = np.arange(np.shape(img)[0])
            else:
                y = self.y

        fimg = img.copy()
        # mask any low level pixels to zero, to help with large area level over backgrounds
        fimg[fimg < (np.max(fimg)*mask)] = 0
        # then a median filter to cut hard hits
        fimg = ndimage.median_filter(fimg,size=5)
        # find first rough position using coarse centre of mass
        fcy,fcx = ndimage.center_of_mass(fimg)
        # then do another centre of mass, within this box (to remove error from large scale variations in image)
        if not box:
            box = int(np.mean([len(x),len(y)])/4) # default to 1/4 of image width/length
        bx1 = int(fcx-(box/2))
        bx2 = int(fcx+(box/2))
        by1 = int(fcy-(box/2))
        by2 = int(fcy+(box/2))
        img_roi = fimg[by1:by2,bx1:bx2]
        scy,scx = ndimage.center_of_mass(img_roi)
        cx = x[int(bx1 + scx)] # we want it in real units
        cy = y[int(by1 + scy)] # we want it in real units

        # def calcCOW(img,X,Y,img_thresh=0.5):
        #     iSel = img>img_thresh
        #     c_x = np.sum(X[iSel]*img[iSel])/np.sum(img[iSel])
        #     c_y = np.sum(Y[iSel]*img[iSel])/np.sum(img[iSel])
        #     return c_x,c_y

        if debug:
            plt.figure()
            # plot original image, not filtered
            im = plt.pcolormesh(x, y, img, vmin=np.percentile(img.flatten(),1), vmax=np.percentile(img.flatten(),99.99), shading='auto')
            plt.scatter(cx,cy, marker='+', color = 'red', s=int(np.mean([len(x),len(y)])/5))
            ax = plt.gca()
            rect = patches.Rectangle((x[bx1], y[by1]), (x[bx2]-x[bx1]), (y[by2]-y[by1]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            cb = plt.colorbar(im)
            plt.xlabel(self.x_units)
            plt.ylabel(self.y_units) 
            plt.axis('equal')
            plt.tight_layout()
            plt.show(block=False)

        return cx,cy  

    def gauss2D(self, U, a0, a1, a2, a3, a4, a5):
        # a0 peak,
        # a1,a3 centers
        # a2,a4 widths
        # a5 angle (radians)
        # background offset?
        f = a0*np.exp( -(
            ( U[:,0]*np.cos(a5)-U[:,1]*np.sin(a5) - a1*np.cos(a5)+a3*np.sin(a5) )**2/(2*a2**2) + 
            ( U[:,0]*np.sin(a5)+U[:,1]*np.cos(a5) - a1*np.sin(a5)-a3*np.cos(a5) )**2/(2*a4**2) ) )
        
        # below is the same as above
        # x = U[:,0]
        # y = U[:,1]
        # amplitude = a0
        # x0 = a1
        # y0 = a3
        # sigma_x = a2
        # sigma_y = a4
        # theta = a5
        # cos_t = np.cos(theta)
        # sin_t = np.sin(theta)
        # a = (cos_t**2) / (2*sigma_x**2) + (sin_t**2) / (2*sigma_y**2)
        # b = (-sin_t*cos_t) / (2*sigma_x**2) + (sin_t*cos_t) / (2*sigma_y**2)
        # c = (sin_t**2) / (2*sigma_x**2) + (cos_t**2) / (2*sigma_y**2)
        # f = amplitude * np.exp(-(a*(x - x0)**2 + 2*b*(x - x0)*(y - y0) + c*(y - y0)**2))

        return f
    
    def gauss2D_fitfunc(self, pG, U, I):
        f = self.gauss2D(U,*pG)
        fErr = np.sqrt(np.mean((f-I)**2))
        return fErr
    
    def fit_gauss2D(self, img, x, y, p_guess=None, r_max=100, tol=1e-4, debug=False): # bounds?

        (Ny,Nx) = np.shape(img)
        (X,Y) = np.meshgrid(x,y)

        if p_guess is None:
            cx, cy = self.find_spot(img, debug=debug)
            # a0 peak; a1,a3 centers; a2,a4 widths; a5 angle
            p_guess = (np.max(img), cx, (abs(np.max(x)-np.min(x))/10), cy, (abs(np.max(y)-np.min(y))/10), 0)
            #print(p_guess)

        # make beam mask
        R = np.sqrt((X-cx)**2+(Y-cy)**2) # some radius
        beamMask = (R<r_max)*(img>np.max(img*np.exp(-1))) # but also only use 1/e intensity (for fit)
        I = img[beamMask].flatten()
        #I = img.copy().flatten()
        # To Do: Some debugging of mask

        XY = np.zeros((np.size(I),2))
        XY[:,0] = X[beamMask].flatten()#[beamMask]
        XY[:,1] = Y[beamMask].flatten()#[beamMask]
        XYfull = np.zeros((np.size(X),2))
        XYfull[:,0] = X.flatten()
        XYfull[:,1] = Y.flatten()

        #(pFit,pcov) = sci.optimize.curve_fit(gauss2Dbeam, XY, I, p0=pGuess)
        a = (XY,I)
        bnds = ((0,None),(None,None),(None,None),(None,None),(None,None),(None,None))
        z = optimize.minimize(self.gauss2D_fitfunc, p_guess, args=a, tol=tol, bounds=bnds, method='Nelder-Mead')
        pFit = z.x
        #print(pFit)
        Ibeam = self.gauss2D(XYfull,*pFit)

        imgBeam = np.reshape(Ibeam,(Ny,Nx))

        return imgBeam, pFit
    
    def fit_spot(self, img, x=None, y=None, p_guess=None, r_max=100, tol=1e-4, debug=False):

        # if a dictionary or string is passed, get img data (and x/y cords)
        if isinstance(img, dict) or isinstance(img, str) or isinstance(img, Path):
            img, x, y = self.get_proc_shot(img,debug=debug)

        if img is None: # no image data?
            return None, None
        
        # no x/y? use calibration? or pixels
        if x is None:
            if self.x is None:
                x = np.arange(np.shape(img)[1])
            else:
                x = self.x
        if y is None:
            if self.y is None:
                y = np.arange(np.shape(img)[0])
            else:
                y = self.y

        img_fit, p_fit = self.fit_gauss2D(img, x, y, p_guess=p_guess, r_max=r_max, tol=tol) # r_max=r_max

        if debug:
            cx = p_fit[1]
            cy = p_fit[3]
            gFWHM = 2 *np.sqrt(2 * np.log(2))
            fwhm_x = p_fit[2] * gFWHM
            fwhm_y = p_fit[4] * gFWHM
            xmin = cx - r_max
            xmax = cx + r_max
            ymin = cy - r_max
            ymax = cy + r_max
            FWHM_ellipse1 = patches.Ellipse(xy=(cx, cy), width=fwhm_x, height=fwhm_y, edgecolor='r', fc='None', linewidth=2)
            FWHM_ellipse2 = patches.Ellipse(xy=(cx, cy), width=fwhm_x, height=fwhm_y, edgecolor='r', fc='None', linewidth=2) # this complains if you try attach the same patch twice, can you copy?
            FWHM_ellipse3 = patches.Ellipse(xy=(cx, cy), width=fwhm_x, height=fwhm_y, edgecolor='r', fc='None', linewidth=2) 

            # data and fit side by side?
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            img_roi = img[mindex(y,ymin):mindex(y,ymax),mindex(x,xmin):mindex(x,xmax)]
            img_fit_roi = img_fit[mindex(y,ymin):mindex(y,ymax),mindex(x,xmin):mindex(x,xmax)]
            img_diff_roi = img_fit_roi - img_roi
            im1 = ax1.pcolormesh(x[mindex(x,xmin):mindex(x,xmax)], y[mindex(y,ymin):mindex(y,ymax)], img_roi, vmin=np.percentile(img_roi.flatten(),1), vmax=np.percentile(img_roi.flatten(),99.99), shading='auto')
            im2 = ax2.pcolormesh(x[mindex(x,xmin):mindex(x,xmax)], y[mindex(y,ymin):mindex(y,ymax)], img_fit_roi, vmin=np.percentile(img_fit_roi.flatten(),1), vmax=np.percentile(img_fit_roi.flatten(),99.99), shading='auto')
            im3 = ax3.pcolormesh(x[mindex(x,xmin):mindex(x,xmax)], y[mindex(y,ymin):mindex(y,ymax)], img_diff_roi, vmin=np.percentile(img_diff_roi.flatten(),1), vmax=np.percentile(img_diff_roi.flatten(),99.99), shading='auto')
            cb = plt.colorbar(im2)
            cb = plt.colorbar(im3)
            ax1.add_patch(FWHM_ellipse1)
            ax2.add_patch(FWHM_ellipse2)
            ax3.add_patch(FWHM_ellipse3)
            ax1.set_title('Data')
            ax2.set_title('Fit')
            ax3.set_title('Difference')
            ax1.set_xlabel(self.x_units)
            ax1.set_ylabel(self.y_units) 
            ax2.set_xlabel(self.x_units)
            ax2.set_ylabel(self.y_units) 
            ax3.set_xlabel(self.x_units)
            ax3.set_ylabel(self.y_units) 
            ax1.axis('equal')
            ax2.axis('equal')
            ax3.axis('equal')
            plt.show(block=False)

        return img_fit, p_fit

    def anaylse_spot(self, img, x=None, y=None, r_max=100, debug=False):

        # if a dictionary or string is passed, get img data
        if isinstance(img, dict) or isinstance(img, str) or isinstance(img, Path):
            shot_dict = img
            img, x, y = self.get_proc_shot(img,debug=debug)

        # no x/y? use calibration? or pixels
        if x is None:
            if self.x is None:
                x = np.arange(np.shape(img)[1])
            else:
                x = self.x
        if y is None:
            if self.y is None:
                y = np.arange(np.shape(img)[0])
            else:
                y = self.y

        metrics = {}

        # fit spot
        img_fit, p_fit = self.fit_spot(img, x, y, r_max=r_max, debug=debug)
        if debug:
            plt.suptitle(shot_dict)

        gFWHM = 2 *np.sqrt(2 * np.log(2))
        metrics['fwhm_x'] = p_fit[2] * gFWHM
        metrics['fwhm_y'] = p_fit[4] * gFWHM
        metrics['centre_x'] = p_fit[1] 
        metrics['centre_y'] = p_fit[3] 
        metrics['angle'] = p_fit[5] 

        # energy fraction? (In ROI)
        xmin = metrics['centre_x'] - r_max
        xmax = metrics['centre_x'] + r_max
        ymin = metrics['centre_y'] - r_max
        ymax = metrics['centre_y'] + r_max
        img_roi = img[mindex(y,ymin):mindex(y,ymax),mindex(x,xmin):mindex(x,xmax)]
        img_fit_roi = img_fit[mindex(y,ymin):mindex(y,ymax),mindex(x,xmin):mindex(x,xmax)]
        iSel = img_roi>=(np.max(img_fit_roi/2))
        metrics['fwhm_energy'] = np.sum(img_roi[iSel])/np.sum(img_roi)

        # calculate a0 map?
        # laser_sigma = laser_duration_FWHM/(2*np.sqrt(2*np.log(2))) # laser_duration_FWHM = 45e-15
        # laser_peak_power = laser_energy/np.sqrt(2*np.pi*laser_sigma**2)  # laser_energy in J
        # W_per_m2_per_count = laser_peak_power/(np.sum(img)*(um_per_pix*1e-6)**2)
        # from scipy.constants import c, epsilon_0, pi, m_e, e
        # lambda_0 = 800e-9
        # omega_0 = 2*pi*c/lambda_0
        # img_a_fit = np.sqrt(2*(img_fit*W_per_m2_per_count)/(c*epsilon_0))*e/(m_e*omega_0*c)
        # a0 = np.max(img_a_fit) ??

        return metrics

    def get_spot_stats(self, shot_dicts, debug=False):
        # median +/- std is better than mean for quoted values?
        # FWHMs, angle, energy fraction FWHM, a0?
        
        stats = {}
        for shot_dict in shot_dicts:
            if debug:
                print(f'Analysing {shot_dict}')
            metrics = self.anaylse_spot(shot_dict, debug=debug)
            for mname in metrics:
                if mname in stats:
                    stats[mname]['values'].append(metrics[mname])
                else:
                    stats[mname] = {'values':  [metrics[mname]]}
                
        # make stats
        for mname in stats:
            stats[mname]['mean'] = np.mean(stats[mname]['values'])
            stats[mname]['median'] = np.median(stats[mname]['values'])
            stats[mname]['std'] = np.std(stats[mname]['values'])

        return stats

    def plot_spot_stats(self, shot_dicts, debug=False):

        stats = self.get_spot_stats(shot_dicts, debug=debug)

        print(f"FWHM X (Mean +/- Std Dev.): {stats['fwhm_x']['mean']:.1f} +/- {stats['fwhm_x']['std']:.1f} {self.x_units} (Median={stats['fwhm_x']['median']:.1f})")
        print(f"FWHM Y (Mean +/- Std Dev.): {stats['fwhm_y']['mean']:.1f} +/- {stats['fwhm_y']['std']:.1f} {self.y_units} (Median={stats['fwhm_y']['median']:.1f})")
        print(f"Centre X (Mean +/- Std Dev.): {int(stats['centre_x']['mean'])} +/- {int(stats['centre_x']['std'])} {self.x_units} (Median={int(stats['centre_x']['median'])})")
        print(f"Centre Y (Mean +/- Std Dev.): {int(stats['centre_y']['mean'])} +/- {int(stats['centre_y']['std'])} {self.y_units} (Median={int(stats['centre_y']['median'])})")
        print(f"Angle (Mean +/- Std Dev.): {np.rad2deg(stats['angle']['mean']):.1f} +/- {np.rad2deg(stats['angle']['std']):.1f} degrees (Median={np.rad2deg(stats['angle']['median']):.1f})")
        print(f"Energy in FWHM (Mean +/- Std Dev.): {(stats['fwhm_energy']['mean']*100):.1f} +/- {(stats['fwhm_energy']['std']*100):.1f}% (Median={(stats['fwhm_energy']['median']*100):.1f})")

        plt.figure()
        plt.scatter(np.arange(len(stats['fwhm_x']['values'])), stats['fwhm_x']['values'])
        plt.ylabel(self.x_units)
        plt.title('FWHM X')
        plt.show(block=False)

        plt.figure()
        plt.scatter(np.arange(len(stats['fwhm_y']['values'])), stats['fwhm_y']['values'])
        plt.ylabel(self.y_units)
        plt.title('FWHM Y')
        plt.show(block=False)

        plt.figure()
        plt.scatter(np.arange(len(stats['fwhm_energy']['values'])), np.array(stats['fwhm_energy']['values'])*100)
        plt.ylabel('%')
        plt.title('FWHM Energy %')
        plt.show(block=False)
        
        plt.figure()
        plt.scatter(np.arange(len(stats['centre_x']['values'])), stats['centre_x']['values'])
        plt.ylabel(self.x_units)
        plt.title('Centre X')
        plt.show(block=False)
        
        plt.figure()
        plt.scatter(np.arange(len(stats['centre_y']['values'])), stats['centre_y']['values'])
        plt.ylabel(self.y_units)
        plt.title('Centre Y')
        plt.show(block=False)

        plt.figure()
        plt.scatter(np.arange(len(stats['angle']['values'])), np.rad2deg(stats['angle']['values']))
        plt.ylabel('Degrees')
        plt.title('Angle')
        plt.show(block=False)

        return
    
    # MONTAGE FUNCTION?

# lambda_0 = 800e-9
# omega_0 = 2*pi*c/lambda_0
# laser_energy = 9 # J
# laser_peak_power = 134e12*laser_energy/6.3 # from previous pulse shape measurements (2017 Xanes)
# ...
# W_per_m2_per_count = laser_peak_power/(np.sum(img)*(um_per_pix*1e-6)**2)
# img_W_per_m2 = img*W_per_m2_per_count
# E_V_per_m = np.sqrt(2*img_W_per_m2/(c*epsilon_0))
# img_a = e*E_V_per_m/(m_e*omega_0*c)
# ...
# img_a_fit = np.sqrt(2*(imgBeam*W_per_m2_per_count)/(c*epsilon_0))*e/(m_e*omega_0*c)
# ...
# a_0.append(np.max(median_filter(img_a,(3))))
# a_0_fit.append(np.max(img_a_fit))
# w_x.append(pFit[2])
# w_y.append(pFit[4])
# theta.append(pFit[5])
# ...
# print(f'Mean measured a_0 = {np.mean(a_0):3.03f} +- {np.std(a_0):3.03f}')
# print(f'Mean fitted gaussian a_0 = {np.mean(a_0_fit):3.03f} +- {np.std(a_0_fit):3.03f}')
# print(f'Spot width intensity 1/e^2 = {np.mean(w_x):3.02f} +- {np.std(w_x):3.02f} X '
#      + f'{np.mean(w_y):3.02f} +- {np.std(w_y):3.02f} microns (x,y)')
# print(f'Mean angle of spot to cam horizontal {np.mean(theta*180/pi):3.03f} degrees')
# print(f'Mean fwhm energy fraction {np.mean(fwhm_energy):3.03f} +- {np.std(fwhm_energy):3.03f}')
# ...

# class Laser:
#     """

#     """
#     def __init__(self, wavelength, refractive_index, FWHM_t, FWHM_t_err, f_number, energy, energy_err, throughput, throughput_err, a0=None, a0_err=None, focal_pos_x=None, focal_pos_x_err=None, focal_pos_y=None, focal_pos_y_err=None, focal_pos_z=None, focal_pos_z_err=None, FWHM_x=None, FWHM_x_err=None, FWHM_y=None, FWHM_y_err=None, angle_rot=None, angle_rot_err=None, energy_frac_FWHM=None, energy_frac_FWHM_err=None, microns_per_pixel=None, microns_per_pixel_err=None):
#         self.l0=wavelength
#         self.n=refractive_index
#         self.FWHM_t=FWHM_t
#         self.FWHM_t_err=FWHM_t_err
#         self.f_number=f_number
#         self.energy=energy
#         self.energy_err=energy_err
#         self.throughput=throughput
#         self.throughput_err=throughput_err
#         self.a0=a0
#         self.a0_err=a0_err
#         #focal_spot=FocalSpot(focal_pos_x, focal_pos_x_err, focal_pos_y, focal_pos_y_err, focal_pos_z, focal_pos_z_err, FWHM_x, FWHM_x_err, FWHM_y, FWHM_y_err, angle_rot, angle_rot_err, energy_frac_FWHM, energy_frac_FWHM_err, microns_per_pixel, microns_per_pixel_err)
#         self.focal_spot=FocalSpot(focal_pos_x, focal_pos_x_err, focal_pos_y, focal_pos_y_err, focal_pos_z, focal_pos_z_err, FWHM_x, FWHM_x_err, FWHM_y, FWHM_y_err, angle_rot, angle_rot_err, energy_frac_FWHM, energy_frac_FWHM_err, microns_per_pixel, microns_per_pixel_err)

#     def calc_waist(self, z, w0, M, z0):
#         waist=(w0**2+M**4*(self.l0/(np.pi*w0))**2*(z-z0)**2)**0.5#w0*np.sqrt(1.0+(((z-focal_pos_z)/Zr)*((z-focal_pos_z)/Zr)))
#         return waist

#     def calc_Raleigh_Range(self, w0):
#         return w0*w0*np.pi/self.l0*self.n

#     def calc_peak_intensity(self):
#         FWHM_x_cm=self.focal_spot.FWHM_x/10**4
#         FWHM_y_cm=self.focal_spot.FWHM_y/10**4
#         peak_intensity_W_per_cm2=self.energy*self.throughput*(self.focal_spot.energy_frac_FWHM/0.5)*(4.0*np.log(2.0)/np.pi)**1.5/(self.FWHM_t*FWHM_x_cm*FWHM_y_cm)
#         peak_intensity_W_per_cm2_percentage_err=((self.energy_err/self.energy)**2+(self.throughput_err/self.throughput)**2+(self.focal_spot.energy_frac_FWHM_err/self.focal_spot.energy_frac_FWHM)**2+(self.FWHM_t_err/self.FWHM_t)**2+(self.focal_spot.FWHM_x_err/self.focal_spot.FWHM_x)**2+(self.focal_spot.FWHM_y_err/self.focal_spot.FWHM_y)**2)**0.5
#         return peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_percentage_err*peak_intensity_W_per_cm2

#     def calc_a0(self):
#         #l0 in microns, peak intensity in Wcm^-2
#         peak_intensity_W_per_cm2, peak_intensity_W_per_cm2_err=self.calc_peak_intensity()
#         a0=0.855*self.l0*(peak_intensity_W_per_cm2/10**18)**0.5
#         return a0, a0*peak_intensity_W_per_cm2_err/peak_intensity_W_per_cm2
