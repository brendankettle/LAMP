import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy import ndimage
import matplotlib.patches as patches
from LAMP.diagnostic import Diagnostic
from LAMP.utils.image_proc import ImageProc

#
# SHOULD THIS JUST BE ALL IN THE CAMERA DIAGNOSTIC?
#

class FocalSpot(Diagnostic):
    """Focal Spot camera
    """

    __version = 0.1
    __authors = ['Brendan Kettle']
    __requirements = ''
    data_type = 'image'

    curr_img = None
    img_units = 'Counts'
    x_mm, y_mm = None, None
    calib_dict = None

    def __init__(self, exp_obj, config_filepath):
        """Initiate parent base Diagnostic class to get all shared attributes and funcs"""
        super().__init__(exp_obj, config_filepath)
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
        plt.xlabel('[um]')
        plt.ylabel('[um]') 
        plt.axis('equal')
        plt.tight_layout()
        plt.show(block=False)

        return fig, plt.gca()
    
    def fit_spot(self):

        return
    
    def find_spot(self, shot_dict, box=None, mask=0.05, debug=False):
        """Use centre of mass to get x,y of spot"""
        # this will (hopefully) do a background correction etc.
        img, x, y = self.get_proc_shot(shot_dict)
        if img is None: # no image data?
            return None, None
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
        cx = bx1 + scx
        cy = by1 + scy
        if debug:
            plt.figure()
            # plot original image, not filtered
            im = plt.pcolormesh(x, y, img, vmin=np.percentile(img.flatten(),1), vmax=np.percentile(img.flatten(),99.99), shading='auto')
            plt.scatter(cx,cy, marker='+', color = 'red', s=int(np.mean([len(x),len(y)])/5))
            ax = plt.gca()
            rect = patches.Rectangle((bx1, by1), (bx2-bx1), (by2-by1), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            cb = plt.colorbar(im)
            plt.title(self.shot_string(shot_dict))
            plt.xlabel('?')
            plt.ylabel('?') 
            plt.axis('equal')
            plt.tight_layout()
            plt.show(block=False)
        return cx,cy
    
    def gauss2Dbeam(self,U,a0,a1,a2,a3,a4,a5):
        # a0 peak,
        # a2,a4 widths
        # a1,a3 centers
        # a5 angle
        f = a0*np.exp( -(
            ( U[:,0]*np.cos(a5)-U[:,1]*np.sin(a5) - a1*np.cos(a5)+a3*np.sin(a5) )**2/(2*a2**2) + 
            ( U[:,0]*np.sin(a5)+U[:,1]*np.cos(a5) - a1*np.sin(a5)-a3*np.cos(a5) )**2/(2*a4**2) ) )
        return f
    
    def gauss2DbeamFit(self,pG,U,I):
        f = self.gauss2Dbeam(U,*pG)
        fErr = np.sqrt(np.mean((f-I)**2))
        return fErr

    def fitBeam(self,x,y,img,r_max=100,pGuess = (1,0,.1,0,.1,0),tol=1e-4):

        (Ny,Nx) = np.shape(img)
        (X,Y) = np.meshgrid(x,y)

        # make beam mask
        R = np.sqrt((X)**2+(Y)**2)
        beamMask = (R<r_max)*(img>np.max(img*np.exp(-1)))

        I = img[beamMask]
        XY = np.zeros((np.size(I),2))
        XY[:,0] = X[beamMask]
        XY[:,1] = Y[beamMask]
        XYfull = np.zeros((np.size(X),2))
        XYfull[:,0] = X.flatten()
        XYfull[:,1] = Y.flatten()

        
        #(pFit,pcov) = sci.optimize.curve_fit(gauss2Dbeam, XY, I,p0=pGuess)
        a = (XY,I)
        z = optimize.minimize(self.gauss2DbeamFit,pGuess,args=a, tol=tol,method='Nelder-Mead')
        pFit = z.x
        Ibeam = self.gauss2Dbeam(XYfull,*pFit)

        imgBeam = np.reshape(Ibeam,(Ny,Nx))

        return imgBeam, pFit

    # MONTAGE FUNCTION

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
# iSel = imgBeam>(np.max(imgBeam/2))
# fwhm_energy = np.sum(img[iSel])/np.sum(img)
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
# x_plot = np.linspace(-rMax,rMax,num=1000)
# y_plot = np.linspace(-rMax,rMax,num=1000)
# [Xp,Yp] = np.meshgrid(x_plot,y_plot)
# XY = np.stack((Xp.flatten(),Yp.flatten()),axis=1)
# p = (np.mean(a_0),0,np.mean(w_x)*np.sqrt(2),0,np.mean(w_y)*np.sqrt(2),np.mean(theta))
# a_xy_avg = gauss2Dbeam(XY,*p)
# a_xy_avg = np.reshape(a_xy_avg,(1000,1000))

# Below is from FocalSpot.py from RR?
#
# import scipy.optimize as opt
#
# def two_d_gaussian(T, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
#     x=T[0]
#     y=T[1]
#     g = amplitude*np.exp(-calc_ellipse(x, y, xo, yo, sigma_x, sigma_y, theta)/2.0)+offset
#     # if np.sum(T-g)<=0:
#     #     return 10**10
#     # else:
#     return g.ravel()

# def calc_ellipse(x, y, xo, yo, sigma_x, sigma_y, theta):
#     a = (np.cos(theta)**2)/(sigma_x**2) + (np.sin(theta)**2)/(sigma_y**2)
#     b = 2.0*np.cos(theta)*np.sin(theta)*(1.0/(sigma_x**2)-1.0/(sigma_y**2))
#     c = (np.sin(theta)**2)/(sigma_x**2) + (np.cos(theta)**2)/(sigma_y**2)
#     g = a*((x-xo)**2) + b*(x-xo)*(y-yo) + c*((y-yo)**2)
#     return g

# def calc_moments_spot(img_array_2D, axs, peak_amp, axs_index):
#     """

#     """
#     im_summed_ax=np.sum(img_array_2D, axis=axs_index)
#     im_summed_ax[np.where(im_summed_ax<0.36*peak_amp)]=0.0
#     central_pos=np.trapz(im_summed_ax*axs, axs)/np.trapz(im_summed_ax, axs)
#     sigma_ax=(np.trapz(im_summed_ax*axs**2, axs)/np.trapz(im_summed_ax, axs)-central_pos**2)**0.5
#     return central_pos, sigma_ax


# def convert_width_to_FHWM(width):
#     return width*(2.0*np.log(2.0))**0.5

# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

# class FocalSpot:
#     """

#     """
# #amplitude, xo, yo, sigma_x, sigma_y, theta, offset
#     def __init__(self, focal_pos_x=None, focal_pos_x_err=None, focal_pos_y=None, focal_pos_y_err=None, focal_pos_z=None, focal_pos_z_err=None, FWHM_x=None, FWHM_x_err=None, FWHM_y=None, FWHM_y_err=None, angle_rot=None, angle_rot_err=None, energy_frac_FWHM=None, energy_frac_FWHM_err=None, microns_per_pixel=None, microns_per_pixel_err=None):
#         self.focal_pos_x=focal_pos_x
#         self.focal_pos_x_err=focal_pos_x_err
#         self.focal_pos_y=focal_pos_y
#         self.focal_pos_y_err=focal_pos_y_err
#         self.focal_pos_z=focal_pos_z
#         self.focal_pos_z_err=focal_pos_z_err
#         self.FWHM_x=FWHM_x
#         self.FWHM_x_err=FWHM_x_err
#         self.FWHM_y=FWHM_y
#         self.FWHM_y_err=FWHM_y_err
#         self.angle_rot=angle_rot
#         self.angle_rot_err=angle_rot_err
#         self.energy_frac_FWHM=energy_frac_FWHM
#         self.energy_frac_FWHM_err=energy_frac_FWHM_err
#         self.microns_per_pixel=microns_per_pixel
#         self.microns_per_pixel_err=microns_per_pixel_err

#     def get_spot_properties_lst_sqrd_fit(self, im):
#         """

#         """
#         y_max, x_max=im.shape
#         x = np.linspace(0, x_max, x_max)#*microns_per_pixel
#         y = np.linspace(0, y_max, y_max)#*microns_per_pixel
#         X, Y = np.meshgrid(x, y)
#         bkg_counts=np.mean(im[0:277, 0:44])
#         im=im-bkg_counts

#         # estimates for fitted elliptical gaussian properties
#         peak_amp=np.amax(im)
#         centre_x_pos, sigma_x=calc_moments_spot(im, x, peak_amp, 0)
#         centre_y_pos, sigma_y=calc_moments_spot(im, y, peak_amp, 1)
#         angle_rot=np.arctan(sigma_x/sigma_y)
#         initial_guess = [peak_amp*0.9,centre_x_pos,centre_y_pos,sigma_x,sigma_y,angle_rot,bkg_counts]

#         # least squares fit focal spot to elliptical gaussian
#         popt, pcov = opt.curve_fit(two_d_gaussian, [X, Y], im.flatten(), p0=initial_guess, bounds=(0, 5000))

#         # calculate energy in FWHM
#         ellipse=calc_ellipse(X, Y, popt[1], popt[2], popt[3], popt[4], popt[5])
#         counts_in_FWHM=np.sum(im[np.where(ellipse<1.0)])
#         frac_total_counts_in_FWHM=counts_in_FWHM/np.sum(im)

#         #convert spatial focal spot properties to micron units
#         popt[1:5]=popt[1:5]*self.microns_per_pixel
#         popt=list(popt)
#         popt.append(frac_total_counts_in_FWHM)
#         return popt

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
