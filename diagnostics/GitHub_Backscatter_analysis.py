import numpy as np
import cv2
from skimage.io import imread
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import find_peaks




#######CALIBRATIONS#####



light_percentage =  0.0000946 #0.0000946% light passes through 
laser_shot_total_energy_J = 25
calibration_pixel_to_J = 1.2846944932107116e-13

# Define reference pixel coordinates and real-world coordinates for image transformation
imgP_pix_ref = np.array([[113, 33], [458, 7], [473, 365], [106, 399]])  # Pixel values
imgP_real = np.array([[0, 150], [150, 150], [150, 0], [0, 0]])  # Real-world coordinates in mm
top_zero = np.array([113, 33])  # Reference measurement of the top left corner


#######TRANSFORM IMAGE#####


base_dir = '/Users/temourfoster/Desktop/Imperial_NLCS_analysis/github/test_input'
output_dir = '/Users/temourfoster/Desktop/Imperial_NLCS_analysis/github/test_output'


# Define new image dimensions and resolution
Nx_new, Ny_new = 1000, 1000
xRange, yRange = 150, 150
x0, y0 = 0, 0
pixel_to_mm_x, pixel_to_mm_y = xRange / Nx_new, yRange / Ny_new
x_mm = x0 + np.linspace(0, Nx_new, num=Nx_new) * pixel_to_mm_x
y_mm = y0 + np.linspace(0, Ny_new, num=Ny_new) * pixel_to_mm_y

# Print pixel calibration
pixel_to_mm = pixel_to_mm_x
print(f"Pixel size in x-direction: {pixel_to_mm_x:.3f} mm/pixel")
print(f"Pixel size in y-direction: {pixel_to_mm_y:.3f} mm/pixel")


# Function to process a single image
def process_image(img_path, output_path):
    img = imread(img_path).astype(float)
    Ny, Nx = img.shape

    # Calculate shift and update pixel coordinates
    p_shift = top_zero - imgP_pix_ref[0]
    imgP_pix = imgP_pix_ref + p_shift

    # Calculate transformation matrix (homography)
    imgP_trans = (imgP_real - [x0, y0]) / [pixel_to_mm_x, pixel_to_mm_y]
    H, status = cv2.findHomography(imgP_pix, imgP_trans)

    # Calculate pixel areas in the original image
    retval, H_inv = cv2.invert(H)
    (X, Y) = np.meshgrid(x_mm, y_mm)
    X_raw = cv2.warpPerspective(X, H_inv, (Nx, Ny))
    Y_raw = cv2.warpPerspective(Y, H_inv, (Nx, Ny))
    imgArea0 = np.abs(np.gradient(X_raw, axis=1) * np.gradient(Y_raw, axis=0))
    imgArea0 = np.median(imgArea0[imgArea0 > 0])

    # Normalize image and apply transformation
    imgCountsPerArea = img / imgArea0
    imgCountsPerArea[imgArea0 == 0] = 0
    imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
    imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

    im_out = cv2.warpPerspective(img, H, (Nx_new, Ny_new)) * pixel_to_mm_x * pixel_to_mm_y

    # Save transformed image to the specified output file
    tiff.imwrite(output_path, im_out.astype(np.float32))

# Process each image in the base_dir and save to output_dir
for img_file in Path(base_dir).glob('*.tiff'):  # Adjust the file extension as needed
    output_file = Path(output_dir) / img_file.name  # Keep the same filename
    process_image(img_file, output_file)

print("Image transformation completed for all input files.")



#######CONVERT IMAGE#####


def integrate_and_plot_single_image(input_file, start_x, start_y):
    # Read the single image
    image = tiff.imread(input_file)

    # Initialize the summed image as just the input image
    summed_image = image

    # Plot the full summed image
    vmin = summed_image.min()
    vmax = summed_image.max()

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 1, 1)
    img1 = ax1.imshow(summed_image, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Image from Output File')
    
    
    # Set axis labels and title
    ax1.set_xlabel('Pixels [no units]')
    ax1.set_ylabel('Pixels [no units]')
    ax1.set_title('Transformed image with ROI selected')
    
    # Create colorbar and set label
    cbar = plt.colorbar(img1, ax=ax1, orientation='horizontal')
    cbar.set_label('Intensity [counts]')
    
    # Plotting lines for the specified area
    x_len = 500 
    y_len = 500
    
    plt.axvline(start_x, color='red')
    plt.axhline(start_y, color='red')
    plt.axvline(start_x + x_len, color='red')
    plt.axhline(start_y + y_len, color='red')
    

    #plt.tight_layout()
    plt.show()
    
    return summed_image


start_x = 160
start_y = 200
summed_image = integrate_and_plot_single_image(output_file, start_x, start_y)



#######ANALYSE IMAGE#####




import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def analyze_and_plot_zoomed_image(summed_image, zoom_region, peak_intensity_threshold=100):
    if zoom_region is None:
        print("No zoom region provided.")
        return

    x, y, width, height = zoom_region
    zoomed_image = summed_image[y:y+height, x:x+width]
    center_x_lineout = zoomed_image[height // 2, :]
    center_y_lineout = zoomed_image[:, width // 2]
    
    # Calculate and print summed pixel intensity of the zoomed region + energy calibration
    total_pixel_intensity = np.sum(zoomed_image)
    actual_intensity = (total_pixel_intensity / light_percentage) * 100 #0.0000946% light passes through   

    

    energy_screen = calibration_pixel_to_J * actual_intensity
#    energy_deposited_gas = laser_shot_total_energy_J - energy_screen  # energy deposited twice; once initial electron acceleration, 2nd from reflection ###NEEDS TO BE FIXED WITH BEAM DUMP DATA
    
    print(f'Total energy of zoomed region: {energy_screen:.2f} J')
#    print(f'Total energy deposited in gas: {energy_deposited_gas:.2f} J')###NEEDS TO BE FIXED WITH BEAM DUMP DATA
    

    
    
    
    x_mm = np.arange(zoomed_image.shape[1]) * pixel_to_mm
    y_mm = np.arange(zoomed_image.shape[0]) * pixel_to_mm

    # Analyze horizontal lineout
    peaks_x, properties_x = find_peaks(center_x_lineout, height=peak_intensity_threshold, distance=50)
    threshold_positions_x_orig = find_threshold_positions(center_x_lineout, peaks_x, properties_x['peak_heights'])
    threshold_positions_x = (threshold_positions_x_orig[:-1])
    first_tuple = threshold_positions_x[0]
    first_element = first_tuple[0]
    threshold_positions_x1 = np.array([first_element])
    threshold_positions_x2 = np.array([threshold_positions_x_orig[-1][-1]])
    
    
    # Analyze vertical lineout
    peaks_y, properties_y = find_peaks(center_y_lineout, height=peak_intensity_threshold, distance=50)
    threshold_positions_y_orig = find_threshold_positions(center_y_lineout, peaks_y, properties_y['peak_heights'])
    threshold_positions_y = (threshold_positions_y_orig[:-1])
    first_tuple = threshold_positions_y[0]
    first_element = first_tuple[0]
    threshold_positions_y1 = np.array([first_element])
    threshold_positions_y2 = np.array([threshold_positions_y_orig[-1][-1]])

    # Convert threshold positions to mm
    threshold_positions_x1_mm = threshold_positions_x1 * pixel_to_mm
    threshold_positions_x2_mm = threshold_positions_x2 * pixel_to_mm
    threshold_positions_y1_mm = threshold_positions_y1 * pixel_to_mm
    threshold_positions_y2_mm = threshold_positions_y2 * pixel_to_mm
    
    spot_size_x = threshold_positions_x2_mm[0] - threshold_positions_x1_mm[0]
    spot_size_y = threshold_positions_y2_mm[0] - threshold_positions_y1_mm[0]
    

    
    print(f'X spot size diameter in mm: {spot_size_x:.2f} mm')
    print(f'Y spot size diameter in mm: {spot_size_y:.2f} mm')

    # Calculate energy spectrum
    energy_spectrum = zoomed_image * calibration_pixel_to_J * (pixel_to_mm ** 2)

    plt.figure(figsize=(20, 12))  # Adjusted figure size

    ax1 = plt.subplot(3, 4, 1)
    img1 = ax1.imshow(summed_image, cmap='gray', vmin=summed_image.min(), vmax=summed_image.max())
    ax1.set_title('Full Summed Image')
#     plt.colorbar(img1, ax=ax1, orientation='horizontal')
    # Set axis labels and title
    ax1.set_xlabel('Pixels [no units]')
    ax1.set_ylabel('Pixels [no units]')
    ax1.set_title('Transformed image with ROI selected')
    
    # Create colorbar and set label
    cbar = plt.colorbar(img1, ax=ax1, orientation='horizontal')
    cbar.set_label('Intensity [counts]')
    
    
    

    ax2 = plt.subplot(3, 4, 2)
    img2 = ax2.imshow(zoomed_image, cmap='gray', vmin=summed_image.min(), vmax=summed_image.max())
    ax2.axhline(height // 2, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(width // 2, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(threshold_positions_x1, color='yellow', linestyle='--', alpha=0.5)
    ax2.axvline(threshold_positions_x2, color='yellow', linestyle='--', alpha=0.5)
    ax2.axhline(threshold_positions_y1, color='yellow', linestyle='--', alpha=0.5)
    ax2.axhline(threshold_positions_y2, color='yellow', linestyle='--', alpha=0.5)
    ax2.set_title('Zoomed View with Center Lines')
#     plt.colorbar(img2, ax=ax2, orientation='horizontal')

    ax2.set_xlabel('Pixels [no units]')
    ax2.set_ylabel('Pixels [no units]')
    cbar = plt.colorbar(img2, ax=ax2, orientation='horizontal')
    cbar.set_label('Intensity [counts]')



    # Plot horizontal lineout
    ax4 = plt.subplot(3, 4, 5)
    ax4.plot(x_mm, center_x_lineout)
    ax4.plot(x_mm[peaks_x], center_x_lineout[peaks_x], "x")
    ax4.axvline(threshold_positions_x1_mm, color='blue', linestyle=':', alpha=0.7)
    ax4.axvline(threshold_positions_x2_mm, color='blue', linestyle=':', alpha=0.7)
    ax4.set_title('Horizontal Lineout at Zoom Center')
    ax4.set_xlabel('Position [mm]')
    ax4.set_ylabel('Intensity [counts]')
    ax4.set_ylim(min(center_x_lineout.min(), center_y_lineout.min()), max(center_x_lineout.max(), center_y_lineout.max()) + 100)

    # Plot vertical lineout
    ax5 = plt.subplot(3, 4, 6)
    ax5.plot(y_mm, center_y_lineout)
    ax5.plot(y_mm[peaks_y], center_y_lineout[peaks_y], "x")
    ax5.axvline(threshold_positions_y1_mm, color='blue', linestyle=':', alpha=0.7)
    ax5.axvline(threshold_positions_y2_mm, color='blue', linestyle=':', alpha=0.7)
    ax5.set_title('Vertical Lineout at Zoom Center')
    ax5.set_xlabel('Position [mm]')
    ax5.set_ylabel('Intensity [counts]')
    ax5.set_ylim(min(center_x_lineout.min(), center_y_lineout.min()), max(center_x_lineout.max(), center_y_lineout.max()) + 100)

    plt.tight_layout()
    plt.show()
    
    # Plot energy spectrum
    ax3 = plt.subplot()
    img3 = ax3.imshow(energy_spectrum, cmap='hot', extent=[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()])
    ax3.set_title('Energy Distribution')
    ax3.set_xlabel('X Position [mm]')
    ax3.set_ylabel('Y Position [mm]')
    cbar = plt.colorbar(img3, ax=ax3, orientation='horizontal')
    cbar.set_label('Energy [J/mm²]')
    
def find_threshold_positions(lineout, peaks, peak_heights, threshold_ratio=1/np.exp(2)):
    """Find positions where intensity drops to a fraction of the peak's height."""
    threshold_positions = []
    for peak, height in zip(peaks, peak_heights):
        threshold_height = height * threshold_ratio
        left_indices = np.where(lineout[:peak] < threshold_height)[0]
        right_indices = np.where(lineout[peak:] < threshold_height)[0]
        left_position = left_indices[-1] if left_indices.size > 0 else None
        right_position = peak + right_indices[0] if right_indices.size > 0 else None
        threshold_positions.append((left_position, right_position))
    return threshold_positions



zoom_region = (start_x, start_y, 500, 500)  # x start position, y start position, x length, y length




peak_intensity_threshold = 200
# Define a suitable threshold for peak intensity
analyze_and_plot_zoomed_image(summed_image, zoom_region, peak_intensity_threshold=peak_intensity_threshold)
