""" 
Module to identify contours surrounding regions of interest defined by area (e.g. white bar on dark background)

Version History:
    2018-03-20: Created by Stephen Town
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_contours_for_sampleframe(cap, pixel_threshold:float=100):
    """ Get contours around white objects on dark backgrounds from color image 
    
    Args:
        cap: capture object, e.g. video file or realsense stream
        pixel_threshold: value to threshold grayscale image, used for noise reduction
    
    Returns:
        contours: array of numpy arrays with details of contours around blobs 
    """
    
    ret, frame = cap.read()

    if ret:
        BW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, tImg = cv2.threshold(BW, pixel_threshold, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(tImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    else:
        return IOError('Could not read frame')


def get_points_within_area_limited_contour(im_size, contours, area_lim):
    """  Get the rows and column indices (points) of pixels within contour of approapriate area
    
    Args:
        im_size: height (n rows) and width (n cols) of image
        contours: tuple containing contours detected within image
        area_lim: limits used to isolate the contour around the region of interest

    Returns:
        pts: tuple containing rows and columns denoting pixels of interest
    """

    mask = np.zeros(im_size)
    gate_open = True                # use a logic gate to check that only one area is within limits

    # Check if each contour falls within the allowable range of areas 
    for i, contour in enumerate(contours):

        if area_lim[0] < cv2.contourArea(contour) < area_lim[1]:
            
            if gate_open:
                cv2.drawContours(mask, contours, i, color=255, thickness=-1)   # fill in the contour and close gate (there should only be one object within )
                gate_open = False
            else:
                raise ValueError('Multiple areas within limits found')
        
        
    # If the gate has not been closed, we must have missed the right contour (or it's not in the image)
    if gate_open:
        return ValueError('No area within limits found')
    else:
        return np.where(mask == 255)


def extract_mean_intensity_from_video(cap:cv2.VideoCapture, pts, color_chan:int=1):
    """ Get the mean intensity of signal within pixels of interest for every frame in video
    
    Args:
        cap: capture object, e.g. video file or realsense stream
        pts: tuple containing rows and columns denoting pixels of interest
        color_chan: color channel from which to sample pixels (0=blue, 1=green, 2=red)

    Returns

        pixel_means: list of floating point values for mean intensity within region of interest for every frame
    """

    pixel_means = []
    
    while(cap.isOpened()):

        ret, frame = cap.read()

        # if you got a frame, ask what the center value was
        if ret == True:

            # Rescale image from 0:255 to 0:1 (legacy scale)
            frame = frame / 255

            # Get intensitities for those points in the green image
            intensities = frame[pts[0], pts[1], color_chan]

            # Initialize list of center value as mean across all pixels
            pixel_means.append(intensities.mean())
        else:
            break

    cap.release()
    return pixel_means


def main():

    import os
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()

    video_path = Path(os.getenv("data_path"))
    video_file = '2018-06-15_Track_14-51-22.avi'
    
    cap = cv2.VideoCapture(str(video_path/video_file))

    contours = get_contours_for_sampleframe(cap, pixel_threshold=100)

    area_lim = (1000, 2000)
    im_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    pts = get_points_within_area_limited_contour(im_size, contours, area_lim)

    mean_intensity = extract_mean_intensity_from_video(cap, pts, color_chan=1)

    # Write to text file
    output_path = Path.cwd() / 'demos/contour_detection'
    output_file = video_file.replace('.avi','_pixelintensities.dat')
    
    np.savetxt( output_path/output_file, np.array(mean_intensity))

    # Plot center values
    plt.plot(mean_intensity)
    plt.xlabel('Sample')
    plt.ylabel('Center Value')
    plt.show()



if __name__ == '__main__':
    main()