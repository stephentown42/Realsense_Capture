""" 
Register images into a common reference frame 




The motion model should approximate the variance in camera position with the fewest degrees of freedom needed. (I.e. don't use affine or homographic models unless you have a good reason)


Sources:

https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

It looks like the original method for finding the homography matrix has been updated
https://opencv.org/evaluating-opencvs-new-ransacs/

https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/

"""

from pathlib import Path
import os, sys
from typing import List

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path.cwd()))
from lib import utils



class keypoint_aligner():
    """ Computes homographic transform between images based on shared features (keypoints) 
    
    Parameters:
        - maxFeatures: number of keypoints to identify
        - keepPercent: top percentage of keypoints to use in alignment after ranking by quality

    Notes:
        Time to compute homography matrix increases as a function of kept features
    """

    def __init__(self, template_img:np.array, maxFeatures:int=500):
        """
        Args:
            template_img: fixed image to which all test images are aligned
            maxFeatures: maximum number of keypoints to consider
        """

        # Find image size
        self.template_img = template_img
        sz = template_img.shape
        self.img_size = (sz[1],sz[0])

        # Detect keypoints in template image
        self.orb = cv2.ORB_create(maxFeatures)
        (self.template_kpts, self.template_descs) = self.orb.detectAndCompute(self.template_img, None)
        

    def match_images(self, test_image:np.array, keepPercent:float=2):
        """
        Find matching keypoints between template and a test image

        Args:
            test_image: test image
            keepPercent: percentage of keypoint matches to keep, allows elimination of noisy keypoints
        """

        # use ORB to detect keypoints and extract (binary) local invariant features
        (kpsA, descsA) = self.orb.detectAndCompute(test_image, None)

        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, self.template_descs, None)

        # sort the matches by their distance (the smaller the distance,
        # the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)

        # keep only the top matches
        keep = int(len(matches) * (keepPercent/100))
        matches = matches[:keep]

        # Return matches and keypoints
        return matches, kpsA


    def show_matches(self, test_img, test_kpts, matches):
        """ Visualize matches for debugging poor image alignment """

        matchedVis = cv2.drawMatches(test_img, test_kpts, self.template_img, self.template_kpts, matches, None)
        # matchedVis = imutils.resize(matchedVis, width=1000)

        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)


    def get_homography_matrix(self, matches, test_kpts):
        """ Compute homography matrix for transforming test image to template image 
        
        Args:
            matches: keypoint matches between images
            test_kpts: keypoints in the test image

        Returns:
            H: Homography matrix
        """

        # Allocate memory for the keypoints (x, y)-coordinates from the top matches
        # we'll use these coordinates to compute our homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")

        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images map to each other
            ptsA[i] = test_kpts[m.queryIdx].pt
            ptsB[i] = self.template_kpts[m.trainIdx].pt

        # compute the homography matrix between the two sets of matched points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.USAC_MAGSAC)
        
        return H

	
    def apply_correction(self, test_image, homography_mat):
        """ use the homography matrix to align the images """
        return cv2.warpPerspective(test_image, homography_mat, self.img_size)


class ecc_aligner():
    """ 
    Compute intensity based alignment using Enhanced Correlation Coefficient (ECC) Maximization
    """

    def __init__(self, template_img, warp_mode=cv2.MOTION_EUCLIDEAN, n_iterations:int=5000, termination_eps:float=1e-10):
        """ 
        Args:
            template_img: fixed image to which all test images are aligned
            warp_mode: motion model best suited to the problem (usually Euclidean)
            n_iterations: number of iterations for finding transform
            termination_eps: termination criteria for search
        """

        # Define the motion model
        self.warp_mode = warp_mode

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Specify the number of iterations.
        self.number_of_iterations = n_iterations;
        
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        self.termination_eps = termination_eps;
        
        # Define termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)
        
        # Find image size
        self.template_img = template_img
        sz = template_img.shape
        self.img_size = (sz[1],sz[0])


    def estimate(self, test_img):
        """ Run the ECC algorithm. The results are stored in warp_matrix. """

        (self.cc, self.warp_matrix) = cv2.findTransformECC(
            self.template_img, test_img, self.warp_matrix, self.warp_mode, self.criteria)


    def apply_correction(self, test_img):
        """ Use warpAffine for Translation, Euclidean and Affine """
        return cv2.warpAffine(test_img, self.warp_matrix, self.img_size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP); 


# Helper functions
def get_file_names_from_db() -> List[str]:
    """ Request a list of the names of all calibration image files in project from database """

    img_files = utils.query_postgres('SELECT id FROM task_switch.calibration_images;')
    return img_files['id'].to_list()


def load_greyscale_image(file_path:Path, file_name:str) -> np.array:
    """ Read image from file in grayscale """
    return cv2.imread( str(file_path / file_name), cv2.IMREAD_GRAYSCALE)
    
    
def combine_images(template_img:np.array, test_img:np.array, aligned_img:np.array, resize_percent:float=50.0) -> np.array:
    """ Create a 2x2 composite image that juxtaposes a template image with test image in the first row, and the same template with the aligned image after registration.
    
    Args:
        template_img: grayscale image to which all others are aligned
        test_img: grayscale image before registration
        aligned_img: grayscale image after registration
        resize_percent: scale factor that allows the resulting image to be shown at convient size
    
    Returns:
        Grayscale image combining input images
    """
    
    combined = np.concatenate((
        np.concatenate((template_img, test_img), axis=1),
        np.concatenate((template_img, aligned_img), axis=1)
        ),axis = 0)

    return resize_image(combined, percent=resize_percent)


def resize_image(img:np.array, percent:float=50.0):
    """ Resize an image by a given percentage """

    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim)


def show_image(img):
    """ Show image in pop-up box
    
    Notes: Press "q" to exit pop-up
    """

    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_arbritrary_image(im_size=(480,640), test_rect=(100, 100, 1, 10)) -> np.array:

    # Create base
    img = np.zeros(im_size, dtype=np.uint8) + 20

    # Add white rectangle to image
    img[test_rect[0]:test_rect[0]+test_rect[2], test_rect[1]:test_rect[1]+test_rect[3]] = 255

    return img



def main():

    # Define paths
    load_dotenv()
    data_path = Path(os.getenv("local_home")) / 'Task_Switching/head_tracking'
    input_path = data_path / 'calibration_images'
    output_path = data_path / 'calibration_alignment_intensity'
    
    # Reference image against which all others are aligned
    # (The reference image should be selected as close to the middle of the variation in images as possible)
    template_file = "2016-04-29 10_31_56.jpg"
    template_img = load_greyscale_image(input_path, template_file)
    # template_img = cv2.GaussianBlur(template_img, (15,15), cv2.BORDER_DEFAULT)

    # Create aligner object for template
    ecc = ecc_aligner(template_img=template_img)
    kpt = keypoint_aligner(template_img=template_img)

    # Test images for alignment
    # file_names = get_file_names_from_db()
    file_names = ['2017-03-27 17_33_33.jpg']

    # Try to align each test image
    for test_file in file_names:
        
        # Skip if already done
        # output_file = output_path / test_file.replace('.jpg','_warp.txt')
        # if output_file.exists():
        #     continue

        # Otherwise load and estimate alignment
        test_img = load_greyscale_image(input_path, test_file)
        # test_img = cv2.GaussianBlur(test_img, (5,5), cv2.BORDER_DEFAULT)
        
        # Use intensity based method
        ecc.estimate(test_img)
        print(f"{test_file}: correlation = {ecc.cc}")
        
        # Use keypoint based method
        matches, test_kpts = kpt.match_images(test_img)
        kpt.show_matches(test_img, test_kpts, matches)

        kpt_homography = kpt.get_homography_matrix(matches, test_kpts)
        test_aligned = kpt.apply_correction(test_img, kpt_homography)
                        
        # Save warp matrix for future
        # np.savetxt(output_file, ecc.warp_matrix)

        # Apply correction and save
        test_aligned = ecc.apply_correction(test_img)
        combined = combine_images(template_img, test_img, test_aligned)
        show_image(combined)

        # image_output = output_path / test_file.replace('.jpg','combined.jpg')
        # cv2.imwrite(str(image_output), combined)



if __name__ == '__main__':
    main()
