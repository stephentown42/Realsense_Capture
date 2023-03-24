import cv2
import matplotlib.pyplot as plt
import numpy as np


# Open video capture object
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/Town/Videos/Jumbo/2018-06-15_Track_14-51-22.avi')

# Read in frame
ret, frame = cap.read()

# Convert from rgb to grayscale
BW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Threshold for pixels above 100 (there's an argument for skipping this as blob detection probably does it anyway)
ret, tImg = cv2.threshold(BW, 100, 255, cv2.THRESH_BINARY_INV)

# Get contours from thresholded image
im2, contours, hierarchy = cv2.findContours(tImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create image mask
mask = np.zeros_like(im2)

# Initialize
count = 0

# For each contour
for contour in contours:

    # Get properties of contour
    # M = cv2.moments(contour)
    area = cv2.contourArea(contour)

    if area > 1000 and area < 2000:

        # cv2.drawContours(frame, contours, count, (255, 0, 0), 3)   # Apply contours to image
        cv2.drawContours(mask, contours, count, color=255, thickness=-1)   # Apply contours to mask and fill (thickness=-1)

    # Increment counter (after registration to allow for zero based)
    count += 1

# Get pixels of interest
pts = np.where(mask == 255)

# Rescale image from 0:255 to 0:1 (legacy scale)
frame = frame / 255

# Get intensitities for those points in the green image
intensities = frame[pts[0], pts[1], 1]

# Initialize list of center value as mean across all pixels
centerVal = [intensities.mean()]

# Run through video
while(cap.isOpened()):

    # Read next frame
    ret, frame = cap.read()

    # if you got a frame, ask what the center value was
    if ret == True:

        # Rescale image from 0:255 to 0:1 (legacy scale)
        frame = frame / 255

        # Get intensitities for those points in the green image
        intensities = frame[pts[0], pts[1], 1]

        # Initialize list of center value as mean across all pixels
        centerVal.append(intensities.mean())
    else:
        break

cap.release()
cv2.destroyAllWindows()

print('finished')

# print(centerVal)
# Plot center values
plt.plot(centerVal)
plt.xlabel('Sample')
plt.ylabel('Center Value')
plt.show()


# # # Draw frame
# cv2.imshow('color_image', mask)

# # # cv2.imshow('labelled image', im_wit_keypoint)
# cv2.waitKey(0)
