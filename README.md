# Realsense_Capture
Functions to capture video and timestamps from realsense cameras and link to other data acquisition devices (TDT, Multichannel systems)

## General
Timestamp data is superimposed on frames for debugging/reconstruction, and also in tab delimited text files for later analysis.
All videos are saved using .avi containers with .h264 encoding.

## SplitScreen_IR_rec
Samples 1280x720 images from both IR cameras, cropping one and scaling the other to give two 640x320 images with high resolution/low range (i.e. covering just the center rectangle of the original image at native resolution) and low resolution/high range (i.e. covering the entire original image at half resolution). For convenience, images are concatenated and saved as one file.

## HighRes_IR_single_rec
Records a single high resolution stream from one IR camera with timestamps superimposed. Note that the output video file is twice as large as that from the splitscreen approach.

## HighRes_IR (Redundant)
Records high resolution streams from both IR camera - this function is problematic as writing to both video frames is slow, leading to dropped frames. 
