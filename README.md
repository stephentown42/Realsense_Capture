# Realsense_Capture
Functions to capture video and timestamps from Intel Realsense cameras. 

In this particular application, we are interested in using specific infra-red sensors in the camera to monitor low-light scenes. We do not currently take advantage of the depth sensing features but do leverage some of the associated on-board hardware (e.g. CPU clock) of the device.

## Get Started

We have deployed the cameras in two testing chambers (names: Jumbo and Squid), each with slightly different features. 


## Requirements
* Installation of [librealsense SDK 2.0](https://github.com/IntelRealSense/librealsense)
* Python
* A RealSense camera (see https://www.intelrealsense.com for more details)

**Optional**
* H264 codec
* TDT Open developer (for coordination with experimental data acquisition)
