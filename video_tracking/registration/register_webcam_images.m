function [MOVINGREG] = register_webcam_images(MOVING,FIXED)
%registerImages  Register grayscale images using auto-generated code from Registration Estimator app.
%  [MOVINGREG] = registerImages(MOVING,FIXED) Register grayscale images
%  MOVING and FIXED using auto-generated code from the Registration
%  Estimator app. The values for all registration parameters were set
%  interactively in the app and result in the registered image stored in the
%  structure array MOVINGREG.
%
% Note: This script is applied to images captured by the webcam rather than
% RV2 camera, and thus requires a more sophisticated transofrmation than
% the affine transform used elsewhere.
%
%
% Auto-generated by registrationEstimator app on 09-Feb-2023
%-----------------------------------------------------------


% Convert RGB images to grayscale
FIXED = rgb2gray(FIXED);
MOVING = rgb2gray(MOVING);

% Default spatial referencing objects
fixedRefObj = imref2d(size(FIXED));
movingRefObj = imref2d(size(MOVING));

% Intensity-based registration
[optimizer, metric] = imregconfig('multimodal');
metric.NumberOfSpatialSamples = 500;
metric.NumberOfHistogramBins = 50;
metric.UseAllPixels = true;
optimizer.GrowthFactor = 1.050000;
optimizer.Epsilon = 1.50000e-06;
optimizer.InitialRadius = 6.25000e-03;
optimizer.MaximumIterations = 1000;

% Align centers
[xFixed,yFixed] = meshgrid(1:size(FIXED,2),1:size(FIXED,1));
[xMoving,yMoving] = meshgrid(1:size(MOVING,2),1:size(MOVING,1));
sumFixedIntensity = sum(FIXED(:));
sumMovingIntensity = sum(MOVING(:));
fixedXCOM = (fixedRefObj.PixelExtentInWorldX .* (sum(xFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.XWorldLimits(1);
fixedYCOM = (fixedRefObj.PixelExtentInWorldY .* (sum(yFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.YWorldLimits(1);
movingXCOM = (movingRefObj.PixelExtentInWorldX .* (sum(xMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.XWorldLimits(1);
movingYCOM = (movingRefObj.PixelExtentInWorldY .* (sum(yMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.YWorldLimits(1);
translationX = fixedXCOM - movingXCOM;
translationY = fixedYCOM - movingYCOM;

% Coarse alignment
initTform = affine2d();
initTform.T(3,1:2) = [translationX, translationY];

% Normalize images
movingInit = mat2gray(MOVING);
fixedInit = mat2gray(FIXED);

% Apply transformation
tform = imregtform(movingInit,movingRefObj,fixedInit,fixedRefObj,'similarity',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);
MOVINGREG.Transformation = tform;
MOVINGREG.RegisteredImage = imwarp(MOVING, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);

% Store spatial referencing object
MOVINGREG.SpatialRefObj = fixedRefObj;

end

