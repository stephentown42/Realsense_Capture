
% Define paths
img_dir = 'G:\Jumbo_calibration_linked';
output_dir = 'G:\Jumbo_calibration_aligned';

% List files for alignment
files = dir( fullfile( img_dir, "*resized.jpg"));

% Load reference image, to which all other images are aligned
template_file = "2016-04-29 10_31_56.jpg";
FIXED = imread( fullfile( img_dir, template_file));

% For each file
for i = 1 : length(files)
        
        % Load and run main function
        disp( files(i).name)
        MOVING = imread( fullfile( img_dir, files(i).name));
        reg = register_webcam_images( MOVING, FIXED);
        
        % Save results
        output_file = replace(files(i).name, '.jpg','.mat');
        save( fullfile(output_dir, output_file), 'reg')
        
        % Save a picture of alignment quality
        fig = figure;
        imshowpair(FIXED, reg.RegisteredImage,'montage')
        saveas(fig, fullfile( output_dir, replace(files(i).name,'.jpg','_align.jpg')))
        close(fig)
end
        