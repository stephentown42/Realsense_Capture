function batch_registerImages()

file_path = 'G:\Jumbo_calibration_linked';
output_path = 'G:\Jumbo_calibration_aligned';

template_file = '2016-04-29 10_31_56.jpg';
template_img = imread( fullfile( file_path, template_file));

% test_file = '2016-10-22 09_41_02.jpg';
test_files = dir( fullfile( file_path , '2017-03*.jpg'));

% For each calibration image
for i = 1 : numel(test_files)
    
    % Check if file processed already
    output_file = replace(test_files(i).name, '.jpg', '.mat');
    output_file = fullfile( output_path, output_file);
    
%     if exist( output_file, 'file')
%         continue
%     end

    % Load image and register to template
    test_img = imread( fullfile( file_path, test_files(i).name));

    reg_obj = registerImages(test_img, template_img);

    % Save results as .mat file (not sure what else to do atm)
    save( output_file, 'reg_obj')

    % Save comparison of template and warped image after registration
    montage_image = replace(test_files(i).name, '.jpg', '_montage.png');
    fusedpair = imfuse( reg_obj.RegisteredImage, test_img, 'montage');
    imwrite(fusedpair, fullfile( output_path, montage_image));
end


