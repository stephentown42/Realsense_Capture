function compare_calibration_im_with_videoframe
%function compare_calibration_im_with_videoframe
%
% Plot the calibration image associated with each video, and a contrast 
% enahnced version of the low exposure video frame.
%
% We know that some videos will be taken after the camera has moved and
% the calibration is no longer valid, and this can be used to check
%
% Output
%   Figure saved to jpg for each session, showing calibration image and
%   video frame in montage view.
%
% To Do:
%   Remove hard-coded paths
%   Upload output files somewhere accessible
%
% Version History:
%   2020-03-13: Created by Stephen Town 
%   2022-08-25: Added video reading and changed input from directory to
%                       tracking metadata

% Load metadata
M = readtable( 'video_metadata_extended.csv');

% Load video images 
V = load_video_images('G:\UCL_Behaving', M);

% Load calibration images
C = load_calibration_images( 'G:\Jumbo_calibration_linked', unique(M.CalibImage));

% Specify save location
save_dir = 'G:\Jumbo_calibration_comparison';

% For each video image
for i = 1 : numel(V)
        
    % Convert to grayscale and adjust contrast to enhance visibility of low
    % exposure image
    vid_im = imadjust(rgb2gray(V(i).image));
    
    % Compare images
    calib_im = find_calib_image(V(i).name, M, C);
    
    % Just show noise if we couldn't find the calibration image
    if isempty(calib_im)
        calib_im = rand(size(vid_idm)) .* 255;
    end
    
    figure
    imshowpair( calib_im, vid_im, 'montage')
    title( sprintf('%s', replace(V(i).name,'_',' ')))
   
    % Save and close
    date_name = datestr(V(i).datetime, 'yyyy-mm-dd_HH-MM-SS');
    save_name = replace(V(i).name, '.avi','.jpg');            
    save_path = fullfile( save_dir, [date_name, '_', save_name]);
    
    saveas(gcf, save_path)
    close(gcf)
    
end


function im = find_calib_image(file_name, M, C);

% Look up name of calibration image assigned to this video
calib_name = M.CalibImage{ strcmp(M.file_name, file_name)};

% For each calibration image
for i = 1 : numel(C)
    
    % If this is the calibration image we're looking for, return
    if strcmp(C(i).name, calib_name)
        im = C(i).image;
        return 
    end
end

% If you get to this point, it's an error
warning('Could not fine %s', calib_name)
im = [];
% error('Could not find calibration image')



function V = load_video_images(file_path, M)
% function V = load_video_images(file_path, M)
%
% Args:
%   file_path: directory containing tanks, with blocks containing original videos 
%   M: table with ferret, block and video file names
%
% Return:
%   V: struct containing sample image from each video

V = struct();

% For each video
for i = 1 : size(M, 1)

    fprintf('%s\n', M.file_name{i})
    vid_path = fullfile( file_path, M.Ferret{i}, M.Block{i}, M.file_name{i});
    vid_obj = VideoReader( vid_path);
    
    V(i).name = M.file_name{i};
    V(i).image = read(vid_obj, 1);
    V(i).datetime = M.DateTime(i);
end


function C = load_calibration_images( file_path, file_names);
% function C = load_calibration_images(M, file_path);
%
% Args:
%   file_path: directory containing calibration images
%   file_names: cell array with image file names 
%
% Load all calibration images into a structure for repeated referencing

C = struct('name',[],'image',[]);

for i = 1 : size(file_names)
    C(i).name = file_names{i};
    C(i).image = imread( fullfile( file_path, file_names{i}));
end