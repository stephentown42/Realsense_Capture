function get_vis_stim_in_video(config_file)
%
% Things weren't working with the correction of frame times, but the
% acquisition of signals from the video files was ok. To focus the analysis
% on the problematic issue, this function performs the signal aquisition
% and saves the uncorrected traces for visual stimuli in the video frame.
% The aim is then to perform the correction separately without needing to
% constantly resample from the video files, which is the slowest part of
% the process. Here, we assume that all trigger times from the Quality
% Controlled behavioural data are correct. All visualiation functions have
% been removed.
%
% Args:
%   config_file: json file listing:
%       - time window over which to sample LED signals
%       - paths for source data, including:
%           - behavioral data (for timestamps of visual stimuli)
%           - video files (for obtaining pixel values)
%           - save directory (for output files)
% 
% Returns:
%   csv files for each video containing the pixel values in frames around
%   the time of visual stimulus presentation. Pixel values are calculated
%   as the mean value of the blue channel in pixels within the bounding 
%   box associated with the white LED used on each trial
%
% Notes:
%   Signals on auditory trials are discarded
%
%
% See also: sync_video_to_visual_stim.m, get_video_data.m
%
% Stephen Town - 11 April 2020

% Load config details
if nargin == 0
    config_file = 'sync_config.json';
end

config = jsondecode(fileread( config_file));

% Add Options
config.check_calib = false;
config.correct_for_lag = false;
config.normalize_frame_vals = false;

% Load metadata (use csv files as matlab database interaction not set up)
video_files = readtable( fullfile( config.paths.Metadata, 'video_files.csv'), 'delimiter',',');
calibration_files = readtable( fullfile(config.paths.Metadata, 'video_calib_images_manual.csv'), 'delimiter',',');
bounding_boxes = readtable( fullfile( config.paths.Metadata, 'bounding_boxes.csv'), 'delimiter',',');

% Analyse only RV2 files
video_files = video_files(video_files.RV2 == 1, :);

% Temp debug
video_files = video_files(video_files.debug == 1, :);
% video_files = video_files(strcmp(video_files.filename ,'F1605_Snorlax_Block_J4-4_Vid0.avi'),:);
    
% For each video
for i = 1 : size(video_files, 1)
    try
        
     % Load behavioural data
     ferret = get_ferret_name(  video_files.ferret(i));
     block = sprintf('Block_%s', video_files.block{i});
     
    [~, B] = load_behavioral_data( config, ferret, block);
    
    if isempty(B)
        fprintf('Could not find QC Behavior for %s %s\n', ferret, block)
        continue
    end
    
    fprintf('Running %s %s (%d trials)\n', ferret, block, size(B, 1))
    B.analysis_window = B.StartTimeCorrected + transpose(config.tWindow(:));
        
    % Load calibration table containing bounding boxes for each spout
    calib_img = calibration_files.calib_image{strcmp(calibration_files.video_file, video_files.filename{i})};
    calib = bounding_boxes( strcmp(bounding_boxes.calib_im, calib_img), :);
     
    % Reformat calibration values into indices 
    n_calib_zones  = size(calib, 1);

    for j = 1 : n_calib_zones
        calib.rows{j} = calib.start_row(j) + [1 :  calib.height(j)];
        calib.cols{j} = calib.start_col(j) + [1 :  calib.width(j)];
        calib.nPixels(j) = calib.height(j) * calib.width(j);
    end
        
    % Get LED signal across the full video
    config.dirs.block = fullfile( config.dirs.Tanks, ferret, block);
%     signal_file = fullfile( config.dirs.LED_vals_over_video, replace(video_files.filename{i},'.avi','.dat'));
%     
%     if ~exist(signal_file, 'file')
%         LED_signal = get_LED_vals_over_video(config, video_files.filename{i}, calib);
%         dlmwrite( signal_file, LED_signal, 'delimiter', ',')
%     end
    
    % Check if this file has been processed
    [to_skip, save_path] = output_exists( config, ferret, block);
%     if to_skip, continue; end
             
    % Get LED signals observed in video around the time of trial onset  
    vid = get_video_data( config, video_files.filename{i}, B, calib);    

    % Save output    
    headers = cell( numel(vid.t), 1);
    for j = 1 : numel(vid.t)
        headers{j} = sprintf('%.3fs', vid.t(j));
    end
    
    table_out = array2table(vid.data, 'variableNames', headers);
    writetable( table_out, save_path, 'delimiter', ',')
    
    catch err        
        fprintf('%s %s:\n', ferret, block)
        warning(err.message)
        parseError(err)
        keyboard
    end
end


function fname = get_ferret_name(fnum)

switch fnum
    case 1506
        fname = "F1506_Phoenix";
    case 1517
		fname = "F1517_Mavis";
    case 1518
        fname = "F1518_Rita";
    case 1602
		fname = "F1602_Agatha";
    case 1607
		fname = "F1607_CifJif";
    case 1613
		fname = "F1613_Ariel";
    case 1605
        fname = 'F1605_Snorlax';
end


function [to_skip, save_path] = output_exists( config, ferret, block)

% Find behavioural file
load_path = fullfile( config.dirs.Behavior.QualityControl, ferret);        
behav_file = dir( fullfile( load_path, ['*_' block '_*']));

if numel(behav_file) == 0
    [to_skip, save_path] = deal(true); 
    return
end
    
% Check if the same name exists in the output directory
save_path = fullfile( config.dirs.FrameValues, ferret, behav_file(1).name);
to_skip = exist( save_path, 'file');

if to_skip > 0 
    fprintf('%s exists - skipping\n', behav_file(1).name)
end


function [A, B] = load_behavioral_data( config, ferret, block)
%
% Returns two tables of trial data recorded during behavior:
%   - A: contains all trials 
%   - B: contains only trials with visual stimuli
%
% If no behavioral file was found for that data, A & B will be empty

load_path = fullfile( config.dirs.Behavior.QualityControl, ferret);        
behav_file = dir( fullfile( load_path, ['*_' block '_*']));


if isempty(behav_file)
    [A, B] = deal([]);
    return
else    
    fprintf('Loading %s\n', behav_file(1).name)
    B = readtable( fullfile( load_path, behav_file(1).name));
end

% Duplicate to keep auditory trials 
A = B(B.Modality == 1, :);

% Remove auditory trials (0 = visual, 1 = auditory, 2 = audiovis)
B( B.Modality == 1, :) = [];
% B( B.StimMissing == 1, :) = [];

% Remove any visual stimuli at unusual locations (edge case)
% idx = any( bsxfun(@eq, B.LEDLocation, [2 10 12]), 2);
% B = B(idx, :);

