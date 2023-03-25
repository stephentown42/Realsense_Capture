function vid = get_video_data( config, file_name, B, calib)
%
% This function measures LED signal from a calibration bounding box to obtain time-varying signals around the point of trial onset.
% Each trace is used to estimate the stimulus onset time within the video, so that lag in video recording can be corrected.
%
% INPUTS:
%   - config: .json file listing paths for project
%   - file_name: video file (.avi) 
%   - B: table, containing timestamps for behavioral trials (m rows)
%   - calib: struct, contains bounding box for selecting pixels
%
% RETURNS:
%   - vid: struct, with fields...
%       - data: m-by-n array containing pixel intensities for each frame within trial window, in each trial
%       - t: n-element vector of frame times
%       - FPS: video frame rate
%       - onset_t: m-element vector containing LED onset times 
%
% NOTES:
%   - for speed, the blue channel of the image is used as a proxy to measure white LED output
%
% Stephen Town - 11th April 2020

% Options
stim_duration = 0.25;   % Min duration LED must be high to class as part of stimulus
signal_threshold = 0.5; % Proportion of normalized signal above which to class the LED as ON

% Connect to video
file_path = fullfile( config.dirs.block, file_name);
obj = VideoReader( file_path);

% Define analysis windows
debug_modifier = 1; %1.03; %0.975
window_frames = floor( diff(B.analysis_window(1,:)) * obj.FrameRate * debug_modifier);
start_frame = round(B.analysis_window(:,1) .* (debug_modifier * obj.FrameRate));    
end_frame = start_frame + window_frames;

% Remove sampling of frames after end of video (which in rare cases occured
% minutes before end of TDT recording)
rm_idx = any([start_frame > obj.NumFrames, end_frame > obj.NumFrames], 2);
start_frame(rm_idx) = [];
end_frame(rm_idx) = [];

% Preassign
nTrials = size(B, 1);
vid = struct('data', zeros( nTrials, window_frames + 1),...
             't', (1:window_frames+1) ./ (obj.FrameRate * debug_modifier),...
             'FPS', obj.FrameRate * debug_modifier,...
             'onset_t', zeros( nTrials, 1));
         
vid.t = vid.t + config.tWindow(1);    % Apply offset to align zero to stim presentation
         
% Reformat calibration values into indices 
for i = 1 : size(calib, 1)   
    calib.rows{i} = calib.start_row(i) + [1 :  calib.height(i)];
    calib.cols{i} = calib.start_col(i) + [1 :  calib.width(i)];
    calib.nPixels(i) = calib.height(i) * calib.width(i);
end

% For each trial for which there's video data
for i = 1 : numel(start_frame)
    
    % Manage requests outside range of video file
    if start_frame(i) < 1
        warning('Frame range begins before index=1 - padding data')
        frames_to_pad = 1 - start_frame(i);
        start_frame(i) = 1;
    else
        frames_to_pad = 0;
    end
    
    if end_frame(i) > obj.NumFrames
        warning('Frame range exceeds end of video - padding data')
        missing_frames = obj.NumFrames - end_frame(i);
        end_frame(i) = obj.NumFrames;
    else
        missing_frames = 0;
    end
    
    % Read a chunk of frames
    frames = read( obj, [start_frame(i) end_frame(i)]);
    
    % Skip if there is no bounding box for this spout (occurs late on in
    % project when spout 12 goes out of view of camera)
    if ~any(calib.spout == B.LEDLocation(i))
        continue;
    end
    
    % Select pixels based on bounding box 
    rows = calib.rows{ calib.spout == B.LEDLocation(i)};
    cols = calib.cols{ calib.spout == B.LEDLocation(i)};
    roi =  frames( rows, cols, :, :);
    
    % Use blue as a proxy for LED
    roi = squeeze( roi(:,:,3,:));
    
    % Average across pixels     
    nPixels = calib.nPixels( calib.spout == B.LEDLocation(i));
    roi = reshape( roi, nPixels, size(frames, 4));
    roi = mean( roi, 1);
        
    % Assign to matrix, using zeros from preassigned array to pad (i.e. no
    % need to extend the roi vector when the padding is already in place)
    if frames_to_pad > 0 
        vid.data(i,frames_to_pad+1:end) = roi;
    elseif missing_frames > 0
        vid.data(i,1:numel(roi)) = roi;
    else
        vid.data(i,:) = roi;
    end
    
    % Assign to matrix
    
end

% Baseline correct and normalize
if config.normalize_frame_vals
    noise_level = mean( vid.data(:, end-9:end), 2);
    vid.data = bsxfun(@minus, vid.data, noise_level);
    vid.data = bsxfun(@rdivide, vid.data, max(vid.data,[],2));
end

% Get onset peak for each trial for which there is video data
if config.correct_for_lag
    
    min_frames = floor(stim_duration * vid.FPS);
    movCorrection = floor(min_frames/2); % Frame shift introduced by moving average

    above_threshold = vid.data > signal_threshold;
    above_threshold = movmean( above_threshold, min_frames);

    for i = 1 : numel(start_frame)    
        if any( above_threshold(i,:))
            onset_idx = find( above_threshold(i,:) == max(above_threshold(i,:)), 1);
            onset_idx = onset_idx - movCorrection;

            if onset_idx > 0
                vid.onset_t(i) = vid.t( onset_idx);
            end
        end
    end
end
