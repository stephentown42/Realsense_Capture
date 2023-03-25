function trackLEDs
% function trackLEDs
%
% Track blue and red LEDs in videos 
%
% INPUTS:
%   - config file (.json) listing directory containing videos for analysis and saving output, and drawing options
%   - video (.avi) files in specified directory
%
% OUTPUTS:
%   - .csv files for each video 
%
% Notes:
%   - Will not overwrite data already extracted, for which a results file exists
%   - Uses weighted image (e.g. blue - (red + green)) 
%   - LED centroids estimated using gaussians
%
% Optional extras
%   - output.show_image: Show tracking in a figure window during processing (slow)
%   - output.save_image: Save tracking figure as video with labelled LED positions(slow)
%
% Stephen Town - March 2020

% Load configuration file and include TDT library in path
config_file = 'tracking_config.json';
config = jsondecode(fileread( config_file));
addpath( genpath( config.tdt_lib))

% Define here for use everywhere 
% (all videos in project have same dimensions)
config.x_pix = transpose(1 : config.image_size(1));
config.y_pix = transpose(1 : config.image_size(2));

% Report precision issues if a larger video size is attempted with this
% function
if any(config.image_size > 655)
    error('Videos with larger dimensions require revision of data type (uint16) in get_centroid2')
end

% Load metadata (faster than getting it every time)
block_table = readtable( fullfile( config.save_dir.metadata, 'video_metadata.csv'));

% Run main function on each video
for i = 1 : size(block_table, 1) 
    
   main( block_table(i,:), config); 
end
        

function main( block, config)

try
    % Check if this file has been processesd    
    if outputs_exist(block, config), return; end       
        
    fprintf('%s\n', block.file_name{1})
    
    % Open video object
    obj = VideoReader( block.full_path{1});   
    
    % Define arrays for output: 3 column tables
    S = struct('blue_peak', nan(obj.NumFrames, 1),...
                'blue_x', nan(obj.NumFrames, 1),...
                'blue_y', nan(obj.NumFrames, 1),...
                'blue_xa', nan(obj.NumFrames, 1),...
                'blue_ya', nan(obj.NumFrames, 1),...
                'blue_xc', nan(obj.NumFrames, 1),...
                'blue_yc', nan(obj.NumFrames, 1),...    
                'red_x', nan(obj.NumFrames, 1),...
                'red_y', nan(obj.NumFrames, 1),...
                'red_xa', nan(obj.NumFrames, 1),...
                'red_ya', nan(obj.NumFrames, 1),...
                'red_xc', nan(obj.NumFrames, 1),...
                'red_yc', nan(obj.NumFrames, 1),...    
                'red_peak', nan(obj.NumFrames, 1));    
    
    % Set up progress report    
    h  = waitbar(0, strrep(block.file_name{1},'_',' '));
    
    % Create graphics objects if requested
    if config.output.show_image
       fig = figure;
       image_h = imshow( rand( config.image_size(2), config.image_size(1), 3));
       
       if config.output.save_image          
           out_obj = VideoWriter( fullfile( config.save_dir.avi, block.file_name{1}));
           open(out_obj);
       end
    end
    
    % For each chunk of frames    
    for frame = 1 : obj.NumFrames

        % Report progress        
        progress = frame / obj.NumFrames;
        status = sprintf('Frame %d of %d', frame, obj.NumFrames);        
        waitbar(progress, h, status)
        
        % Load video
        rgb_im = read(obj, frame);
        
        % Weight in favor of signal channel and against other channels
        blue_IM = rgb_im(:,:,3) - rgb_im(:,:,2) - rgb_im(:,:,1);
        red_IM = rgb_im(:,:,1) - rgb_im(:,:,2) - rgb_im(:,:,3);         
        
        % Find max values
        S.blue_peak(frame) = max(blue_IM(:));
        S.red_peak(frame) = max(red_IM(:));
                                   
        % get Centroid
%         [blue_LED.x(frame), blue_LED.y(frame)] = getCentroid(blue_IM, blue_LED.peak(frame));
%         [red_LED.x(frame), red_LED.y(frame)] = getCentroid(red_IM, red_LED.peak(frame)); 
                
        [blue_cx, blue_cy] = get_centroid2(blue_IM, config);
        [red_cx, red_cy] = get_centroid2(red_IM, config);
                
        S.blue_x(frame) = blue_cx(2);   % Position estimated
        S.blue_y(frame) = blue_cy(2);        
        S.blue_xa(frame) = blue_cx(1);  % Gaussian parameters for assessing fit
        S.blue_ya(frame) = blue_cy(1);  
        S.blue_xc(frame) = blue_cx(3);
        S.blue_yc(frame) = blue_cy(3);
                
        S.red_x(frame) = red_cx(2);
        S.red_y(frame) = red_cy(2);             
        S.red_xa(frame) = red_cx(1);
        S.red_ya(frame) = red_cy(1);
        S.red_xc(frame) = red_cx(3);
        S.red_yc(frame) = red_cy(3);
        
        % Show if requested
        if config.output.show_image || config.output.save_image                       
            
            blue_pos = [S.blue_x(frame) S.blue_y(frame)];
            red_pos = [S.red_x(frame) S.red_y(frame)] ;
            
            rgb_im = insertMarker(rgb_im, blue_pos, 'x','color','c');
            rgb_im = insertMarker(rgb_im, red_pos, 'x','color','y');            
        end
        
        if config.output.show_image, set(image_h, 'CData', rgb_im); end            
        if config.output.save_image, writeVideo(out_obj, rgb_im); end
    end

    % Close progress bar
    close(h)
    
    % Tidy up graphics objects (if requested)
    if config.output.show_image        
        if config.output.save_image
            close(out_obj)
        end        
       close(fig)
    end
          
    % Write data if requested
    if config.output.save_data
       
        T = struct2table(S);
        file_name = replace(block.file_name{1},'.avi','.csv');
        file_path = fullfile( config.save_dir.csv, file_name);
        writetable(T, file_path, 'delimiter', ',')
    end
    
    
%     %% Supervision of thresholding (and intervention if necessary)
%     
%     % Plot threshold for green LED
%     fT = figure; hold on
% 
% %     blueThresh = queryThreshold(blue, 'b');
% %     redThresh  = queryThreshold(red,  'r');
%      
%     blueThresh = fixedThreshold(blue_LED, 'b', 50);
%     redThresh = fixedThreshold(red_LED, 'r', 50);
%         
%     % Show example frames
%     [rFig, bFig] = showExampleFrames(red_LED, blue_LED, redThresh, blueThresh, obj);
%     
%     % Save figures
%     saveas( fT,   fullfile( dataDir, saveFig))
%     saveas( bFig, fullfile( dataDir, strrep(saveFig,'.fig','_BlueFlt.fig')))
%     saveas( rFig, fullfile( dataDir, strrep(saveFig,'.fig','_RedFlt.fig')))
%     
%     % Close figures and tidy up
%     close([fT rFig bFig])
%     clear fT rFig rFig excludedFrames includedFrames
%     
%     % Apply threshold    
%     blue_LED(blue_LED(:,1) < blueThresh, :) = NaN;
%     red_LED(red_LED(:,1) < redThresh, :) = NaN;
%     
%     % Attempt to recover missing frames
%     blue_LED(:,2) = clean_trajectory_STmod( blue_LED(:,2)); 
%     blue_LED(:,3) = clean_trajectory_STmod( blue_LED(:,3));
%     red_LED(:,2)  = clean_trajectory_STmod( red_LED(:,2));
%     red_LED(:,3)  = clean_trajectory_STmod( red_LED(:,3));
%         
%     % Automatic exclusion of large jumps
%     blue_LED = removeJumps(blue_LED, 20);
%     red_LED  = removeJumps(red_LED, 20);
                       
    
catch err
%     close(h)
%     close(writerObj);
    
    err
    keyboard
end


function should_skip = outputs_exist(block, config)
% 
% Returns true if either the figure or data file associated with this block
% already exists in the save directory
          
should_skip = false;

if config.output.save_data        % Behavioural data 
    should_skip = report_existing_file( block, config, 'csv');    
end

if config.output.save_image        % Image figure   
    should_skip = report_existing_file( block, config, 'avi');
end


function should_skip = report_existing_file(block, config, file_type)

file_path = get_output_path( block, config, file_type);
should_skip = exist( file_path, 'file');

if should_skip
    fprintf('%s %s - already processed\n', block.Ferret{1}, block.Block{1})
end


function save_path = get_output_path( block, config, fileType)
   
file_name = replace(block.file_name{1},'.avi', ['.' fileType]);

% Ensure save path is available
eval( sprintf('save_path = config.save_dir.%s;', fileType))
        
if ~isfolder( save_path)
    mkdir( save_path)
end

save_path = fullfile( save_path, file_name);


function [x, y] = getCentroid(IM, iMax)

[x, y] = deal(nan);

% Create binary image
isMax = IM == iMax;   

if any(isMax(:))
    
    % Get blob properties
    RP = regionprops(isMax,'Centroid','Area');
    
    % Choose the blob closest to the center of the arena
%     if numel( RP) > 1,
%         keyboard
%     end
    
%     % Choose the largest blob
    blobArea = cat(1,RP.Area);
    blobIdx  = find(blobArea == max(blobArea));
        
    % Throw away the losers
    centroid = cat(1,RP.Centroid);
    
    % If there's a clear winner (give up otherwise)
    if sum(blobIdx) == 1        
        
        % Assign the winner
        x = centroid(blobIdx, 1);
        y = centroid(blobIdx, 2);
        
    else
        x = mean(centroid(:,1));
        y = mean(centroid(:,2));
    end
end


function [cx, cy] = get_centroid2(im, config)

% Fit a single gaussian (y = a*exp(-((x-b)/c)^2)
try
    fx = fit( config.x_pix, transpose(mean(im, 1)),'gauss1');
    cx = coeffvalues(fx);
catch
    cx = [0 0 0];
end

try
    fy = fit( config.y_pix, mean(im,2),'gauss1');
    cy = coeffvalues(fy);
catch 
    cy = [0 0 0];
end

% Return b, the coffecient for the centre of the curve
% (Note that here we're saving as 100 times the value as an unsigned 16 bit
% integer. This reduces space in the saved file while allowing us a
% sub-pixel resolution without saving a ridiculous number of decimal
% places)
% x = uint16(cx(2) * 100);  
% y = uint16(cy(2) * 100);

% FUTURE: note that we could do some smarter stuff here with the other
% coefficients to detect poor fitting 

% Assess quality of fit (higher values are better as we expect nice tight gaussians)
% quality_x = cx(1) / cx(3);
% quality_y = cy(1) / cy(3);
% 
% if quality_x < config.quality_threshold 
%     x = uint16(0);
% end
% 
% if quality_y < config.quality_threshold
%     y = uint16(0);   
% end


function block_table = get_blocks_to_analyze( config)
%
% Input:
%   - Config: Struct containing relevant paths from config file
%
% Output:
%   - block_table: Table containing list of all videos

% For each ferret
config.n_ferrets = numel( config.ferrets);

[tank, block, file_name, full_path] = deal([]);
[frame_rate, nFrames, duration, full_path] = deal([]);
count = 0;

for i = 1 : config.n_ferrets
    
    % List blocks    
    tank_path = fullfile( config.tank_dir, config.ferrets{i});
    blocks = dir( fullfile( tank_path, 'Block_J*'));
    
    % For each block
    for j = 1 : numel(blocks)
        
        % Look for video file
        block_path = fullfile( tank_path, blocks(j).name);                
        vid_files = dir(fullfile(block_path, '*Vid0.avi'));

        % Append to list
        for k = 1 : numel(vid_files)
                    
            count = count + 1;
            tank{count,1} = tank_path;
            block{count,1} = blocks(j).name;
            file_name{count,1} = vid_files(k).name;
            full_path{count, 1} = fullfile( block_path, vid_files(k).name);
                        
            obj = VideoReader( full_path{count});
        end
    end
end

block_table = table(tank, block, file_name, full_path);



function thresh = queryThreshold(t, color)

nFrames = size(t,1);
bp = plot( [1 nFrames], [0 0], color,'LineStyle','--'); 
plot( single(t(:,1)),color)

% Query user
happy = 'No';

while strcmp('No',happy)
    delete(bp)
    [~, thresh] = ginput(1);
    
    bp    = plot( [1 nFrames], [thresh thresh], color,'LineStyle','--');
    happy = input('Are you happy with the new threshold (Yes/No)?','s');
end
       

function thresh = fixedThreshold(t, color, thresh)

nFrames = size(t,1);

plot( single(t(:,1)),color)
    
bp = plot( [1 nFrames], [thresh thresh], color,'LineStyle','--');
       



function [rFig, bFig] = showExampleFrames(red, blue, rT, bT, obj)

% Show example frames in which we would include / exclude data
nExamples = 8;

excludedFrames.red = find(red(:,1) < rT);
includedFrames.red = find(red(:,1) > rT);
excludedFrames.blue = find(blue(:,1) < bT);
includedFrames.blue = find(blue(:,1) > bT);

excludedFrames.red = randomSelection(excludedFrames.red, nExamples);
includedFrames.red = randomSelection(includedFrames.red, nExamples);
excludedFrames.blue = randomSelection(excludedFrames.blue, nExamples);
includedFrames.blue = randomSelection(includedFrames.blue, nExamples);


% Show frames in which the red threshoold isn't met
rFig = figure('color',[1 1/2 1/2],'position',get(0,'ScreenSize'));
sp   = dealSubplots(4,4);

for i = 1 : nExamples
    
    if numel(excludedFrames.red) >= i
        video = read(obj, excludedFrames.red(i));
        image(video, 'parent', sp(i))
    end
    
    if numel(includedFrames.red) >= i
        video = read(obj, includedFrames.red(i));
        image(video, 'parent', sp(i+nExamples))
    end
end

set(sp,'xcolor','none','ycolor','none')
title(sp(1),'Excluded')
title(sp(1+nExamples),'Included')
axis(sp,'tight')


% Show frames in which the blue threshold isn't met
bFig = figure('color',[1/2 1/2 1],'position',get(0,'ScreenSize'));
sp   = dealSubplots(4,4);

for i = 1 : nExamples
    
    if numel(excludedFrames.blue) >= i
        video = read(obj, excludedFrames.blue(i));
        image(video, 'parent', sp(i))
    end
    
    if numel(includedFrames.blue) >= i
        video = read(obj, includedFrames.blue(i));
        image(video, 'parent', sp(i+nExamples))
    end
end

set(sp,'xcolor','none','ycolor','none')
title(sp(1),'Excluded')
title(sp(1+nExamples),'Included')
axis(sp,'tight')


