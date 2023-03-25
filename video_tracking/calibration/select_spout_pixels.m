function select_spout_pixels(file_path)
%
% Manual identification of regions around response spouts that will light up with presentation of visual stimuli. 
% Used in synchronization process to measure lag across recording using light levels within bounding boxes.
%
% INPUT:
%   - file_path: path to calibration image from which to manually identify bounding box
%
% OUTPUT:
%   - csv file with info on bounding boxes for each spout in image
%
% Stephen Town - March 2020


% Run in batch mode through all jpg files in current directory
files = dir(fullfile(file_path, '*.jpg'));

for i = 1 : numel(files)
    
    main( file_path, files(i).name)
end


function main(file_path, file_name)

    % Skip if already done
    save_name = strrep(file_name, '.jpg', '.csv');
    save_path = fullfile( file_path, save_name);

    if exist(save_name, 'file'), return; end

    % Read image
    im = imread( fullfile( file_path, file_name));

    % Create output table
    T = table([2; 10; 12], 'VariableNames', {'Spout'});
    position = nan(3,4);

    % Draw figure and request input (double-click bounding box to complete each
    % choice)
    fig = figure;
    imshow(im)

    for i = 1 : 3
        title(sprintf('Select spout %d', T.Spout(i)))
        h = imrect;
        position(i,:) = wait(h);
    end

    % Convert to concise data format and save
    position = uint16(position);
    T.start_col = position(:,1);
    T.start_row = position(:,2);
    T.width = position(:,3);
    T.height = position(:,4);

    % Write output
    writetable(T, save_name, 'delimiter', ',')

    close(fig)
% % Show selected region
% position = round(position);
% 
% rows = [0:position(4)] + position(2);
% cols = [0:position(3)] + position(1);
% 
% im(rows, cols, 1) = uint16(255);
