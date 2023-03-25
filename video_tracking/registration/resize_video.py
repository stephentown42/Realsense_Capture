""" 

Some later videos in the project were recorded using a different camera with a different aspect ratio. Rather than determine how to align the calibration and tracking results for different conditions, it seems easier to just resize the exceptional videos to the same size as the majority of other videos recorded.


Version History:
    2023-02-06: Created by Stephen Town

 """
from pathlib import Path
import sys, os

# from dotenv import load_dotenv
import cv2 
import numpy as np

# sys.path.insert(0, str(Path.cwd()))
# from lib.utils import query_postgres



def resize_video(file_path:Path, file_name:str, required_size:tuple) -> None:
    """ 
    
    Args:
        file_path: directory containing video file
        file_name: name of video file from which to extract information
        required_size: target size for output video
    
    Output:
        Strings from each frame are written to a text file with the same name as the video
    """

    input_path = file_path / file_name
    output_path = file_path / file_name.replace('.avi','_resized.avi')

    if output_path.exists():
        print('{file_name} already processed')
        return

    assert input_path != output_path
    
    cap = cv2.VideoCapture( str(input_path))
    start_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    start_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Starting size = ({start_width}, {start_height})")
    print(f"Required size = {required_size}")

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MPEG"), 
        cap.get(cv2.CAP_PROP_FPS),
        required_size)

        
    while cap.isOpened():
    
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        new_frame = cv2.resize(frame, required_size)        
        out.write(new_frame)

        # # Uncomment if you want to see the image
        # cv2.imshow('frame', new_frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        
    cap.release()
    out.release()
    # cv2.destroyAllWindows()


def get_sample_images(file_path:Path, file_name:str) -> None:
    """ 
    
    Args:
        file_path: directory containing video file
        file_name: name of resized video file from which to extract information
    
    Output:
        Sample images are 
    """

    
    # Create video reader object
    input_path = file_path / file_name
    cap = cv2.VideoCapture( str(input_path))

    # Get 10 sample images for each video    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    

    for frame_number in np.linspace(0, n_frames-1, 10):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        output_file = file_name.replace('.avi', f"_{frame_number:.0f}.jpg")
        output_path = file_path / 'sample_images' / output_file

        cv2.imwrite(str(output_path), frame)


def main():
    
    # Settings 
    required_size = (640, 480) 

    # Local directory containing source videos (too large to read)
    file_path = Path('C:\Users\Squid\Videos\Frontiers_HE')
    # file_path = Path(os.getenv('local_home')) / 'Task_Switching/videos/'
    
<<<<<<< Updated upstream
    # List the specific videos in the time range in which TDT samples are only available within the image
    files = query_postgres(""" 
        SELECT filename as name
        FROM task_switch.video_files
        WHERE session_dt > '2018-01-29 08:40:00.000'
            AND session_dt < '3018-02-19 11:40:00.000';
    """)
=======
    # # List the specific videos in the time range in which TDT samples are only available within the image
    # files = query_postgres(""" 
    #     SELECT filename as name
    #     FROM task_switch.video_files
    #     WHERE session_dt > '2018-01-31 04:40:00.000'
    #         AND session_dt < '2018-01-31 08:40:00.000';
    # """)
>>>>>>> Stashed changes

    # Extract information from video
    # for file_name in files.name.to_list():
    for file_name in file_path.glob('*Track*.avi')
        resize_video(file_path, file_name, required_size) 
        
        new_name = file_name.replace('.avi','_resized.jpg')
        print(f"{new_name},{file_name}")

        get_sample_images(file_path, new_name)


if __name__ == '__main__':
    main()