""" 
Use the Google Tesseract Optical Character Recognition (OCR) to extract TDT samples within video frames 

The script is specifically developed for recovering time stamps of frames in six videos recorded between 31st Jan and 19th Feb 2018 (a point when valuable neural recordings were also being made)

Requirements:
    In addition to python and associated modules, this script requires the tesseract program to be installed on your local machine.

Version History:
    2023-02-05: Created by Stephen Town

 """
from pathlib import Path
import sys, os

from dotenv import load_dotenv
import cv2 as cv
import numpy as np
import pytesseract

sys.path.insert(0, str(Path.cwd()))
from lib.utils import query_postgres

# Path to tesseract executable on local machine (this example is for Linux)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def extract_samples_from_frames(file_path:Path, file_name:str) -> None:
    """ 
    
    Args:
        file_path: directory containing video file
        file_name: name of video file from which to extract information
    
    Output:
        Strings from each frame are written to a text file with the same name as the video
    """

    video_path = file_path / file_name
    cap = cv.VideoCapture( str(file_path / file_name))
    
    output_path = file_path / file_name.replace('.avi','.txt')
    with open(output_path, 'w') as f:
    
        while cap.isOpened():
        
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Select rectangle within which all text should fall (upto 10^9)
            n_rows = 28 # 14 for smaller videos
            multiplier = 2
            img_rgb = img_rgb[:n_rows, :100, :]

            # Use average pixel intensity across image to identify when the bounding box in the image stops
            signal = np.mean(img_rgb[0,:,:], axis=1)
            idx = np.argmax(signal < 90) - 1

            # Add further filtering if required
            if idx >= 0:
                img_rgb = img_rgb[:, :idx, :]
                img_rgb = cv.resize(img_rgb, (int(idx*multiplier), int(n_rows*multiplier)))
            
            im_str = pytesseract.image_to_string(img_rgb)
            im_str = im_str.replace('\x0c','')
            im_str = im_str.replace('\n','')

            f.write(f"{im_str}\n")
            # frame_count += 1
            print(im_str)

            # Uncomment if you want to see the image
            cv.imshow('frame', img_rgb)
            if cv.waitKey(1) == ord('q'):
                break
            
        cap.release()
        cv.destroyAllWindows()


def main():
    
    # Local directory
    file_path = Path(os.getenv('local_home')) / 'Task_Switching/videos/'
    
    # List the specific videos in the time range in which TDT samples are only available within the image
    files = query_postgres(""" 
        SELECT filename as name
        FROM task_switch.video_files
        WHERE session_dt > '2018-01-31 14:00:00.000'
            AND session_dt < '2018-01-31 14:40:00.000';
    """)

    # Extract information from video
    for file_name in files.name.to_list():
        extract_samples_from_frames(file_path, file_name) 


if __name__ == '__main__':
    main()