""" 
Plotting module for LED tracking

Used for visualizing trajectories of LEDs on individidual sessions / across 
trials, as well as deciding on hyper-parameters for data processing.


Version History:
    2022-10-01: Created by Stephen Town
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import os, sys

import cv2
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import loading as vload
import transform as vtran


####
# Video functions
def draw_APC_activity(frame, norm_rates) -> None:
    """ Add rectangles to video frame to show activity of each electrode for array of 32 channels """

    # white box around outside of array
    cv2.rectangle(frame, (565, 345), (630, 470), (255,255,255)) 

    for chan_row in range(8):
        for chan_col in range(4):
            
            r_start = (570+(chan_col * 15), 350+(chan_row*15))
            r_end = (r_start[0]+10, r_start[1]+10)
            
            chan = (chan_row*4)+chan_col
            r_color = int(norm_rates[chan])

            if chan_row >= 4:
                r_color = (0, r_color, r_color)
            else:
                r_color = (0, r_color, 0)

            cv2.rectangle(frame, r_start, r_end, r_color, thickness=-1)


def draw_PFC_activity(frame, norm_rates):
    """ Add rectangles to video frame to show activity of each electrode for array of 16 channels """

    cv2.rectangle(frame, (495, 405), (560, 470), (255,255,255)) # white box

    for chan_row in range(4):
        for chan_col in range(4):
            
            r_start = (500+(chan_col * 15), 410+(chan_row*15))
            r_end = (r_start[0]+10, r_start[1]+10)
            
            chan = (chan_row*4)+chan_col
            r_color = int(norm_rates[chan])

            cv2.rectangle(frame, r_start, r_end, (r_color, 0, r_color), thickness=-1)


@dataclass()
class video_figure():

    video_file: Path
    font_face: int=cv2.FONT_HERSHEY_DUPLEX
    font_scale: float = 0.5
    start_time: float=0.0
    duration: float=np.inf
    save: bool=False
    fps: float = 30

    def __post_init__(self):
        """ Open file and get properties """

        # Make sure parameters make sense (negative time doesn't exist)
        assert self.start_time >= 0.0
        assert self.duration >= 0.0

        # Get input video properties
        self.cap = cv2.VideoCapture(self.video_file)        
        self.width = int( self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        self.height = int( self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        # Formatting for text annotations
        self.font_style = dict(                 
            fontFace = self.font_face, 
            fontScale = self.font_scale,
            color = (255, 255, 255),
            thickness = 1
        )

        if self.save:
            self.create_save_file()

    ################################################################################
    # Setup functions (the video file isn't yet opened while we do this)    
    def create_save_file(self):
        """ Create object for writing frames """

        save_file = Path(self.video_file)
        save_file = save_file.parent / (save_file.stem + '_annotated.mp4')

        self.vout = cv2.VideoWriter(
            filename = str(save_file), 
            fourcc = cv2.VideoWriter_fourcc(*"XVID"), 
            fps = self.fps, 
            frameSize = (self.width, self.height))

    ####################################################################
    # Add data for annotations
    def add_LEDs(self, LEDs):
        """ Add LEDs for times of interest"""
    
        # Filter for times of interest
        self.LEDs = LEDs[
            (LEDs['time'] >= self.start_time) &
            (LEDs['time'] < (self.start_time + self.duration))
        ].set_index(keys='frame')

        self.n_frames = self.LEDs.shape[0]


    def add_Sensors(self, sensors):
        """ Add Sensors for times of interest """
    
        self.sensors = (
            sensors
            .drop(sensors[sensors['time'] < self.start_time].index)
            .drop(sensors[sensors['time'] > self.start_time+self.duration].index)
            .set_index(keys='frame')
        )

        self.sensor_ids = [x for x in sensors.columns if 'sens' in x]

    ####################################################################
    # Run functions
    def run(self):

        current_frame_idx = self.LEDs.index.min()

        # Run the clip
        while(self.cap.isOpened()):

            # Move to read time
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            
            # Read frame and move index forward    
            ret, frame = self.cap.read()
            current_frame_idx += 1
            current_time = self.LEDs.loc[current_frame_idx,'time']

            # Annotate frame
            frame = self.add_timing_text(frame, current_time, current_frame_idx)

            frame = self.draw_LED_info(frame, current_frame_idx)
            frame = self.draw_Sensors(frame, current_frame_idx)

            cv2.imshow('Frame', frame)

            if self.save:
                self.vout.write(frame)

            # Stop if at end of analysis window
            if current_frame_idx > self.LEDs.index.max():
                break

            # Stop if user requests
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        """ Release windows and close files """
        self.cap.release()
        cv2.destroyAllWindows()

        if self.save:
            self.vout.release()


    ###########################################################################
    # Annotations

    def add_timing_text(self, frame, current_time:float, current_frame_idx:int):
        """ Add time and frame to frame """
        
        cv2.putText(frame, f"Time: {(current_time):.1f}s", org=(20, 460), **self.font_style)
        cv2.putText(frame, f"Frame: {current_frame_idx:.0f}", org=(20, 430), **self.font_style)

        return frame


    def draw_LED_info(self, frame, current_frame_idx:int):
        """ Plot information that can be drawn from LED data:
            - Head position
            - Head speed
        """

        if hasattr(self, 'LEDs'):

            frame_info = self.LEDs.loc[current_frame_idx]

            # Head position                
            if not np.isnan(frame_info['headx']) and not np.isnan(frame_info['heady']):
                
                cv2.drawMarker(
                    frame, 
                    (int(frame_info['headx']), int(frame_info['heady'])),
                    color=(150,255,150),
                    markerType=cv2.MARKER_CROSS, 
                    thickness=1
                )

            # Speed Visualization
            # Always draw text speed bar
            font_style = self.font_style.copy()
            font_style['color'] = (255,0,255)   # magenta
            cv2.putText( frame, "Speed", org=(300, 465), **font_style)

            # Optional magenta scaled bar on pale violet background for speed (or filled grey is nan)
            if not np.isnan(frame_info['speed']):

                cv2.line(frame, pt1=(360, 460), pt2=(460, 460), color=(255,200,255), thickness=3)
            
                norm_speed = 360 + np.round(frame_info['speed']/5)
                cv2.line(
                    frame, 
                    pt1=(360, 460), 
                    pt2=(int(norm_speed), 460), 
                    color=(255,0,255), 
                    thickness=2)
            else:
                cv2.line(frame, pt1=(360, 460), pt2=(460, 460), color=(180,180,180), thickness=3)

        return frame


    def draw_Sensors(self, frame, current_frame_idx:int):
        """ Show sensors as white bars that increase when a sensor is thought to be active """

        if hasattr(self, 'sensors'):

            sens_info = self.sensors.filter(items=[current_frame_idx], axis=0).to_dict('records')[0]
            
            cv2.putText(frame, "Sensor", org=(500, 465), **self.font_style)

            for chan, sensor_id in enumerate(self.sensor_ids):
                xpos = 580 + chan*10
                ypos = 460 - int(sens_info[sensor_id]*20)
                cv2.line(frame, pt1=(xpos, 465), pt2=(xpos, ypos), color=(255,255,255), thickness=2)


        return frame


def play_video(video_file:str, 
    trail : int=1,
    duration : Optional[float] = None,
    LEDs : Optional[pd.DataFrame] = None,
    n_frames : Optional[int] = None,
    norm_spike_rates : Optional[np.array] = None,
    save : Optional[bool] = False,
    sensors : Optional[np.array] = None,
    start_frame : Optional[int] = None,
    start_time : Optional[float] = None, 
    trials : Optional[pd.DataFrame] = None
    ):
    """  

    TO DO: 
        Add bars (1-20 pixels high) showing spike rate for each unit
    
    Args:
        video_file: path to video file
        LEDs:
        start_frame:
        n_frames: how many frames to get
        start_time: 
        duration:
    """

    # Open video
    cap = cv2.VideoCapture(video_file)
    
    fps = cap.get(cv2.CAP_PROP_FPS)   
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Parse input settings
    if start_time is None:
        if start_frame is None:
            start_frame = 0                 # Play from start by default
            n_frames = np.inf        
    else:                                   # Play a specific duration
        start_frame = np.round(fps * start_time).astype(int)
        n_frames = np.round(fps * duration).astype(int)
        
    current_frame = start_frame
    font = cv2.FONT_HERSHEY_DUPLEX
    white = (255,255,255)

    draw_speed = 'speed' in LEDs.columns
        
    # Add current trial number to each frame
    if trials is not None:

        trials = (
            trials
            .assign(responseframe = lambda x: np.round(x.response_time * fps))
            .astype({'responseframe':'int'})
            .reset_index(names=['trial_num'])
        )

        trials['startframe'] = trials['responseframe'].shift(fill_value=0).astype(int)

        LEDs = pd.merge(LEDs, trials[['trial_num','startframe']], how='left', left_on='frame', right_on='startframe')
        LEDs['trial_num'].iloc[0] = 0
        LEDs['trial_num'] = LEDs['trial_num'].ffill().astype(int)


    # Filter LEDs and sensors for times of interest
    if LEDs is not None:
        LEDs = (
            LEDs
            .drop(LEDs[LEDs['frame'] < start_frame].index)
            .drop(LEDs[LEDs['frame'] > start_frame+n_frames].index)
            .set_index(keys='frame')
        )

    if sensors is not None:
        
        sensors = sensors[sensors['time'] >= start_time]
        sensors = sensors[sensors['time'] < start_time+duration]
        sensors = sensors.set_index(keys='frame')
        # sensors = (
        #     sensors
        #     .drop(sensors[sensors['time'] < start_time].index)
        #     .drop(sensors[sensors['time'] > start_time+duration].index)
        #     .set_index(keys='frame')
        # )

        n_sensors = sensors.shape[1]-1
    else:
        n_sensors = None

    # Open file for saving
    if save:
        save_file = Path(video_file)
        save_file = save_file.parent / (save_file.stem + '_annotated.mp4')

        vout = cv2.VideoWriter(
            filename=str(save_file), 
            fourcc=cv2.VideoWriter_fourcc(*"XVID"), 
            fps=fps, 
            frameSize=(int(width), int(height)))


    # Run the clip
    while(cap.isOpened()):

        # Move to read time
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read frame and move index forward    
        ret, frame = cap.read()
        current_frame += 1
        current_time = current_frame / fps

        # Add time and frame to image
        cv2.putText(frame, f"Time: {(current_time):.1f}s", org=(20, 460), fontFace=font, fontScale=0.5, color=white, thickness=1)
        cv2.putText(frame, f"Frame: {current_frame:.0f}", org=(20, 430), fontFace=font, fontScale=0.5, color=white, thickness=1)


        # Add spike counts for APC array
        if norm_spike_rates is not None:
            draw_APC_activity(frame, norm_spike_rates[0:32, current_frame])
            # draw_PFC_activity(frame, norm_spike_rates[32:,current_frame])
    

        # If adding tracking data
        if LEDs is not None:

            frame_info = LEDs.filter(items=[current_frame], axis=0).to_dict('records')[0]
            
            # Plot current head position (if it exists)
            if not np.isnan(frame_info['headx']) and not np.isnan(frame_info['heady']):
                cv2.drawMarker(frame, (int(frame_info['headx']), int(frame_info['heady'])),
                color=(150,255,150), markerType=cv2.MARKER_CROSS, thickness=1)

            # Plot speed bar
            if draw_speed:
                cv2.line(frame, pt1=(360, 460), pt2=(460, 460), color=(255,200,255), thickness=3)
                cv2.putText(frame, "Speed", org=(300, 465), fontFace=font, fontScale=0.5, color=(255,0,255), thickness=1)

                norm_speed = 360 + np.round(frame_info['speed']/5).astype(int)
                cv2.line(frame, pt1=(360, 460), pt2=(norm_speed, 460), color=(255,0,255), thickness=2)

            # Plot sensors
            if n_sensors:

                sens_info = sensors.filter(items=[current_frame], axis=0).to_dict('records')[0]
                cv2.putText(frame, "Sensor", org=(500, 465), fontFace=font, fontScale=0.5, color=(255,255,255), thickness=1)

                for chan in range(n_sensors):
                    xpos = 580 + chan*10
                    ypos = 460 - int(sens_info[f"sens{chan}"]*20)
                    cv2.line(frame, pt1=(xpos, 465), pt2=(xpos, ypos), color=(255,255,255), thickness=2)

                # print(ypos)

            # Add current trial
            if trials is not None:
                cv2.putText(frame, f"Trial: {(frame_info['trial_num']):.0f}", org=(20, 400),
                fontFace=font, fontScale=0.5, color=white, thickness=1)

            # Plot trail history
            if trail > 1:

                # Find tracking data within range of trail (line that shows history)
                rows = LEDs[(LEDs['frame'] <= current_frame) & (LEDs['frame'] > current_frame-trail)].index
                xy = LEDs[['frame','headx','heady']].loc[rows].dropna()

                if xy.shape[0] > 0:

                    # Plot trail
                    if xy.shape[0] > 1:

                        for i in range(trail-1):        # for each starting point (the rest of this code could be accelerated)
                        
                            start_point = xy.loc[xy[xy['frame'] == current_frame-i].index]
                            end_point = xy.loc[xy[xy['frame'] == current_frame-i-1].index]

                            if start_point.shape[0] > 0 and end_point.shape[0] > 0: 
                                
                                start_point = start_point.to_dict('records')[0]
                                start_x, start_y = int(start_point['headx']), int(start_point['heady'])

                                end_point = end_point.to_dict('records')[0]
                                end_x, end_y = int(end_point['headx']), int(end_point['heady'])

                                cv2.line(frame, pt1=(start_x, start_y), pt2=(end_x, end_y), color=(200,255,200), thickness=1)

        cv2.imshow('Frame', frame)

        if save:
            vout.write(frame)

        # Stop if at end of analysis window
        if current_frame-start_frame > n_frames:
            break

        # Stop if user requests
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    if save:
        vout.release()



def plot_all_positions(df, markersize:int, alpha:float, im_size:Tuple[int, int]):
    """ Plot two-column panel showing pixel locations of blue and red LEDs """

    fig, axs = plt.subplots(1,3, **{'figsize':(10,5)})

    sns.scatterplot(
        data = df,
        x = 'blue_LEDx',
        y = 'blue_LEDy',
        s = markersize,
        hue = 'blue_LEDlikelihood',
        alpha = alpha,
        palette='plasma',
        ax = axs[0],
        legend=False
    )

    sns.scatterplot(
        data = df,
        x = 'red_LEDx',
        y = 'red_LEDy',
        s = markersize,
        hue = 'red_LEDlikelihood',
        alpha = alpha,
        palette='plasma',
        ax = axs[1],
        legend=False
    )

    sns.scatterplot(
        data = df,
        x = 'headx',
        y = 'heady',
        s = markersize,
        hue = 'heady',
        alpha = alpha,
        palette='Wistia',
        ax = axs[2],
        legend=False
    )

    for ax in axs:
        ax.set_xlim([0, im_size[0]])
        ax.set_ylim([0, im_size[1]])
        ax.invert_yaxis()

    return ax


@dataclass
class session_trajectory_plot():
    """ 
    
    Note that there must be at least one set of results to plot
    
     """

    time_window: Tuple
    imsize: Tuple[float,float]
    figsize: Tuple[float,float]=(5.4,5.4)


    def __post_init__(self):
        """ Draw figure canvas upon object creation """
    
        self.fig, self.axs = plt.subplots(4,1, sharex=True, **{'figsize':self.figsize})


    def add_tracking_data(self, df:pd.DataFrame, color:str='k'):
        """ Draw lines showing x and y position over time """

        # Filter input data for plot time
        df = df[(df.time >= self.time_window[0]) & (df.time < self.time_window[1])]

        # Plot lines
        df.plot(x='time',y ='heady', ax=self.axs[0], legend=False, **{'color':color})
        df.plot(x='time',y ='headx', ax=self.axs[1], legend=False, **{'color':color})

        # Format axes
        self.axs[0].set_ylabel('Y (px)')
        self.axs[1].set_ylabel('X (px)')

        self.axs[0].set_ylim((0, self.imsize[0]))
        self.axs[1].set_ylim((0, self.imsize[1]))


    def add_spout_reflines(self, spout_x:int, spout_y) -> None:
        """ Draw reference lines indicating where central platform should be """

        def add_refline(ax, val):
            xlim = ax.get_xlim()
            ax.plot(xlim,[val, val],'--', color='#444444', zorder=0)

        add_refline(self.axs[0], spout_y)
        add_refline(self.axs[1], spout_x)

        self.spout_y = spout_y
        self.spout_x = spout_x


    def add_trial_marks(self, x:np.array, fmt:str='.r') -> None:
        """ Add markers to show the onset of trial times 
        
        Args:
            x: timestamps of events to show
        
        """

        # Filter timestamps for plot time window
        x = x[(x >= self.time_window[0]) & (x < self.time_window[1])]

        # Default values if the class hasn't already had spout location added
        if hasattr(self, 'spout_y'):
            spout_y = self.spout_y
        else:
            spout_y = 0
            
        if hasattr(self, 'spout_y'):
            spout_x = self.spout_x
        else:
            spout_x = 0
    
        # Plot
        self.axs[0].plot(x, np.full_like(x, spout_y), fmt, alpha=0.5, markersize=8)   
        self.axs[1].plot(x, np.full_like(x, spout_x), fmt, alpha=0.5, markersize=8)   

        


    def add_speed_data(self, df:pd.DataFrame, color:str='r') -> None:
        """ Add line plot showing speed vs. time"""

        # Filter input data for plot time
        df = df[(df.time >= self.time_window[0]) & (df.time < self.time_window[1])]

        # Plot lines
        df.plot(x='time',y='speed', ax=self.axs[2], legend=False, **{'color':color})

        self.axs[2].set_ylabel('Speed (pix/s)')


    def add_spiketime_scatter(self, spike_times):
        """ Add scatterplot of spike times during session"""

        for y, (chan_name, chan_spikes) in enumerate(spike_times.items()):

            # Filter for spikes just within plot window
            chan_spikes = chan_spikes[(chan_spikes >= self.time_window[0]) & (chan_spikes < self.time_window[1])]

            y_fill = np.full_like(chan_spikes, y)

            # Plot on bottom row
            self.axs[3].scatter(x=chan_spikes, y=y_fill, s=0.1, c='k')

        # Format axes
        self.axs[3].set_xlabel('Time (s)')
        self.axs[3].set_ylabel('Chan')

    @staticmethod
    def show():
        """ Render in notebook or window """
        plt.tight_layout()
        plt.show()



#         self.primary_data = df[(df.time >= self.time_window[0]) & (df.time < self.time_window[1])]

#         self.min_x, max_x = time_window

#         self.fig, self.axs = plt.subplots(2,1, sharex=True, **{'figsize':(12, 3)})


#     def add_center_lines(self, center):
#         """ Add horizontal bands that indicate the position of the central platform """

#         min_x

#         self.axs[0].plot([min_x, max_x],[center[1], center[1]],'--', color='#444444', zorder=0)  # y
#         self.axs[1].plot([min_x, max_x],[center[0], center[0]],'--', color='#444444', zorder=0)  # x



        
#     def add_trial_times(self, trial_times:np.array):
#         """ Add markers to show when a trial is happening """

#         if center is None:
#             self.axs[0].plot(trial_times, np.ones_like(trial_times), '.r')
#             self.axs[1].plot(trial_times, np.ones_like(trial_times), '.r')
#         else:
#             self.axs[0].plot(trial_times, np.full_like(self.trial_times, center[1]), '.r', alpha=0.5, markersize=8)   # y
#             self.axs[1].plot(trial_times, np.full_like(self.trial_times, center[0]), '.r', alpha=0.5, markersize=8)   # x



def plot_session_trajectory(
    df : pd.DataFrame, 
    fps : Optional[int]=None, 
    trial_times : Optional[np.array]=None,
    center : Optional[Tuple[int, int]]=None,
    time_window : Optional[Tuple[float, float]]=None, 
    axs: Optional=None, 
    color: str='k'):
    """ Plot the position of the animal as a function of time for the entire session
    
    Args:
        center: (x,y) coordinates of estimated center position, where we expect the animal to return to before the start of each trial
        time_window: (start, stop) period of session to consider

    Returns
        axes for plot
    
     """

    if axs is None:
        fig, axs = plt.subplots(2,1, sharex=True, **{'figsize':(12, 3)})

    # Plot versus frame number    
    if fps is None:
        df.plot(x='frame',y ='headx', ax=axs[1], legend=False)
        df.plot(x='frame',y ='heady', ax=axs[0], legend=False)

        min_x, max_x = df['frame'].min(), df['frame'].max()

        axs[1].set_xlabel('Frame')

    # Plot versus time
    else:

        # Filter data for required time if only plotting a specific window
        if time_window is not None:
            df = df[(df.time >= time_window[0]) & (df.time < time_window[1])]

            trial_times = trial_times[(trial_times >= time_window[0]) & (trial_times < time_window[1])]
            min_x, max_x = time_window
        
        # Otherwise get min and max time for full session being plotted
        else:
            min_x, max_x = df['time'].min(), df['time'].max()
        

        df.plot(x='time',y ='heady', ax=axs[0], legend=False, **{'color':color})
        df.plot(x='time',y ='headx', ax=axs[1], legend=False, **{'color':color})

        
        if center is None:
            axs[0].plot(trial_times, np.ones_like(trial_times), '.r')
            axs[1].plot(trial_times, np.ones_like(trial_times), '.r')
        else:
            axs[0].plot(trial_times, np.full_like(trial_times, center[1]), '.r', alpha=0.5, markersize=8)   #y
            axs[1].plot(trial_times, np.full_like(trial_times, center[0]), '.r', alpha=0.5, markersize=8)   # x

        axs[1].set_xlabel('Time (s)')

    
    axs[0].set_ylabel('Y (px)')
    axs[1].set_ylabel('X (px)')

    # Add reference lines for center (optional)
    if center is not None:

        axs[0].plot([min_x, max_x],[center[1], center[1]],'--', color='#444444', zorder=0)  # y
        axs[1].plot([min_x, max_x],[center[0], center[0]],'--', color='#444444', zorder=0)  # x


    return axs

@dataclass()
class trial_trajectory_plot():

    data: np.array
    im_size: Tuple[int, int]
    time_window: Tuple[float, float]
    fig_size: Tuple[float, float] = (12.0, 6.1)
 
    def __post_init__(self):

        # Get both mean and standard deviation
        self.stats = dict(
            mean = np.nanmean(self.data, axis=0),
            std = np.nanstd(self.data, axis=0)
        )

        # Create time vector
        self.n_trials, self.n_samps, _ = self.data.shape
        self.time_vec = np.linspace(self.time_window[0], self.time_window[1], num=self.n_samps)
        
        zero_point = np.interp(0, self.time_vec, np.arange(self.n_samps))

        self.heatmap_xticks = [0, zero_point, self.n_samps-1]
        self.heatmap_xticklabels = [str(self.time_window[0]), '0', str(self.time_window[1])]

        # Create canvas
        self.fig, self.axs = plt.subplots(2,2, **{'figsize':self.fig_size})
        # cmap = Colormap('rainbow', N=self.n_trials)

    
    def draw_mean(self):
        """ Draw average trajectory across trials """
        
        self.axs[1,0].errorbar(x=self.time_vec, y=self.stats['mean'][:,0], yerr=self.stats['std'][:,0])
        self.axs[1,1].errorbar(x=self.time_vec, y=self.stats['mean'][:,1], yerr=self.stats['std'][:,1])

        self.axs[1,0].set_xlabel('Time (s)')
        self.axs[1,1].set_xlabel('Time (s)')

        self.axs[1,0].set_xlabel('X (px)')
        self.axs[1,1].set_xlabel('Y (px)')

        self.axs[1,0].set_ylim((0, self.im_size[1]))
        self.axs[1,1].set_ylim((0, self.im_size[0]))


    def draw_heatmap(self):
        """ Draw heatmaps of x and y position as a function of time
        for every trial """

        self.axs[0,0].imshow(self.data[:,:,0])
        self.axs[0,1].imshow(self.data[:,:,1])

        def convert_ticks_to_time(ax, xtick, xticklabels):    
            ax.set_xticks(xtick)
            ax.set_xticklabels(xticklabels)

        convert_ticks_to_time(self.axs[0,0], self.heatmap_xticks, self.heatmap_xticklabels)
        convert_ticks_to_time(self.axs[0,1], self.heatmap_xticks, self.heatmap_xticklabels)

        self.axs[0,0].set_xlabel('Time (s)')
        self.axs[0,1].set_xlabel('Time (s)')
        
        self.axs[0,0].set_ylabel('Trial')
        self.axs[0,1].set_ylabel('Trial')




def scatter_at_time(df:pd.DataFrame, fps:float, t:np.array, im_size:Tuple[int, int]):
    """ Show the scatter plot of xy positions of the head at a trigger time """

    # Get LED positions in frames before and after event 
    traj = vtran.get_trial_trajectories(df, fps, t, [-1/fps, 1/fps])
    traj = np.squeeze(np.mean(traj, axis=1))

    # Plot as a nice scatter plot that's definitely going to be very pretty
    fig, ax = plt.subplots(1,1)

    ax.scatter(traj[:,0], traj[:,1], s=3, alpha=0.5)
   
    ax.set_xlim([0, im_size[0]])
    ax.set_ylim([0, im_size[1]])
    ax.invert_yaxis()


    plt.show()


@dataclass
class StimTriggeredSyncHeatmap:

    data: pd.DataFrame
    auto_plot: bool=True

    def __post_init__(self):

        # Get time vector from data
        self.tvec = np.array([float(t[:-1]) for t in self.data.columns])
        self.frame_idx = np.round(self.tvec * 30)

        self.thresholded_data = self.threshold_signal()

        # Automatically generate plots without further action (legacy)
        if self.auto_plot:
            
            self.fig, self.axs = plt.subplots(1,2, sharex=True, sharey=True)

            self.plot_heatmap(self.axs[0], self.data.to_numpy())
            self.plot_heatmap(self.axs[1], self.thresholded_data)


    def report_empty_trials(self):
        """ Report how many trials have no signal at all (not even noise) """

        n_trials_empty = sum(self.data.sum(axis=1) == 0)
        p_trials_empty = np.mean(self.data.sum(axis=1) == 0)
        n_trials_total = self.data.shape[0]

        print(f"{n_trials_empty} of {n_trials_total} have no signal ({p_trials_empty*100:.1f}%)")


    def plot_foreach_stimulus_location(self, trial_info:pd.DataFrame, cbar_on:bool=True):

        # Create columns for each LED location
        ax_idx = -1
        n_locations = len(trial_info['visual_location'].unique())
        
        self.fig, self.axs = plt.subplots(n_locations, 1, sharex=True, sharey=True)

        for location, loc_data in trial_info.groupby('visual_location'):
            
            ax_idx += 1
            frame_vals = self.data.iloc[loc_data.index]

            self.plot_heatmap(self.axs[ax_idx], frame_vals, cbar_on=cbar_on, cbar_loc='right')
            self.axs[ax_idx].set_title(location, fontsize=8)

        self.axs[ax_idx].set_xlabel('Frame')            


    def manual_plot(self):


        self.fig, self.axs = plt.subplots(1,2, sharex=True, sharey=True)

        self.plot_heatmap(self.axs[0], self.data.to_numpy())        
        self.plot_heatmap(self.axs[1], self.thresholded_data)




    def threshold_signal(self):
        """ Turn visual signal image (0-255) to binary (0-1) based on whether value was above rms """

        z = self.data.to_numpy()
        rms = np.sqrt(np.mean(z**2, axis=1))

        thresholded_data = np.zeros_like(z)

        for row in range(len(rms)):
            thresholded_data[row, z[row,:] > rms[row]] = 1

        return thresholded_data


    def plot_heatmap(self, ax, z, cbar_on:bool=True, cbar_loc:str='bottom'):
        """ Plot original data that we expect to show the stimulus happening earlier and earlier across trials 
        
        z: data, where rows = trials, columns = frames
        """
       
        imh = ax.imshow(z)

        if cbar_on:
            plt.colorbar(imh, ax=ax, location=cbar_loc)

        ax.set_ylim([0, z.shape[0]])
        ax.set_ylabel('Trial')

        self.update_xticks(ax)
        # ax.invert_yaxis()


    def update_xticks(self, ax):
        # Update ticks to show frame relative to stimulus trigger

        trigger_idx = np.where(self.frame_idx == 0.0)[0][0]
        xticks = [0, trigger_idx, len(self.frame_idx)-1]
        xticklabels = [self.frame_idx[0], 0.0, self.frame_idx[-1]]

        ax.set_xlim((0, len(self.frame_idx)-1))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticklabels])
        
#######################################################################################
# Calibration image / bounding box plots        

def draw_bounding_box_on_img(img, df:pd.DataFrame):
    """ 
    Args:
        img: image, either a frame from a video or the data in a calibration image    
        df: dataframe listing bounding box information for each spout

    Returns
        OpenCV window showing image, press q to close
    """

    for spout, sdata in df.groupby('spout'):
        
        box = sdata.to_dict('records')[0]

        start_point = (box['start_col'], box['start_row'])
        end_point = (box['start_col']+box['width'], box['start_row']+box['height'])
        mid_point = (box['start_col']+int(box['width']/2), box['start_row']+int(box['height']/2))

        cv2.rectangle(img, start_point, end_point,(0,255,0), 1)
        cv2.putText(img, str(spout), mid_point, 0, 0.3, (0,255,0))

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def draw_boundingbox_on_calib_image(calib_im:str):
    """ Show bounding boxes on an image with labels indicating which spout is which.
    """

    sample_im_dir = Path('data/sample/calibration_images')
    sample_im_path = sample_im_dir / calib_im
    
    img = cv2.imread(str(sample_im_path))
    
    df = vload.get_bounding_boxes(calib_im)

    draw_bounding_box_on_img(img, df)


def get_calib_im_for_vid( video_file:str) -> str:
    """  Need to upgrade to database when eventually ready"""

    df = pd.read_csv('data/tables/video_calib_images_manual.csv')
    idx = df[df['video_file'].str.match(video_file)].index

    return df.loc[idx, 'calib_image'].to_list()


def draw_bounding_box_on_frame(video_file:str):
    """ Show bounding boxes on a frame from a video """

    # Read image from camera
    load_dotenv()
    video_path = Path(os.getenv('local_home')) / 'Task_Switching/videos' 
    
    cap = cv2.VideoCapture( str(video_path / video_file))
    res, img = cap.read()
    assert res

    # Get calibration information for this video
    calib_im = get_calib_im_for_vid( video_file)
    assert(len(calib_im) == 1)

    df = vload.get_bounding_boxes(calib_im[0])

    draw_bounding_box_on_img(img, df)





def main():

    # file_path = Path('/home/stephen/Data/Task_Switching/head_tracking/new_stimTriggeredFrameVals')
    # file_name = 'F1613_Block_J4-45.csv'
    # test = StimTriggeredSyncHeatmap(file_path / file_name)
    # plt.show()

    # draw_boundingbox_on_calib_image("2016-09-27 17_51_31.jpg")
    draw_bounding_box_on_frame("F1613_Ariel_Block_J4-13_Vid0.avi")


    # fp = '/home/stephen/Data/Task_Switching/videos/F1605_Snorlax_Block_J4-43_Vid0.avi'

    # if Path(fp).exists():
    #     # read_video(video_file=fp, start_frame=1000, n_frames=300)
    #     play_video(video_file=fp, start_time=90.1, duration=50.1)
    


if __name__ == '__main__':
    main()