import pickle
import cv2 
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance,measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimun_distance = 1
        self.lk_params = dict(
            winSize=(20, 20),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        )

        first_frame_graysacle = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_features = np.zeros_like(first_frame_graysacle)
        mask_features[ : , 0:300 ] = 1
        mask_features[  100: , 600:950] = 1
        mask_features[  : , 1250:1900] = 1
        mask_features[ : , 1400:1900 ] = 1
        mask_features[ 850: , : ] = 1

        
        self.features = dict(  #parameters for corner detection
            maxCorners = 300,
            qualityLevel = 0.92,
            minDistance = 5,
            blockSize = 10,
            mask = mask_features
        )
    
    def adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] + camera_movement[0], 
                                         position[1] + camera_movement[1] * 1)
                    
                    
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
        
    
    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        #Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0,0]]*len(frames)  #make list of frame sizes
        
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY) #leaves the 1st frame in grayscale
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features) #and looks for the best features
        
        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params) # compares original frame with the frame that is in FOR.. and calculates its flow using the Lucas-Kanade method 
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0
            for i, (new,old) in enumerate(zip(new_features,old_features)): #(i, ((10, 20), (15, 25)))....
                new_features_point = new.ravel()  #[10, 20]
                old_features_point = old.ravel()  #[15, 25]
                distance = measure_distance(old_features_point, new_features_point)
                
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point) 
                    
            if max_distance > self.minimun_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y] #list that shows how much movement in x-axis and y-axis for each frame
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features) #before finishing using the frame we leave it as old_features to compare it with the next one
            
            old_gray = frame_gray.copy() # lo mismo para old_frame
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)
        
            
        return camera_movement
    
    def apply_smooth_camera_movement(self, movements, half_win_size):
        smooth_movements = []

        for i in range(len(movements)):
            start_index = max(0, i - half_win_size)
            end_index = min(len(movements), i + half_win_size + 1)
            window = movements[start_index:end_index]
            avg_movement = np.median(window, axis=0)
            smooth_movements.append(avg_movement)

        return smooth_movements
    
    def calculate_coordinates_points_over_frames(self, initial_points, camera_movement_per_frame):
        # Initialize the dictionary to store the coordinates of each point
        coordinates_points = {
            "point_a_up_left": [],
            "point_b_down_left": [],
            "point_c_down_right": [],
            "point_d_up_right": []
        }
        
        accumulated_x = 0
        accumulated_y = 0
        
        # Mapping starting points to their labels
        points_map = {
            "point_a_up_left": initial_points[0],
            "point_b_down_left": initial_points[1],
            "point_c_down_right": initial_points[2],
            "point_d_up_right": initial_points[3]
        }
        
        # Calculate the adjusted coordinates for each frame
        for movement_x, movement_y in camera_movement_per_frame:

            accumulated_x += movement_x * 1.8 #2
            accumulated_y += movement_y * 0  #After testing the adjustment to the y axis hurt more than it helped because the algorithm. not the best for 3d moviesmentional

            
            for point_label, initial_point in points_map.items():
                adjusted_x = initial_point[0] + accumulated_x 
                adjusted_y = initial_point[1]  + accumulated_y
                

                coordinates_points[point_label].append([adjusted_x, adjusted_y])
        
        return coordinates_points
    
    
    
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        
        for frame_num, frame in enumerate (frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            frame = cv2.putText(frame,f"Camera Movement X:{x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y:{y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            
            output_frames.append(frame)
        
        return output_frames