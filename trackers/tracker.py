from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import cv2 
import pandas as pd


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def interpolate_ball_position (self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        
        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def detect_frames(self,frames):
        # takes the list of frames and processes the detections, returning a list of Attributes ("boxes", "masks", "probs", "keypoints", "obb",  etc.) 
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            
        return detections
    
    def get_objet_tracks(self, frames, read_from_stub=False, stub_path=None):
                
        # In case stub_path has already been done, you save the entire function and do not execute it
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks    
        
        detections = self.detect_frames(frames)
        
        tracks={
        "players":[],  # [frame_num][track_id] = {'bbow:[x1,y1,x2,y2]}
        "referees":[],
        "ball":[]
        }
                
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names # attribute which is a dict of what names {0:ball, 1:player ,2:goalkeeper, 3:referee}
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            #convert to supervision Detection format:
            detection_supervision = sv.Detections.from_ultralytics(detection) # gives you different attributes (.xyxy, .confidence , .class_id, .tracker_id, .data)
            
            #convert goalkeeper to player
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_id] = cls_names_inv['player']
                
            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
            
                if cls_id == cls_names_inv["player"]: # the condition of a listing of TRUE, TRUE, TRUE, FALSE, TRUE... and so that the following line is fulfilled only with "players"
                    tracks["players"][frame_num][track_id] = {"bbox":bbox} # save the coordinates in tracks{players:{frame_num:{track_id: [x1, y1, x2, y2]}}}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}        
        
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center = (x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle= -45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
            )
        
        rectangle_with = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_with//2
        x2_rect = x_center + rectangle_with//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect+12
            if x1_text > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
                        
        return frame
        
    def draw_triangle(self, frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0,color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0,(0,0,0),2)
        
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_1_color=None, team_2_color=None):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Get the number of times each team had ball control
        team_1_num_frames = np.count_nonzero(team_ball_control_till_frame == 1)
        team_2_num_frames = np.count_nonzero(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames
        
        if total_frames == 0:
            team_1_percentage = 50.0
            team_2_percentage = 50.0
        else:
            team_1_percentage = (team_1_num_frames / total_frames) * 100
            team_2_percentage = (team_2_num_frames / total_frames) * 100
        
        # Draw text with ball control percentages
        cv2.putText(frame, f'Team 1 Ball Control: {team_1_percentage:.2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2_percentage:.2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        # Draw circles to the left of the text
        circle_radius = 10

        # Draw circle for Team 1
        if team_1_color is not None:
            team_1_color = tuple(int(c) for c in np.clip(team_1_color, 0, 255))  # Ensure BGR format and correct range
            cv2.circle(frame, (1385 - circle_radius, 885 + circle_radius - 5), circle_radius, team_1_color, -1)
        
        # Draw circle for Team 2
        if team_2_color is not None:
            team_2_color = tuple(int(c) for c in np.clip(team_2_color, 0, 255))  # Ensure BGR format and correct range
            cv2.circle(frame, (1385 - circle_radius, 935 + circle_radius - 5), circle_radius, team_2_color, -1)
        
        return frame
    

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position
                    
        
     
    def draw_annotations(self, video_frames, tracks, team_ball_control, team_1_color=None, team_2_color=None):
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num] 
            
            #draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))
                     
            #draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,225,225))
            
            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))
            
            #Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_1_color, team_2_color)
            
                                
            output_video_frames.append(frame)
            
        return output_video_frames
    

        