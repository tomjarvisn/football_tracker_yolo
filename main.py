from utils import read_video,save_video
from trackers import Tracker
import cv2 
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallassigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

import pprint
import time

import os
os.environ['OMP_NUM_THREADS'] = '1'#limitará OpenMP a utilizar un solo hilo
start_time = time.time()


def main():
    #Read the video
    video_frames = read_video('input_videos/08fd33_4.mp4')

   
    #initialize tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_objet_tracks(video_frames,
                                     read_from_stub=True,   #cambiar desp a true cuando terminemos
                                     stub_path='stubs/track_stubs.pkl')

    #itnerpolate Ball positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    #get objet position
    tracker.add_position_to_tracks(tracks)

    #Estimate Camera Movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    # Make camera movements smooth
    camera_movement_per_frame = camera_movement_estimator.apply_smooth_camera_movement(camera_movement_per_frame, half_win_size=10)
    
    #Adjust position from camera movement
    camera_movement_estimator.adjust_positions_to_tracks(tracks,camera_movement_per_frame)
        
    #Get the cordinates of the corners of the feild for every frame    
    correction = -150
    
    initial_conrdinates_points = [
    [-450+correction,270], # arriba izq 
    [-2000+correction,1050], #abajo izq
    [3100+correction,850],  # abajo der
    [1600+correction,240] # arriba der
    ]
    coordinates_points_over_frames = camera_movement_estimator.calculate_coordinates_points_over_frames(initial_conrdinates_points, camera_movement_per_frame)
        
    #View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_tracks(tracks,coordinates_points_over_frames)
    
    # coordinates_points_over_field_graph
    graph_edge = 5
    coordinates_points_over_graph = [
        [1450 + graph_edge, 10 + graph_edge],  # top-left
        [1450 + graph_edge, 270 - graph_edge],  # botton-left
        [1950 - graph_edge, 270 - graph_edge],  # botton-right
        [1950 - graph_edge, 10 + graph_edge]  # top-right
    ]

    # Aplicar la nueva transformación gráfica
    view_transformer.add_graph_position_to_graph(tracks, coordinates_points_over_graph)
    
    
    # Speed and Distance estimator
    speed_and_speed_estimator = SpeedAndDistance_Estimator()
    speed_and_speed_estimator.add_speed_and_distance_to_tracks(tracks)
    
        
    # Assigner player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[480],
                                    tracks['players'][480])
    
    # the first asignment from the model. (it might have errors)
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_color[team]
    
    #cleaning the assignment from the model        
    prime_players_team_1, prime_players_team_2 = team_assigner.first_players_id_detected(tracks)
    team_assigner.clean_players_id(tracks,prime_players_team_1, prime_players_team_2)
    team_assigner.drop_not_prime_players(tracks,prime_players_team_1, prime_players_team_2)
    team_assigner.switch_player_id(tracks, prime_players_team_1, prime_players_team_2, frames_analized=24, buffer_frames=16, min_distance=40)
    team_assigner.correct_asigment_team(tracks, prime_players_team_1, prime_players_team_2) 
    
    
    # Assign Ball aquisition
    player_assigner = PlayerBallassigner()
    
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]) #assigne the last team with the 'assigned_player'
    
    team_ball_control = np.array(team_ball_control)
    
    # draw output
    
    ## draw objet tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control,
                                                   team_1_color = team_assigner.team_color.get(1),
                                                   team_2_color = team_assigner.team_color.get(2))
    
    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ##draw speed and distance
    output_video_frames = speed_and_speed_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    #draw field graph
    output_video_frames = view_transformer.draw_position_in_graph(output_video_frames, tracks, field_image_path='input_videos/football_field.jpg')
    
    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
    with open('tracks_output.txt', 'w') as f:
        pprint.pprint(tracks, stream=f)
    
    #Checking the time tha takes the script
    end_time = time.time()
    total_time = end_time - start_time    
    print(f"Execution time: {total_time/60:.2f} minutes")

if __name__ == '__main__':
    main()