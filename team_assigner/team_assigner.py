from sklearn.cluster import KMeans
import sys
sys.path.append('../')
from utils import measure_distance, distance_frame


class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}
        self.team_1_goalkeeper = 206
        self.team_2_goalkeeper = 97
        
    def get_clustering_model(self, image):
                #rechape the image into 2d array
        image_2d = image.reshape(-1,3)

        #performs kMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2,init="k-means++",n_init=1, random_state=42)
        kmeans.fit(image_2d)
        
        return kmeans
    
    def get_player_color (self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #only the half
        top_half_image = image[0:int(image.shape[0]/2),:]
        #get clusterin model
        kmeans = self.get_clustering_model(top_half_image)
    
        #get the cluster labels for each pixel
        labels = kmeans.labels_

        #rechape the labels into the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # get the players cluster
        
        corner_cluster = [clustered_image[0,0],
                  clustered_image[0,-1],
                  clustered_image[-1,0],
                  clustered_image[-1,-1]]
        
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = non_player_cluster - 1
        
        player_color = kmeans.cluster_centers_[player_cluster]
        
        return player_color
        
    def assign_team_color(self, frame, player_detections):
        # Entrenar un nuevo modelo si read_from_stub es False o el archivo de stub no existe
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Entrenar el modelo KMeans
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=100, random_state=24)
        self.kmeans.fit(player_colors)
            
        # Asignar los colores del equipo
        self.team_color[1] = self.kmeans.cluster_centers_[0]
        self.team_color[2] = self.kmeans.cluster_centers_[1]


        
    def get_player_team(self, frame, player_bbox, player_id):
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1
        
        
        #Asiggned goalkeepers teams
        if player_id == self.team_1_goalkeeper:
            team_id = 1  
        if player_id == self.team_2_goalkeeper:
            team_id = 2            
         
        return team_id
    
    ################################################################################################################
    ############################ From here is we are goin to coorect the players id ################################
    ################################################################################################################
    
    # Assing the main players from each team 
    def first_players_id_detected(self, tracks, frames_detected=24):
        prime_players_team_1 = [self.team_1_goalkeeper]
        prime_players_team_2 = [self.team_2_goalkeeper]

        players_detect_team_1 = {}
        players_detect_team_2 = {}

        for frame_num, player_tracks in enumerate(tracks['players']):
            for player_id, track in player_tracks.items():
                if track['team'] == 1:
                    if player_id in players_detect_team_1:
                        players_detect_team_1[player_id] += 1
                    else:
                        players_detect_team_1[player_id] = 1
                elif track['team'] == 2:
                    if player_id in players_detect_team_2:
                        players_detect_team_2[player_id] += 1
                    else:
                        players_detect_team_2[player_id] = 1
            
            for player_id, count in players_detect_team_1.items():
                if count == frames_detected and player_id not in prime_players_team_1 and player_id not in prime_players_team_2 and len(prime_players_team_1) < 11:
                    prime_players_team_1.append(player_id)
                        
            for player_id, count in players_detect_team_2.items():
                if count == frames_detected and player_id not in prime_players_team_2 and player_id not in prime_players_team_1 and len(prime_players_team_2) < 11:
                    prime_players_team_2.append(player_id)
                        
        return prime_players_team_1, prime_players_team_2
    
    # funtion to modify players_id number
    def modify_player_id(self, tracks, old_id, list_new_id):
        for frame_num, player_tracks in enumerate(tracks['players']):
            if old_id in player_tracks:
                if len(list_new_id) != 0:
                    new_id = list_new_id[0]
                    player_tracks[new_id] = player_tracks.pop(old_id)
                    tracks['players'][frame_num] = player_tracks
                else:
                    del tracks['players'][frame_num][old_id]
        return tracks
    
    # Funtion 
    def clean_players_id(self, tracks,prime_players_team_1, prime_players_team_2, n_frames_looking_back=40, min_distance=75):
        prime_players_both_teams = prime_players_team_1.copy()
        prime_players_both_teams.extend(prime_players_team_2)
        
        frame_num = 0  # We initialize the frame_num counter
        
        while frame_num < len(tracks['players']):
            player_tracks = tracks['players'][frame_num]
            
            # Variable to indicate if any modification was made to the loop
            modified = False
            
            # List to store the players that need to be modified
            players_to_modify = []
            
            for player_id, track in player_tracks.items():
                if player_id in prime_players_both_teams:
                    continue
                
                # We look for prime_ids that could be candidates to be replaced that do not overlap in the frames
                ##1st we look for the candidates who are not in the frame that we are already looking for to clear several becoming the pre-candidates (1st round)
                
                first_posible_players_id = []
                second_posible_players_id = []
                third_posible_players_id = []
                
                for prime_player_id in prime_players_both_teams:
                    if prime_player_id not in tracks['players'][frame_num]:
                        first_posible_players_id.append(prime_player_id)
                
                # 2nd Now we look at what frames the player_id is in
                frames_player_id = []
                for player_frame_num, _ in enumerate(tracks['players']):
                    if player_id in tracks['players'][player_frame_num]:
                        frames_player_id.append(player_frame_num)
                
                # we look in each of these frames which of the pre-candidates would not be superimposed on another frame with player_id (they go to the 2nd round). 
                for prime_player_id in first_posible_players_id:
                    frames_prime_player_id = []
                    for prime_frame_num, _ in enumerate(tracks['players']):
                        if prime_player_id in tracks['players'][prime_frame_num]:
                            frames_prime_player_id.append(prime_frame_num)
                
                    if not set(frames_player_id).intersection(frames_prime_player_id):
                        second_posible_players_id.append(prime_player_id)
                
                print(f'frame {frame_num} posible IDs for {player_id} are {second_posible_players_id}')
                
                # first we find the (adjusted) position of the first time the player_id appears
                player_position = track['position_adjusted']
                print(f'En el frame {frame_num} para el player {player_id} se encuentra en {player_position} por primera vez')    

                ## passes to the next round we discard the candidates in relation to their distance between player_id and the last time it appeared
                
                # EWe start searching frames backwards (no further than 24 frames) until we find the last location of the possible possible_prime_player_id
                for posible_prime_player_id in second_posible_players_id:
                    frame_numer_location = frame_num - 1
                    frame_number_limit = frame_num - n_frames_looking_back
                    
                    while frame_numer_location >= frame_number_limit:
                        if frame_numer_location in range(len(tracks['players'])):
                            
                            if posible_prime_player_id in tracks['players'][frame_numer_location]:
                                prime_player_position = tracks['players'][frame_numer_location][posible_prime_player_id]['position_adjusted']
                                players_distance = measure_distance(player_position, prime_player_position)
                                
                                if players_distance < min_distance:
                                    third_posible_players_id.append(posible_prime_player_id)
                                    break  # Exit the while loop if we find the analyzed player is TRUE
                            frame_numer_location -= 1  # Move to next frame
                        else:
                            break  # Exit the while loop if we go out of frame range.

                
                print(f'sus candidatos finales para {player_id} son {third_posible_players_id}')
                # If we find players to modify, we modify them and mark them as modified
                
                # If we find players to modify, we add them to the players_to_modify list
                if third_posible_players_id:
                    players_to_modify.append((player_id, third_posible_players_id))
            
            # Modify the players after you have finished iterating over the dictionary
            for player_id, new_ids in players_to_modify:
                tracks = self.modify_player_id(tracks, player_id, new_ids)
                modified = True
            
            # If any modification was made, we restart the loop
            if modified:
                frame_num = 0
                continue
            
            frame_num += 1  # Increment the frame_num counter to move to the next frame
        
        return tracks
    
    def drop_not_prime_players(self, tracks, prime_players_team_1, prime_players_team_2):
        prime_players_both_teams = prime_players_team_1.copy()
        prime_players_both_teams.extend(prime_players_team_2)
        
        players_to_drop = []
        
        # First we identify the players_id that are not prime and that we want to eliminate
        for frame_num, player_tracks in enumerate(tracks['players']):
            for player_id, _ in player_tracks.items():
                if player_id not in prime_players_both_teams and player_id not in players_to_drop:
                    players_to_drop.append(player_id)

        # We remove the non-prime players_id after we have finished iterating
        for player_id in players_to_drop:
            for frame_num, player_tracks in enumerate(tracks['players']):
                # Check if the player_id is present in the current frame before trying to delete it
                if player_id in player_tracks:
                    del tracks['players'][frame_num][player_id]

    def swap_players_id(self, tracks, frame_num, player_a, player_b):
        for swap_frame in range(frame_num, len(tracks['players'])):
            track_info_player_a = None
            track_info_player_b = None
            if player_a in tracks['players'][swap_frame]:
                track_info_player_a = tracks['players'][swap_frame][player_a]
                del tracks['players'][swap_frame][player_a]
            if player_b in tracks['players'][swap_frame]:
                track_info_player_b = tracks['players'][swap_frame][player_b]        
                del tracks['players'][swap_frame][player_b]
            if track_info_player_a is not None and track_info_player_b is not None:
                #  We exchange player information
                tracks['players'][swap_frame][player_a] = track_info_player_b
                tracks['players'][swap_frame][player_b] = track_info_player_a
            elif track_info_player_a is None and track_info_player_b is not None:
                # If player_a is not present but player_b is, player_a becomes None
                tracks['players'][swap_frame][player_a] = track_info_player_b
            elif track_info_player_a is not None and track_info_player_b is None:
                # If player_a is present but player_b is not, player_b becomes None
                tracks['players'][swap_frame][player_b] = track_info_player_a
            # If both are None, we do not make any changes

        return tracks
    
    
    
    def switch_player_id(self, tracks, prime_players_team_1, prime_players_team_2, frames_analized=24, buffer_frames=16, min_distance=40):

        frame_num = 0  # We initialize the frame_num counter

        while frame_num < len(tracks['players']):
            player_tracks = tracks['players'][frame_num]

            for player_id, track in player_tracks.items():
                # 1. having the wrong team assigned in the frame (they would be the players who are from team_2 GREEN)
                if track['team'] == 1:
                    if player_id in prime_players_team_1:
                        continue
                # 2. Let's see if in the next 24 frames at least buffer_frames are also on the wrong team
                    frames_wrong_team = 0
                    for obs_frame in range(frame_num, min(frame_num + frames_analized, len(tracks['players']))):
                        if player_id in tracks['players'][obs_frame]:
                            if tracks['players'][obs_frame][player_id]['team'] == 1:
                                frames_wrong_team += 1
                        if frames_wrong_team == buffer_frames:
                            break
                # 3. Look for players from the opposing team who are also on the wrong team (candidates
                    if frames_wrong_team == buffer_frames:
                        first_possible_crossover_players = []
                        for opponent_player_id, opponent_track in tracks['players'][frame_num].items():
                            if opponent_track['team'] == 2 and opponent_player_id in prime_players_team_1:
                                frames_wrong_opponent_team = 0
                                for obs_frame in range(frame_num, min(frame_num + frames_analized, len(tracks['players']))):
                                    if opponent_player_id in tracks['players'][obs_frame]:
                                        if tracks['players'][obs_frame][opponent_player_id]['team'] == 2:
                                            frames_wrong_opponent_team += 1
                                    if frames_wrong_opponent_team == buffer_frames:
                                        first_possible_crossover_players.append(opponent_player_id)
                                        break

                        # 4. the distance between them should be < 40 pixels
                        # If we have more than one candidate, we will stick with the one that is closest
                        min_distance_found = min_distance + 1  # We initialize with a value greater than the minimum allowed
                        best_crossover_player = None
                        for possible_crossover_player in first_possible_crossover_players:
                            distance_between_players = distance_frame(tracks,frame_num, possible_crossover_player, player_id)
                            if distance_between_players < min_distance_found:
                                min_distance_found = distance_between_players
                                best_crossover_player = possible_crossover_player
                        if best_crossover_player is not None:
                            # Perform the exchange of player IDs in all frames
                            self.swap_players_id(tracks, frame_num, player_id, best_crossover_player)
                            print(f'Jugadores {best_crossover_player} y {player_id} cruzados en el frame {frame_num}')
                            break
                    
                # 1. having the wrong team assigned in the frame (they would be the players who are from team_1 WHITE)
                elif track['team'] == 2:
                    if player_id in prime_players_team_2:
                        continue
                # 2. Let's see if in the next 24 frames at least buffer_frames are also on the wrong teamo
                    frames_wrong_team = 0
                    for obs_frame in range(frame_num, min(frame_num + frames_analized, len(tracks['players']))):
                        if player_id in tracks['players'][obs_frame]:
                            if tracks['players'][obs_frame][player_id]['team'] == 2:
                                frames_wrong_team += 1
                        if frames_wrong_team == buffer_frames:
                            break
                # 3. Look for players from the opposing team who are also on the wrong team (candidates)
                    if frames_wrong_team == buffer_frames:
                        first_possible_crossover_players = []
                        for opponent_player_id, opponent_track in tracks['players'][frame_num].items():
                            if opponent_track['team'] == 1 and opponent_player_id in prime_players_team_2:
                                frames_wrong_opponent_team = 0
                                for obs_frame in range(frame_num, min(frame_num + frames_analized, len(tracks['players']))):
                                    if opponent_player_id in tracks['players'][obs_frame]:
                                        if tracks['players'][obs_frame][opponent_player_id]['team'] == 1:
                                            frames_wrong_opponent_team += 1
                                    if frames_wrong_opponent_team == buffer_frames:
                                        first_possible_crossover_players.append(opponent_player_id)
                                        break
                        # 4. the distance between them must be < 40 pixels
                        #  If we have more than one candidate, we will stick with the one that is closest
                        min_distance_found = min_distance + 1  # We initialize with a value greater than the minimum allowed
                        best_crossover_player = None
                        for possible_crossover_player in first_possible_crossover_players:
                            distance_between_players = distance_frame(tracks, frame_num, possible_crossover_player, player_id)
                            if distance_between_players < min_distance_found:
                                min_distance_found = distance_between_players
                                best_crossover_player = possible_crossover_player
                        if best_crossover_player is not None:
                            # Realizar el intercambio de IDs de jugadores en todos los frames
                            self.swap_players_id(tracks, frame_num, player_id, best_crossover_player)
                            print(f'Jugadores {best_crossover_player} y {player_id} cruzados en el frame {frame_num}')
                            break


            frame_num += 1  # Increment the frame_num counter to move to the next frame

        return tracks
    
    def correct_asigment_team(self,tracks, prime_players_team_1, prime_players_team_2):
        for frame_num, player_tracks in enumerate(tracks['players']):    
            for player_id, track in player_tracks.items():
                if player_id in prime_players_team_1:
                    tracks['players'][frame_num][player_id]['team'] = 1
                    tracks['players'][frame_num][player_id]['team_color'] = self.team_color[1]
                elif player_id in prime_players_team_2:
                    tracks['players'][frame_num][player_id]['team'] = 2 
                    tracks['players'][frame_num][player_id]['team_color'] = self.team_color[2]
                else:
                    print(f'hmmm in the frame {frame_num} the player {player_id} its not in any of both teams. this happens when when appears new player_id that should be clean in the first place')
                    
                    