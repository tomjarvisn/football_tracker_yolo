import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        self.court_width = 68
        self.court_length = 105  # largo y ancho del área de la cancha (en metros)

        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]  # las 4 coordenadas de vértices (en metros) en un área cuadrada ▯
        ]).astype(np.float32)
        
        self.perspective_transformer = None
        self.pixel_vertices = None  # Inicializar pixel_vertices como None

    def update_pixel_vertices(self, pixel_vertices):
        """Actualizar la matriz de transformación de perspectiva con nuevos vértices de píxeles."""
        self.pixel_vertices = np.array(pixel_vertices).astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = int(point[0]), int(point[1])
        # Comprobar si el punto está dentro de las coordenadas de píxeles
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshape_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshape_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)

    def add_transform_position_to_tracks(self, tracks, coordinates_points_over_frames):
        for frame_num, points in enumerate(zip(*coordinates_points_over_frames.values())):
            pixel_vertices = [list(p) for p in points]
            self.update_pixel_vertices(pixel_vertices)
            
            for object, object_tracks in tracks.items():
                for frame_num, track in enumerate(object_tracks):
                    for track_id, track_info in track.items():
                        position = track_info['position_adjusted']     
                        position = np.array(position)
                        position_transformed = self.transform_point(position)
                        if position_transformed is not None:
                            position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num][track_id]['position_transformed'] = position_transformed

    def add_graph_position_to_graph(self, tracks, coordinates_points_over_graph):
        """Agrega posiciones graficadas transformadas a tracks."""
        graph_vertices = np.array(coordinates_points_over_graph).astype(np.float32)
        graph_transformer = cv2.getPerspectiveTransform(self.target_vertices, graph_vertices)

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position_transformed = track_info.get('position_transformed')
                    if position_transformed is not None:
                        reshape_point = np.array(position_transformed).reshape(-1, 1, 2).astype(np.float32)
                        position_graphed = cv2.perspectiveTransform(reshape_point, graph_transformer)
                        position_graphed = position_graphed.squeeze().tolist()
                        tracks[object][frame_num][track_id]['position_graphed'] = position_graphed
                    else:
                        tracks[object][frame_num][track_id]['position_graphed'] = None

    #Draw position into ghraph
    
    def overlay_football_field_on_video_frames(self, video_frames, field_image_path):
    
        # Verifica el tamaño del primer frame del video para asegurarte de que las coordenadas sean válidas
        frame_height, frame_width = video_frames[0].shape[:2]
        
        # Definir las coordenadas de los puntos destino en el frame
        dst_points = [
            [1500, 10],    # Top-left
            [1900, 10],    # Top-right
            [1900, 270],   # Bottom-right
            [1500, 270]    # Bottom-left
        ]

        # Convertir a numpy array
        dst_points = np.array(dst_points, dtype=np.float32)

        # Leer la imagen del campo de fútbol
        football_field = cv2.imread(field_image_path)

        # Obtener las dimensiones de la imagen del campo de fútbol
        (h, w) = football_field.shape[:2]

        # Definir las coordenadas de los puntos origen (esquinas de la imagen)
        src_points = np.array([
            [0, 0],       # Top-left
            [w - 1, 0],   # Top-right
            [w - 1, h - 1],  # Bottom-right
            [0, h - 1]    # Bottom-left
        ], dtype=np.float32)

        # Calcular la transformación de perspectiva
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Procesar cada frame del video
        for i in range(len(video_frames)):
            frame = video_frames[i]

            # Aplicar la transformación de perspectiva a la imagen del campo de fútbol
            warped_field = cv2.warpPerspective(football_field, matrix, (frame_width, frame_height))

            # Crear una máscara para la imagen del campo de fútbol
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_points.astype(np.int32), (255, 255, 255))

            # Invertir la máscara para el área fuera del campo de fútbol
            mask_inv = cv2.bitwise_not(mask)

            # Eliminar el área del campo de fútbol del frame original
            frame_bg = cv2.bitwise_and(frame, mask_inv)

            # Superponer la imagen del campo de fútbol en el frame
            combined_frame = cv2.add(frame_bg, warped_field)

            # Reemplazar el frame en la lista de frames
            video_frames[i] = combined_frame

        return video_frames
    
    def draw_position_in_graph(self, video_frames, tracks, field_image_path):
        """Dibuja un círculo en la imagen en la ubicación especificada por 'position' con el 'color' dado."""
        self.overlay_football_field_on_video_frames(video_frames, field_image_path)
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Crear una copia del frame actual
            for player_id, track in tracks['players'][frame_num].items():
                graph_position = track['position_graphed']
                color = track['team_color']
                color = tuple(int(c) for c in np.clip(color, 0, 255))
                
                # Comprobar si la posición graficada no es None
                if graph_position is not None:
                    center = (int(graph_position[0]), int(graph_position[1]))
                    radius = 8
                    # Dibuja el círculo principal
                    cv2.circle(frame, center, radius, color, thickness=-1)
                    # Dibuja el contorno delgado en negro
                    cv2.circle(frame, center, radius, (0, 0, 0), thickness=1)

            for _, track in tracks['ball'][frame_num].items():
                graph_position = track['position_graphed']
                    
                # Comprobar si la posición graficada no es None
                if graph_position is not None:
                    center = (int(graph_position[0]), int(graph_position[1]))
                    radius = 5
                    # Dibuja el círculo principal
                    cv2.circle(frame, center, radius, (0, 0, 255), thickness=-1)
                    # Dibuja el contorno delgado en negro
                    cv2.circle(frame, center, radius, (0, 0, 0), thickness=1)


            output_video_frames.append(frame)
            
        return output_video_frames