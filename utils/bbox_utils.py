def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    
    return int((x1+x2)/2) , int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0] #diferencia de unidades entre x1 y x2

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 #hipotenusa entre cordenadas

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1] #distancia entre puntos

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2) #coordenadas de abajo y al centro

def distance_frame(tracks, frame, a, b):
    point_a = tracks['players'][frame][a]['position_adjusted']
    point_b = tracks['players'][frame][b]['position_adjusted']
    distance_a_b = measure_distance(point_a,point_b)
    return distance_a_b