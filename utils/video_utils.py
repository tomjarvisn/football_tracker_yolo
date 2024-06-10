import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # La variable ret indica si la lectura del fotograma fue exitosa o no. 
        frames.append(frame)
    return frames

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 'XVID' es un identificador de códec (algoritmo para comprimir o descomprimir videos)
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # output_video_frames[0].shape[1] representa el ancho del primer fotograma.
    # output_video_frames[0].shape[0] representa la altura del primer fotograma.
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()