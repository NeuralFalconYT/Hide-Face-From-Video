import cv2
import time
from matplotlib import text
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from utils import mask_overlay

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks]
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    return annotated_image

def mediapipe_config():
    model_path = "face_landmarker.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    return landmarker

landmarker = None
def reset_landmarker():
    global landmarker
    try:
        if landmarker:
            landmarker.close()  
    except:
        pass
    landmarker = mediapipe_config()  


def face_point(results, frame):
    ih, iw, ic = frame.shape
    faces = []
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            face = []
            for id, lm in enumerate(face_landmarks):
                x, y = int(lm.x * iw), int(lm.y * ih)
                face.append([id, x, y])
            faces.append(face)
    return faces

def letterbox(image, target_width, target_height):
    """Resize image keeping aspect ratio, pad with black to fit target size."""
    ih, iw = image.shape[:2]
    scale = min(target_width / iw, target_height / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - nw) // 2
    y_offset = (target_height - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas



import subprocess
import os
import shutil
import os, shutil, subprocess
import uuid 
def replace_audio_with_ffmpeg(video_path, audio_path):

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs("./save_video/", exist_ok=True)
    output_path = f"./save_video/{base_name}_.mp4"

    gpu = False
    if gpu:
        print("CUDA GPU is available. Running on GPU.")
    else:
        print("No CUDA GPU found. Falling back to CPU.")

    video_codec = "h264_nvenc" if gpu else "libx264"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", video_codec,
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.abspath(output_path)



def add_audio(input_video, mask_video, save_video="final.mp4"):
    """
    Extract audio as WAV from input_video and add it to mask_video.
    If extraction fails, just copy mask_video to save_video.
    """
    try:
        os.makedirs("./temp", exist_ok=True)
        audio_file = os.path.abspath("./temp/temp_audio.wav")

        input_video = os.path.normpath(os.path.abspath(input_video))
        mask_video  = os.path.normpath(os.path.abspath(mask_video))
        save_video  = os.path.normpath(os.path.abspath(save_video))

        extract_cmd = [
            "ffmpeg", "-y", "-i", input_video, "-vn",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            audio_file, "-hide_banner", "-loglevel", "error"
        ]
        subprocess.run(extract_cmd, check=True,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
            raise Exception("No audio track extracted")

        gpu=False
        video_codec = "h264_nvenc" if gpu else "libx264"
        merge_cmd = [
            "ffmpeg",
            "-y",
            "-i", mask_video,
            "-i", audio_file,
            "-c:v", video_codec,
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            save_video
        ]
    
    
        subprocess.run(merge_cmd, check=True,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

        os.remove(audio_file)
        return True

    except Exception as e:
        print("Audio merge failed:", e)
        try:
            shutil.copy(mask_video, save_video)  
        except Exception as e2:
            print("Fallback copy failed:", e2)
            return False
        return False


def is_camera_source(source):
    if isinstance(source, int):
        return True
    if isinstance(source, str) and not os.path.isfile(source):
        
        try:
            idx = int(source)
            return True
        except ValueError:
            return False
    return False

def add_mask(upload_video,
             mask_name="Blue Mask",mask_up=10, mask_down=10,display=False):
    reset_landmarker()  
    output_video="./temp/mask.mp4"
    os.makedirs("./temp", exist_ok=True)
    cap = cv2.VideoCapture(upload_video)
    if not cap.isOpened():
        print("Cannot access video file")
        exit()
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if input_fps <= 0 or input_fps > 120: 
        input_fps = 25  

    OUTPUT_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    OUTPUT_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, input_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))


    frame_count = 0
    fps = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video processing complete.")
            break
        # Flip frame
        # frame = cv2.flip(frame, 1)
        raw_frame=frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Create the mesh visualization
        visualized_image = draw_landmarks_on_image(frame_rgb, results)
        visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)
        
        # Create the mask overlay image
        faces = face_point(results, frame)
        if len(faces) > 0:
            masked_frame = mask_overlay(frame, faces, mask_up, mask_down, mask_name)
        else:
            masked_frame = frame
        out.write(masked_frame)
        if display:
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()

            fps_text = f"FPS: {fps:.2f}"
            
            SCREEN_W, SCREEN_H =  480, 270 
            cv2.putText(raw_frame, fps_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # cv2.putText(middle, fps_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(right, fps_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            left=letterbox(raw_frame, SCREEN_W, SCREEN_H)
            middle = letterbox(visualized_image, SCREEN_W, SCREEN_H)
            right = letterbox(masked_frame, SCREEN_W, SCREEN_H)
            

            combined_image = np.hstack((left,middle, right))



            cv2.imshow("Background Preview", combined_image)
            cv2.imshow("OBS", masked_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("Releasing resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    random_str = str(uuid.uuid4())[:5]  
    if is_camera_source(upload_video):
        print("Using Camera Index:", upload_video)
        return None, None

    save_video_path="./temp/"+os.path.basename(upload_video).split('.')[0] + "_" +mask_name.replace(" ","_") + "_" + random_str+".mp4"
    sucess=add_audio(upload_video,output_video, save_video_path)
    if sucess:
        print(f"Masked video saved to {save_video_path}")
        return save_video_path,save_video_path
    else:
        print("Failed to save masked video.")
        return output_video,output_video

# mask_names=["Front Man Mask", "Guards Mask", "Red Mask", "Blue Mask"]
# add_mask(0,mask_name=mask_names[0],mask_up=10, mask_down=10,display=True)
