# documentar codigo, explicar que hace cada parte
import cv2
from pathlib import Path

def extract_frames():
    actual_directory = Path(__file__).parent
    video_folder = actual_directory / "NITYMED"
    output_folder = actual_directory / "dataset_images"

    mapped_class = {
        "Microsleep": "microsueno",
        "Yawning": "bostezo",
        "Normal": "normal"
    }
    
    mapped_gender = {
        "Female": "mujeres",
        "Male": "hombres"
    }

    print(f"Searching for videos in {video_folder}...")

    video_path = list(video_folder.rglob("*.mp4")) + list(video_folder.rglob("*.avi"))

    if not video_path:
        print("No videos found in the specified folder.")
        return
    
    print(f"Found {len(video_path)} videos. Starting extraction...")

    frames_per_second = 1

    for video in video_path:
        class_destine = None
        gender_destine = "otros"
        
        for parent in video.parents:
            if parent.name in mapped_class:
                class_destine = mapped_class[parent.name]
            if parent.name in mapped_gender:
                gender_destine = mapped_gender[parent.name]
        
        if not class_destine:
            print(f"Warning: No class mapping found for video {video.name}. Skipping.")
            continue

        folder_destine = output_folder / class_destine / gender_destine
        
        folder_destine.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))

        if fps_original == 0:
            print(f"Warning: Could not determine FPS for video {video.name}. Skipping.")
            continue

        salto_frames = fps_original // frames_per_second
        frame_count = 0
        saved_frames = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % salto_frames == 0:
                name_frame = f"{video.stem}_f{saved_frames:04d}.jpg"
                path_saved = folder_destine / name_frame

                cv2.imwrite(str(path_saved), frame)
                saved_frames += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_frames} frames from video {video.name}.")

if __name__ == "__main__":
    extract_frames()