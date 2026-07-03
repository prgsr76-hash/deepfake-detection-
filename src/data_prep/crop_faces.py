import cv2
import os
import csv

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

base_frames_dir = 'data/frames'
base_faces_dir = 'data/faces'

print("Phase 2A: Cropping new fake faces from overnight extraction...")

for manip in os.listdir(base_frames_dir):
    manip_path = os.path.join(base_frames_dir, manip)
    if not os.path.isdir(manip_path): continue
    
    for split in os.listdir(manip_path):
        split_path = os.path.join(manip_path, split)
        if not os.path.isdir(split_path): continue
            
        for vid_id in os.listdir(split_path):
            vid_path = os.path.join(split_path, vid_id)
            if not os.path.isdir(vid_path): continue
                
            output_dir = os.path.join(base_faces_dir, manip, split, vid_id)
            os.makedirs(output_dir, exist_ok=True)
            
            for frame_file in os.listdir(vid_path):
                if not frame_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                    
                input_img_path = os.path.join(vid_path, frame_file)
                output_img_path = os.path.join(output_dir, frame_file)
                
                if os.path.exists(output_img_path):
                    continue 
                    
                img = cv2.imread(input_img_path)
                if img is None: continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    offset_x, offset_y = int(w * 0.15), int(h * 0.15)
                    
                    img_h, img_w, _ = img.shape
                    x1, y1 = max(0, x - offset_x), max(0, y - offset_y)
                    x2, y2 = min(img_w, x + w + offset_x), min(img_h, y + h + offset_y)
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        resized = cv2.resize(crop, (256, 256))
                        cv2.imwrite(output_img_path, resized)

print("Phase 2B: Building master_faces.csv directly from the faces folder...")

total_logged = 0
with open('master_faces.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_path', 'label', 'split', 'manipulation'])
    
    for manip in os.listdir(base_faces_dir):
        manip_path = os.path.join(base_faces_dir, manip)
        if not os.path.isdir(manip_path): continue
        
        label = 0 if manip.lower() == 'real' else 1
        
        for split in os.listdir(manip_path):
            split_path = os.path.join(manip_path, split)
            if not os.path.isdir(split_path): continue
                
            for vid_id in os.listdir(split_path):
                vid_path = os.path.join(split_path, vid_id)
                if not os.path.isdir(vid_path): continue
                    
                for face_file in os.listdir(vid_path):
                    if face_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        face_img_path = os.path.join(vid_path, face_file)
                        
                        face_img_path = face_img_path.replace('\\', '/')
                        
                        csv_writer.writerow([face_img_path, label, split, manip])
                        total_logged += 1

print(f"\nMission Accomplished! Total valid faces ready for PyTorch: {total_logged}")