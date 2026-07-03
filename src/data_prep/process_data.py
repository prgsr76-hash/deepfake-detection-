import cv2
import os
import json
import csv

max_frames = 30
splits = ['train', 'val', 'test']
categories = {
    'original': 'real',
    'Deepfakes': 'DeepFakes',
    'FaceSwap': 'FaceSwap'
}

csv_file = open('master.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_path', 'label', 'split', 'manipulation'])

def extract_and_log(video_path, output_dir, label, split, manip_name):
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        if len(existing_files) == max_frames:
            for count in range(max_frames):
                img_path = os.path.join(output_dir, f"{count:04d}.jpg")
                csv_writer.writerow([img_path, label, split, manip_name])
            return 

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return 
        
    indices = set([int(i * total_frames / max_frames) for i in range(max_frames)])
    count = 0
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in indices:
            img_name = f"{count:04d}.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            
            csv_writer.writerow([img_path, label, split, manip_name])
            count += 1
            if count >= max_frames:
                break
        frame_num += 1
    cap.release()

for split in splits:
    print(f"Processing {split} split...")
    
    json_path = f'{split}.json'
    if not os.path.exists(json_path):
        continue
        
    with open(json_path, 'r') as f:
        split_data = json.load(f)
        
    for orig_cat, new_cat in categories.items():
        label = 0 if new_cat == 'real' else 1
        print(f"  Checking {orig_cat} -> {new_cat}...")
        
        for pair in split_data:
            if orig_cat == 'original':
                video_names = [f"{pair[0]}.mp4", f"{pair[1]}.mp4"]
            else:
                video_names = [f"{pair[0]}_{pair[1]}.mp4", f"{pair[1]}_{pair[0]}.mp4"]
            
            for vid_name in video_names:
                video_path = f"data/videos/{orig_cat}/{vid_name}"
                
                if os.path.exists(video_path):
                    vid_id = vid_name.replace('.mp4', '')
                    output_dir = f"data/frames/{new_cat}/{split}/{vid_id}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    extract_and_log(video_path, output_dir, label, split, new_cat)

csv_file.close()
print("\nExtraction complete!")