import pandas as pd

master = pd.read_csv('master_faces.csv')
intersection = pd.read_csv('hard_intersection.csv')

def extract_video_id(path):
    parts = path.split('/')
    return f"{parts[2]}_{parts[-2]}"

master['video_id'] = master['image_path'].apply(extract_video_id)

# all frames from the 136 hard fake videos
fake_frames = master[master['video_id'].isin(intersection['video_id'])].copy()
print(f"Fake frames: {len(fake_frames)} from {fake_frames['video_id'].nunique()} videos")

# real videos from test split
real_videos_pool = master[(master['split'] == 'test') & (master['label'] == 0)].copy()
real_video_ids = real_videos_pool['video_id'].unique()

# sample equal number of real VIDEOS (not frames) to match fake video count
import random
random.seed(42)
n_fake_videos = fake_frames['video_id'].nunique()
sampled_real_ids = random.sample(list(real_video_ids), min(n_fake_videos, len(real_video_ids)))

real_frames = real_videos_pool[real_videos_pool['video_id'].isin(sampled_real_ids)].copy()
print(f"Real frames: {len(real_frames)} from {real_frames['video_id'].nunique()} videos")

# combine
id_test = pd.concat([fake_frames, real_frames], ignore_index=True)
id_test.to_csv('id_test.csv', index=False)

print(f"\nFinal ID test set: {len(id_test)} total frames")
print(f"  Fake: {len(fake_frames)} frames, {fake_frames['video_id'].nunique()} videos")
print(f"  Real: {len(real_frames)} frames, {real_frames['video_id'].nunique()} videos")
print("\nManipulation breakdown (fakes):")
print(fake_frames['manipulation'].value_counts())
