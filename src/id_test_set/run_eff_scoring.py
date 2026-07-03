import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_model import EfficientNetB0

device = 'cpu'
model = EfficientNetB0()
model.load_state_dict(torch.load('efficientnet_b0.pth', map_location=device))
model.eval()
model.to(device)

eff_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def eff_inference_fn(model, img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = eff_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        return torch.sigmoid(logit).item()

def score_all(model, inference_fn, df):
    scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        s = inference_fn(model, row['image_path'])
        scores.append(s)
        if i % 500 == 0:
            print(f"Scored {i}/{len(df)}")
    df = df.copy()
    df['fake_score'] = scores
    return df

def video_level_scores(df):
    return df.groupby('video_id')['fake_score'].mean().reset_index()

master = pd.read_csv('master_faces.csv')

def extract_video_id(path):
    parts = path.split('/')
    return f"{parts[2]}_{parts[-2]}"

master['video_id'] = master['image_path'].apply(extract_video_id)
test_fakes = master[(master['split'] == 'test') & (master['label'] == 1)]

print(f"Scoring {len(test_fakes)} frames with EfficientNet...")

eff_scored = score_all(model, eff_inference_fn, test_fakes)
eff_scored.to_csv('eff_scored_frames.csv', index=False)

eff_video = video_level_scores(eff_scored)
eff_video.to_csv('eff_video_scores.csv', index=False)

print("\nDone. Hardest 10 videos for EfficientNet (lowest fake_score = most confusing):")
print(eff_video.sort_values('fake_score').head(10))run_eff_scoring.py
