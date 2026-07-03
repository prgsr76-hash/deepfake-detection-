import pandas as pd

# load both video-level score files
meso_video = pd.read_csv('meso_video_scores.csv')
eff_video = pd.read_csv('eff_video_scores.csv')

print(f"MesoNet scored {len(meso_video)} videos")
print(f"EfficientNet scored {len(eff_video)} videos")

def bottom_two_thirds(video_df, score_col='fake_score'):
    n_keep = int(len(video_df) * (2/3))
    return set(video_df.nsmallest(n_keep, score_col)['video_id'])

meso_hard = bottom_two_thirds(meso_video)
eff_hard = bottom_two_thirds(eff_video)

print(f"MesoNet hard set: {len(meso_hard)} videos")
print(f"EfficientNet hard set: {len(eff_hard)} videos")

hard_intersection = meso_hard & eff_hard
print(f"\nIntersection (fooled BOTH models): {len(hard_intersection)} videos")

# save the intersection list
intersection_df = pd.DataFrame({'video_id': list(hard_intersection)})
intersection_df.to_csv('hard_intersection.csv', index=False)
print("\nSaved hard_intersection.csv")

print("\nSample of intersection videos:")
print(intersection_df.head(10))
