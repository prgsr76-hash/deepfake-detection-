import pandas as pd
import os
from perturb import apply_perturbation

id_test = pd.read_csv('id_test.csv')
print(f"Applying perturbations to {len(id_test)} images...")

perturbation_types = ['jpeg', 'blur', 'contrast', 'saturation', 'noise']
output_root = 'id_test_perturbed'

for ptype in perturbation_types:
    out_dir = os.path.join(output_root, ptype)
    os.makedirs(out_dir, exist_ok=True)
    out_rows = []

    print(f"\nProcessing perturbation: {ptype}")
    for idx, row in id_test.iterrows():
        try:
            perturbed_img = apply_perturbation(row['image_path'], ptype, seed=idx)
            out_path = os.path.join(out_dir, f"{idx:06d}.jpg")
            perturbed_img.save(out_path)
            out_rows.append({
                'image_path': out_path,
                'label': row['label'],
                'video_id': row['video_id'],
                'manipulation': row['manipulation'],
                'orig_path': row['image_path']
            })
        except Exception as e:
            print(f"Failed on {row['image_path']}: {e}")

        if idx % 1000 == 0:
            print(f"  {ptype}: {idx}/{len(id_test)} done")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(os.path.join(output_root, f"{ptype}.csv"), index=False)
    print(f"Saved {len(out_df)} images for {ptype}")

print("\nAll perturbations complete.")
