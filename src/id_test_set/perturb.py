from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import os

def apply_perturbation(img_path, perturb_type, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = Image.open(img_path).convert('RGB')

    if perturb_type == 'jpeg':
        quality = random.randint(30, 95)
        tmp_path = img_path + "_tmp.jpg"
        img.save(tmp_path, quality=quality)
        img = Image.open(tmp_path).convert('RGB')
        os.remove(tmp_path)

    elif perturb_type == 'blur':
        kernel = random.choice(range(10, 21, 2))
        img = img.filter(ImageFilter.GaussianBlur(radius=kernel / 2))

    elif perturb_type == 'contrast':
        factor = random.uniform(0.5, 2.5)
        img = ImageEnhance.Contrast(img).enhance(factor)

    elif perturb_type == 'saturation':
        factor = random.uniform(0.0, 2.5)
        img = ImageEnhance.Color(img).enhance(factor)

    elif perturb_type == 'noise':
        arr = np.array(img).astype(np.float32) / 255.0
        mean = random.uniform(-0.3, 0.3)
        var = random.uniform(0, 1)
        noise = np.random.normal(mean, var ** 0.5, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    else:
        raise ValueError(f"Unknown perturbation: {perturb_type}")

    return img


if __name__ == "__main__":
    import pandas as pd
    master = pd.read_csv('master_faces.csv')
    test_fakes = master[(master['split'] == 'test') & (master['label'] == 1)]
    test_img = test_fakes.iloc[0]['image_path']  # grab a real sample path automatically
    print(f"Using sample image: {test_img}")

    for ptype in ['jpeg', 'blur', 'contrast', 'saturation', 'noise']:
        out = apply_perturbation(test_img, ptype, seed=42)
        out.save(f"sample_{ptype}.jpg")
        print(f"Saved sample_{ptype}.jpg")
