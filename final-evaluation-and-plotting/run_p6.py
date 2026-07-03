import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from fvcore.nn import FlopCountAnalysis
from torchvision import transforms
from PIL import Image

from mesonet import MesoNet4
from efficientnet_model import EfficientNetB0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading models...")
    mesonet = MesoNet4().to(device)
    if os.path.exists('mesonet4/mesonet4_best.pth'):
        mesonet.load_state_dict(torch.load('mesonet4/mesonet4_best.pth', map_location=device))
        print("Loaded mesonet4_best.pth")
    mesonet.eval()

    efficientnet = EfficientNetB0().to(device)
    eff_pth = 'efficientnet_b0/efficientnet_b0_best.pth'
    if os.path.exists(eff_pth):
        efficientnet.load_state_dict(torch.load(eff_pth, map_location=device))
        print("Loaded efficientnet_b0_best.pth")
    elif os.path.exists('efficientnet_b0_best.pth'):
        efficientnet.load_state_dict(torch.load('efficientnet_b0_best.pth', map_location=device))
        print("Loaded efficientnet_b0_best.pth")
    else:
        print("Warning: Could not find EfficientNet weights! Please ensure you extracted efficientnet_b0.pth.zip")
    efficientnet.eval()

    # Transforms
    transform_meso = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    transform_eff = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def evaluate_on_csv(model, transform, csv_path):
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if 'split' in df.columns:
            df = df[df['split'] == 'test']
        preds = []
        targets = []
        for _, row in df.iterrows():
            path_col = 'face_path' if 'face_path' in row else 'image_path'
            if path_col not in row:
                path_col = df.columns[0] # guess first column if missing
            img_path = str(row[path_col])
            
            # Intelligent path resolution
            possible_paths = [
                img_path,
                os.path.join('id_test_perturbed', 'id_test_perturbed', img_path),
                os.path.join(os.path.dirname(csv_path), img_path)
            ]
            
            found_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    found_path = p
                    break
                    
            if found_path is None:
                continue
                
            img = Image.open(found_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(img_t).item()
            preds.append(score)
            targets.append(row['label'])
            
        if len(targets) == 0: 
            return None
        return targets, preds

    # Task 1 & 2
    base_dir = os.path.join('id_test_perturbed', 'id_test_perturbed')
    perturbations = ['clean', 'jpeg', 'blur', 'contrast', 'saturation', 'noise']
    results = {'MesoNet4': [], 'EfficientNet-B0': []}

    clean_csv = os.path.join(base_dir, 'clean.csv')
    if not os.path.exists(clean_csv) and os.path.exists('id_test.csv'):
        clean_csv = 'id_test.csv'

    print("\n--- Task 1: ROC Curve on ID Test Set (Clean) ---")
    clean_m = evaluate_on_csv(mesonet, transform_meso, clean_csv)
    clean_e = evaluate_on_csv(efficientnet, transform_eff, clean_csv)

    if clean_m and clean_e:
        fpr_m, tpr_m, _ = roc_curve(clean_m[0], clean_m[1])
        fpr_e, tpr_e, _ = roc_curve(clean_e[0], clean_e[1])
        auc_m_clean = roc_auc_score(clean_m[0], clean_m[1])
        auc_e_clean = roc_auc_score(clean_e[0], clean_e[1])
        
        plt.figure()
        plt.plot(fpr_m, tpr_m, label=f"MesoNet4 (AUC: {auc_m_clean:.3f})")
        plt.plot(fpr_e, tpr_e, label=f"EfficientNet-B0 (AUC: {auc_e_clean:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Task 1: ROC Curve on Clean ID Test Set')
        plt.legend()
        plt.savefig('task1_roc_curve.png')
        print("Saved task1_roc_curve.png")
    else:
        print("Could not find clean.csv to generate Task 1 plot.")

    print("\n--- Task 2: Perturbation vs AUC ---")
    for perturb in perturbations:
        csv_path = os.path.join(base_dir, f"{perturb}.csv")
        if perturb == 'clean':
            csv_path = clean_csv
            
        print(f"Evaluating {perturb}...")
        m_res = evaluate_on_csv(mesonet, transform_meso, csv_path)
        e_res = evaluate_on_csv(efficientnet, transform_eff, csv_path)
        
        results['MesoNet4'].append(roc_auc_score(m_res[0], m_res[1]) if m_res else None)
        results['EfficientNet-B0'].append(roc_auc_score(e_res[0], e_res[1]) if e_res else None)

    plt.figure()
    valid_perts = [p for i, p in enumerate(perturbations) if results['MesoNet4'][i] is not None]
    m_aucs = [a for a in results['MesoNet4'] if a is not None]
    e_aucs = [a for a in results['EfficientNet-B0'] if a is not None]

    if valid_perts:
        plt.plot(valid_perts, m_aucs, marker='o', label='MesoNet4')
        plt.plot(valid_perts, e_aucs, marker='x', label='EfficientNet-B0')
        plt.xlabel('Perturbation Type')
        plt.ylabel('AUC')
        plt.title('Task 2: Perturbation vs AUC')
        plt.legend()
        plt.savefig('task2_perturbation.png')
        print("Saved task2_perturbation.png")
    else:
        print("No perturbation CSVs found to generate Task 2 plot.")

    print("\n--- Task 3 & 4: FLOPs & Params ---")
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # fvcore might print warnings, capturing them is fine
    meso_flops = FlopCountAnalysis(mesonet, dummy_input).total()
    eff_flops = FlopCountAnalysis(efficientnet, dummy_input).total()

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    meso_params = count_params(mesonet)
    eff_params = count_params(efficientnet)

    id_auc_m = results['MesoNet4'][0] if len(results['MesoNet4']) > 0 and results['MesoNet4'][0] is not None else 0
    id_auc_e = results['EfficientNet-B0'][0] if len(results['EfficientNet-B0']) > 0 and results['EfficientNet-B0'][0] is not None else 0

    plt.figure()
    plt.scatter([meso_flops], [id_auc_m], label='MesoNet4', color='blue', s=100)
    plt.scatter([eff_flops], [id_auc_e], label='EfficientNet-B0', color='orange', s=100)
    plt.xlabel('FLOPs')
    plt.ylabel('ID Test AUC (Clean)')
    plt.title('Task 3: FLOPs vs AUC')
    plt.legend()
    plt.savefig('task3_flops.png')
    print("Saved task3_flops.png")

    plt.figure()
    plt.scatter([meso_params], [id_auc_m], label='MesoNet4', color='blue', s=100)
    plt.scatter([eff_params], [id_auc_e], label='EfficientNet-B0', color='orange', s=100)
    plt.xlabel('Parameters')
    plt.ylabel('ID Test AUC (Clean)')
    plt.title('Task 4: Params vs AUC')
    plt.legend()
    plt.savefig('task4_params.png')
    print("Saved task4_params.png")

    print("\n--- Task 5: Inference Time ---")
    def measure_inference_time(model, transform, n=100):
        # Create a dummy image instead of loading from disk to measure raw model throughput
        img_t = torch.randn(1, 3, 256, 256).to(device)
        for _ in range(10): # warmup
            with torch.no_grad():
                _ = model(img_t)
        times = []
        for _ in range(n):
            start = time.time()
            with torch.no_grad():
                _ = model(img_t)
            if device.type == 'cuda': torch.cuda.synchronize()
            times.append(time.time() - start)
        return np.mean(times) * 1000

    meso_inf = measure_inference_time(mesonet, transform_meso)
    eff_inf = measure_inference_time(efficientnet, transform_eff)

    plt.figure()
    plt.scatter([meso_inf], [id_auc_m], label='MesoNet4', color='blue', s=100)
    plt.scatter([eff_inf], [id_auc_e], label='EfficientNet-B0', color='orange', s=100)
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('ID Test AUC (Clean)')
    plt.title('Task 5: Inference Time vs AUC')
    plt.legend()
    plt.savefig('task5_inference_time.png')
    print("Saved task5_inference_time.png")

    print("\n--- Task 6: Summary Table ---")
    def get_standard_auc(csv_path):
        if not os.path.exists(csv_path): return "?"
        df = pd.read_csv(csv_path)
        return f"{roc_auc_score(df['true_label'], df['fake_score']):.3f}"

    std_auc_m = get_standard_auc('mesonet4/mesonet_predictions.csv')
    std_auc_e = get_standard_auc('efficientnet_b0/efficientnet_predictions.csv')

    id_m_str = f"{id_auc_m:.3f}" if id_auc_m else "?"
    id_e_str = f"{id_auc_e:.3f}" if id_auc_e else "?"

    table = f"""
# Deepfake Detection Benchmark - Final Report

| Model | Standard AUC | ID Test AUC | FLOPs | Params | Inference (ms) |
|-------|--------------|-------------|-------|--------|----------------|
| MesoNet-4 | {std_auc_m} | {id_m_str} | {meso_flops:,} | {meso_params:,} | {meso_inf:.2f} |
| EfficientNet-B0 | {std_auc_e} | {id_e_str} | {eff_flops:,} | {eff_params:,} | {eff_inf:.2f} |
"""
    print(table)
    with open('summary_table.md', 'w') as f:
        f.write(table)
    print("Saved summary_table.md")

if __name__ == "__main__":
    main()
