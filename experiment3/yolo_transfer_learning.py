from scipy.signal import freqz_zpk
from ultralytics import YOLO
import yaml
import torch
import os
import json
import numpy as np
import pathlib
import pandas as pd


# ----------------------- Configuration -----------------------
BASE_MODEL_PATH = '../experiment1/yolo_base_model/model_training_run/training/weights/best.pt'
DATASET_PATH = '../dataset_b/split_dataset_2000'
DATASET_TEST_PATH = '../dataset_b/dataset_test'
YAML_PATH = 'transferLearning.yaml'
OUTPUT_BASE_DIR = 'yolo_transfer_learning_2000'
NUM_CLASSES = 4
CLASS_NAMES = ['sphere', 'cone', 'cylinder', 'cube']
IMG_SIZE = 640
EPOCHS = 10
PATIENCE = 5
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'



os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
training_project_dir = os.path.join(OUTPUT_BASE_DIR, 'model_training_run')
evaluation_project_dir = os.path.join(OUTPUT_BASE_DIR, 'model_evaluation_run')


def check_dataset_structure(base_path):
    expected_dirs = [
        ('train', 'images'), ('train', 'labels'),
        ('val', 'images'), ('val', 'labels'),
    ]
    for split, typ in expected_dirs:
        path = os.path.join(base_path, split, typ)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Faltando: {path}")
    print("Estrutura do dataset_inicial verificada.")

check_dataset_structure(DATASET_PATH)


# --- Function to create YAML ---
def create_yaml(path):
    data = {
        'train': os.path.join(DATASET_PATH, 'train', 'images'),
        'val': os.path.join(DATASET_PATH, 'val', 'images'),
        'test': os.path.join(DATASET_TEST_PATH, 'images'),
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES
    }
    with open(path, 'w') as f:
        yaml.dump(data, f)
    print(f"Arquivo YAML criado em {path}")
    return path

yaml_file = create_yaml(YAML_PATH)


# ------------------- Load Pre-trained Model --------------------
model = YOLO(BASE_MODEL_PATH)


# ------------------------- Model Training -----------------------------

train_results = model.train(
    data=yaml_file,
    epochs=EPOCHS,
    freeze = 10,
    imgsz=IMG_SIZE,
    project=training_project_dir,
    device=DEVICE,
    verbose=True
)

# -------------------------- Final Evaluation ---------------------------
test_metrics = model.val(
    data=yaml_file,
    split='test',
    project=evaluation_project_dir,
    device=DEVICE
)


sample_images_dir = os.path.join(DATASET_TEST_PATH, 'images')
sample_images = sorted([f for f in os.listdir(sample_images_dir) if f.endswith('.jpg') or f.endswith('.png')])

run_summary_path = os.path.join(OUTPUT_BASE_DIR, 'run_summary.json')

summary_data = {
    'data_source_config': {
        'pre_split_dataset_path': os.path.abspath(DATASET_PATH),
        'total_images_in_split_folders': 'N/A',
        'training_images_count': 'N/A',
        'validation_images_count': 'N/A',
        'test_images_count': len(sample_images),
        'train_percentage': 'N/A',
        'validation_percentage': 'N/A',
        'test_percentage': 'N/A',
    },
    'hyperparameters_used': {
        'lr0': 0.005,
        'batch': 16,
        'label_smoothing': 0.0,
        'weight_decay': 0.00025,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'imgsz': IMG_SIZE
    },
    'training_duration_seconds': "N/A",
    'trained_model_best_weights_path': model.ckpt_path if hasattr(model, 'ckpt_path') else 'Unknown',
    'yolo_training_output_dir': training_project_dir,
    'validation_metrics_from_training_run': train_results.results_dict if train_results else "Not available or training failed/skipped",
    'test_set_metrics_on_trained_model': test_metrics.results_dict if test_metrics else "Skipped or Failed"
}

def convert_types_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, pathlib.Path): return str(obj)
    elif pd.isna(obj) or obj is None: return None
    return obj

with open(run_summary_path, 'w') as f:
    json.dump(convert_types_for_json(summary_data), f, indent=4)

print(f"\nRun summary saved to: {run_summary_path}")
