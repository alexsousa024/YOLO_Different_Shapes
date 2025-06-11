import os
import shutil
import yaml
from ultralytics import YOLO
import torch
import time
import json
import pandas as pd
import numpy as np
import pathlib # For isinstance check

# --- Configuration ---
# Path to your pre-split dataset_inicial
pre_split_dataset_base_path = 'split_dataset' # Main directory for pre-split data
output_base_dir = 'yolo_complex_model'  # Main output directory
img_ext = '.png' # Ensure this matches your image files in split_dataset

# Splitting configuration is no longer needed as data is pre-split
# test_set_size = 0.2
# validation_from_remaining_size = 0.25
# random_state = 42

# Model & Training
base_model_path = 'yolov8n.pt'
num_classes = 3 # As per your provided snippet
class_names = ['sphere', 'cone', 'cylinder'] # As per your provided snippet
img_size = 640

# Predefined Hyperparameters (using common defaults or your chosen values)
predefined_hyperparameters = {
    'lr0': 0.005,
    'batch': 16,
    'label_smoothing': 0.0,
    'weight_decay': 0.00025,
}
training_epochs = 10  # Epochs for the training run
training_patience = 5 # Patience for early stopping

# --- Setup Device ---
if torch.backends.mps.is_available():
    device = 'mps'
    print("MPS is available! Using MPS for training.")
elif torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available! Using CUDA for training.")
else:
    device = 'cpu'
    print("MPS/CUDA not available. Falling back to CPU. This will be slow.")

# --- Prepare Output Directories ---
os.makedirs(output_base_dir, exist_ok=True)
run_summary_path = os.path.join(output_base_dir, 'run_summary.json')
training_project_dir = os.path.join(output_base_dir, 'model_training_run')
evaluation_project_dir = os.path.join(output_base_dir, 'model_evaluation_run')

# --- Data Preparation from Pre-split Folders ---
print("\n--- Preparing Data from Pre-split Folders ---")
print(f"Using pre-split data from: {os.path.abspath(pre_split_dataset_base_path)}")

# Define source paths for images and labels within the pre_split_dataset_base_path
train_img_source_dir = os.path.join(pre_split_dataset_base_path, 'train', 'images')
train_lbl_source_dir = os.path.join(pre_split_dataset_base_path, 'train', 'labels')
val_img_source_dir = os.path.join(pre_split_dataset_base_path, 'val', 'images') # Using 'val' as per your screenshot
val_lbl_source_dir = os.path.join(pre_split_dataset_base_path, 'val', 'labels')
test_img_source_dir = os.path.join(pre_split_dataset_base_path, 'test', 'images')
test_lbl_source_dir = os.path.join(pre_split_dataset_base_path, 'test', 'labels')

# Check if source directories exist
source_dirs_to_check = {
    "Train Images": train_img_source_dir, "Train Labels": train_lbl_source_dir,
    "Validation Images": val_img_source_dir, "Validation Labels": val_lbl_source_dir,
    "Test Images": test_img_source_dir, "Test Labels": test_lbl_source_dir,
}
all_sources_exist = True
for name, path in source_dirs_to_check.items():
    if not os.path.isdir(path):
        print(f"ERROR: {name} directory not found: {path}")
        all_sources_exist = False
if not all_sources_exist:
    print("One or more source directories are missing. Please check your 'split_dataset' structure. Exiting.")
    exit()

# Get lists of image filenames (without path) from the respective 'images' directories
try:
    train_images = sorted([f for f in os.listdir(train_img_source_dir) if f.endswith(img_ext)])
    val_images = sorted([f for f in os.listdir(val_img_source_dir) if f.endswith(img_ext)])
    test_images = sorted([f for f in os.listdir(test_img_source_dir) if f.endswith(img_ext)])
except FileNotFoundError as e:
    print(f"Error listing files from source directories: {e}. Exiting.")
    exit()

if not train_images: print(f"Warning: No training images found in {train_img_source_dir} with extension {img_ext}.")
if not val_images: print(f"Warning: No validation images found in {val_img_source_dir} with extension {img_ext}.")
if not test_images: print(f"Warning: No test images found in {test_img_source_dir} with extension {img_ext}.")

total_images_in_split_folders = len(train_images) + len(val_images) + len(test_images)
if total_images_in_split_folders == 0:
    print(f"No images found across train, val, test in '{pre_split_dataset_base_path}'. Exiting.")
    exit()

print(f"Total images found in pre-split folders: {total_images_in_split_folders}")
print(f"Training images: {len(train_images)} ({(len(train_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%)")
print(f"Validation images: {len(val_images)} ({(len(val_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%)")
print(f"Test images: {len(test_images)} ({(len(test_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%)")


# --- Function to create YAML ---
def create_data_yaml(file_path, train_img_dir, val_img_dir, nc, names, test_img_dir=None):
    yaml_content = {
        'train': os.path.abspath(train_img_dir),
        'val': os.path.abspath(val_img_dir),
        'nc': nc,
        'names': names
    }
    if test_img_dir:
        yaml_content['test'] = os.path.abspath(test_img_dir) # This will point to the copied test images
    with open(file_path, 'w') as f:
        yaml.dump(yaml_content, f)
    print(f"Created YAML: {file_path}")

# --- Function to copy files for a split ---
# base_img_path and base_lbl_path will now be the source paths from pre_split_dataset_base_path
def setup_split_files(image_list, source_img_dir, source_lbl_dir, target_img_dir, target_lbl_dir):
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)
    copied_count = 0
    missing_labels_count = 0
    print(f"  Copying from {source_img_dir} and {source_lbl_dir}")
    for img_file_basename in image_list: # image_list now contains just basenames
        base_name_no_ext = os.path.splitext(img_file_basename)[0]
        label_file_basename = base_name_no_ext + '.txt'

        src_img_path = os.path.join(source_img_dir, img_file_basename)
        src_lbl_path = os.path.join(source_lbl_dir, label_file_basename)

        dest_img_path = os.path.join(target_img_dir, img_file_basename)
        dest_lbl_path = os.path.join(target_lbl_dir, label_file_basename)

        try:
            if not os.path.exists(src_img_path):
                print(f"    Warning: Source image file not found: {src_img_path}. Skipping.")
                continue
            shutil.copy(src_img_path, dest_img_path)

            if os.path.exists(src_lbl_path):
                shutil.copy(src_lbl_path, dest_lbl_path)
            else:
                print(f"    Warning: Label file not found for {img_file_basename} at {src_lbl_path}. Image copied, label skipped.")
                missing_labels_count +=1
            copied_count +=1
        except Exception as e: # Changed FileNotFoundError to broader Exception for robustness
            print(f"    Warning: Could not copy file for {img_file_basename} or its label: {e}")
    print(f"  Copied {copied_count} images to {target_img_dir}.")
    if missing_labels_count > 0:
         print(f"  Warning: {missing_labels_count} images were copied without their corresponding label files from source.")


# --- Setup Data Directories for Training (Train/Val) in a temporary run location ---
run_data_dir = os.path.join(output_base_dir, 'run_data_setup') # Holds data for this specific run
train_img_dir_setup = os.path.join(run_data_dir, 'images/train')
train_lbl_dir_setup = os.path.join(run_data_dir, 'labels/train')
val_img_dir_setup = os.path.join(run_data_dir, 'images/val')
val_lbl_dir_setup = os.path.join(run_data_dir, 'labels/val')

print(f"\n--- Setting up temporary data directories in '{run_data_dir}' ---")
if os.path.exists(run_data_dir): shutil.rmtree(run_data_dir) # Clean previous run data

print("Setting up training files (copying from pre-split source)...")
if train_images:
    setup_split_files(train_images, train_img_source_dir, train_lbl_source_dir, train_img_dir_setup, train_lbl_dir_setup)
else:
    os.makedirs(train_img_dir_setup, exist_ok=True) # Create empty dir if no files
    os.makedirs(train_lbl_dir_setup, exist_ok=True)

print("Setting up validation files (copying from pre-split source)...")
if val_images:
    setup_split_files(val_images, val_img_source_dir, val_lbl_source_dir, val_img_dir_setup, val_lbl_dir_setup)
else:
    os.makedirs(val_img_dir_setup, exist_ok=True)
    os.makedirs(val_lbl_dir_setup, exist_ok=True)


train_val_yaml_path = os.path.join(run_data_dir, 'data_train_val.yaml')
create_data_yaml(train_val_yaml_path, train_img_dir_setup, val_img_dir_setup, num_classes, class_names)

# --- Model Training ---
print("\n--- Training the Model (single run with predefined hyperparameters) ---")
model = YOLO(base_model_path)
training_run_name = 'training'

print(f"Using Hyperparameters: {predefined_hyperparameters}")
print(f"Training for {training_epochs} epochs with patience {training_patience}.")

train_start_time = time.time()
trained_model_best_weights_path = None
training_output_dir = None
training_results_object = None
train_time = 0

if not train_images or not val_images:
    print("WARNING: Training or validation set is empty (after attempting to load from pre-split). Cannot start training.")
else:
    try:
        train_args = {
            'data': train_val_yaml_path,
            'epochs': training_epochs,
            'patience': training_patience,
            'imgsz': img_size,
            'device': device,
            'project': training_project_dir,
            'name': training_run_name,
            'exist_ok': True,
            'verbose': True,
            **predefined_hyperparameters
        }
        train_args_clean = {k:v for k,v in train_args.items() if v is not None}

        training_results_object = model.train(**train_args_clean)
        train_time = time.time() - train_start_time

        print(f"Model training completed. Time: {train_time:.2f}s")
        if training_results_object and hasattr(training_results_object, 'save_dir'):
            training_output_dir = str(training_results_object.save_dir) # Ensure string
            trained_model_best_weights_path = os.path.join(training_output_dir, 'weights/best.pt')
            print(f"Best weights from training saved at: {trained_model_best_weights_path}")
        else:
            # Fallback for older ultralytics or if save_dir is not directly on results
            training_output_dir = os.path.join(training_project_dir, training_run_name)
            potential_weights_path = os.path.join(training_output_dir, 'weights/best.pt')
            if os.path.exists(potential_weights_path):
                trained_model_best_weights_path = potential_weights_path
                print(f"Training finished. Best weights found at: {trained_model_best_weights_path}")
            else:
                print("Training finished, but couldn't determine save directory or find best.pt automatically.")
                print(f"Please check the '{training_output_dir}' directory manually.")

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        if train_start_time > 0: train_time = time.time() - train_start_time


# --- Final Evaluation on Test Set ---
print("\n--- Evaluating the Trained Model on the Test Set ---")
test_metrics_results = {}
if trained_model_best_weights_path and os.path.exists(trained_model_best_weights_path):
    if not test_images:
        print("Test set is empty (no images found in pre-split test folder). Skipping test set evaluation.")
    else:
        test_data_setup_dir = os.path.join(run_data_dir, 'test_data_for_eval')
        test_img_dir_eval = os.path.join(test_data_setup_dir, 'images/test')
        test_lbl_dir_eval = os.path.join(test_data_setup_dir, 'labels/test')

        print(f"\nSetting up temporary test data directories in '{test_data_setup_dir}' ---")
        if os.path.exists(test_data_setup_dir): shutil.rmtree(test_data_setup_dir)
        print("Setting up test files for evaluation (copying from pre-split source)...")
        setup_split_files(test_images, test_img_source_dir, test_lbl_source_dir, test_img_dir_eval, test_lbl_dir_eval)

        test_yaml_path = os.path.join(test_data_setup_dir, 'data_test_only.yaml')
        create_data_yaml(test_yaml_path,
                         train_img_dir=test_img_dir_eval, # Dummy, model.val uses 'test' path
                         val_img_dir=test_img_dir_eval,   # Dummy
                         nc=num_classes, names=class_names,
                         test_img_dir=test_img_dir_eval)  # Actual test path

        model_for_evaluation = YOLO(trained_model_best_weights_path)
        print("Evaluating on the test set...")
        try:
            test_eval_metrics = model_for_evaluation.val(
                data=test_yaml_path,
                split='test',
                device=device,
                project=evaluation_project_dir,
                name='test_set_evaluation_results',
                exist_ok=True,
                verbose=False
            )
            test_metrics_results = {
                'precision': test_eval_metrics.box.mp if hasattr(test_eval_metrics.box, 'mp') else None,
                'recall': test_eval_metrics.box.mr if hasattr(test_eval_metrics.box, 'mr') else None,
                'mAP50': test_eval_metrics.box.map50 if hasattr(test_eval_metrics.box, 'map50') else None,
                'mAP50-95': test_eval_metrics.box.map if hasattr(test_eval_metrics.box, 'map') else None,
                'test_run_output_dir': str(test_eval_metrics.save_dir) if hasattr(test_eval_metrics, 'save_dir') else None
            }
            print("Test Set Metrics:")
            for k_test, v_test in test_metrics_results.items():
                if isinstance(v_test, float): print(f"  {k_test}: {v_test:.4f}")
                elif v_test is not None : print(f"  {k_test}: {str(v_test)}")
                else: print(f"  {k_test}: N/A")
        except Exception as e_eval:
            print(f"Error during test set evaluation: {e_eval}")
            import traceback
            traceback.print_exc()
            test_metrics_results = {"error": str(e_eval)}
else:
    print("Trained model weights not found or training failed/skipped. Skipping test set evaluation.")

# --- Save Summary ---
print("\n--- Saving Run Summary ---")

validation_metrics_from_training = {}
if training_results_object and hasattr(training_results_object, 'box'):
    val_box = training_results_object.box
    validation_metrics_from_training = {
        'val_mAP50': val_box.map50 if hasattr(val_box, 'map50') else None,
        'val_mAP50-95': val_box.map if hasattr(val_box, 'map') else None,
        'val_precision': val_box.mp if hasattr(val_box, 'mp') else None,
        'val_recall': val_box.mr if hasattr(val_box, 'mr') else None,
    }
elif training_output_dir and os.path.exists(os.path.join(training_output_dir, 'results.csv')):
    try:
        results_csv_path = os.path.join(training_output_dir, 'results.csv')
        df_results = pd.read_csv(results_csv_path)
        df_results.columns = df_results.columns.str.strip()
        if not df_results.empty:
            last_epoch_metrics = df_results.iloc[-1]
            validation_metrics_from_training = {
                key: last_epoch_metrics.get(col_name) for key, col_name in [
                    ('val_mAP50', 'metrics/mAP50(B)'),
                    ('val_mAP50-95', 'metrics/mAP50-95(B)'),
                    ('val_precision', 'metrics/precision(B)'),
                    ('val_recall', 'metrics/recall(B)'),
                ] if last_epoch_metrics.get(col_name) is not None
            }
            if not validation_metrics_from_training:
                print(f"Note: Could not find expected metric columns in {results_csv_path} or values were null.")
        else:
             print(f"Note: results.csv at {results_csv_path} was empty.")
    except Exception as e_csv:
        print(f"Note: Could not parse validation metrics from results.csv: {e_csv}")


summary_data = {
    'data_source_config': {
        'pre_split_dataset_path': os.path.abspath(pre_split_dataset_base_path),
        'total_images_in_split_folders': total_images_in_split_folders,
        'training_images_count': len(train_images),
        'validation_images_count': len(val_images),
        'test_images_count': len(test_images),
        'train_percentage': f"{(len(train_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%",
        'validation_percentage': f"{(len(val_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%",
        'test_percentage': f"{(len(test_images)/total_images_in_split_folders*100 if total_images_in_split_folders > 0 else 0):.1f}%",
    },
    'hyperparameters_used': {**predefined_hyperparameters, 'epochs': training_epochs, 'patience': training_patience, 'imgsz': img_size},
    'training_duration_seconds': f"{train_time:.2f}" if train_time > 0 else "N/A or Failed/Skipped",
    'trained_model_best_weights_path': trained_model_best_weights_path,
    'yolo_training_output_dir': training_output_dir,
    'validation_metrics_from_training_run': validation_metrics_from_training if validation_metrics_from_training else "Not available or training failed/skipped",
    'test_set_metrics_on_trained_model': test_metrics_results if test_metrics_results else "Skipped or Failed"
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