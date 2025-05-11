import os
import shutil
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Parâmetros
dataset_path = 'dataset/images'
labels_path = 'dataset/labels'
img_ext = '.png'
k_folds = 5

# Obter todos os ficheiros de imagem
images = [f for f in os.listdir(dataset_path) if f.endswith(img_ext)]
images.sort()

# K-Fold split
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    print(f"\n🔁 Fold {fold + 1}/{k_folds}")

    # Criar estrutura temporária para treino/validação
    split_dir = f'splits/fold{fold}'
    train_img_dir = os.path.join(split_dir, 'images/train')
    val_img_dir = os.path.join(split_dir, 'images/val')
    train_lbl_dir = os.path.join(split_dir, 'labels/train')
    val_lbl_dir = os.path.join(split_dir, 'labels/val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Copiar imagens e labels para as pastas corretas
    for idxs, img_dir, lbl_dir in [
        (train_idx, train_img_dir, train_lbl_dir),
        (val_idx, val_img_dir, val_lbl_dir),
    ]:
        for i in idxs:
            img_file = images[i]
            base_name = os.path.splitext(img_file)[0]
            shutil.copy(os.path.join(dataset_path, img_file), os.path.join(img_dir, img_file))
            shutil.copy(os.path.join(labels_path, base_name + '.txt'), os.path.join(lbl_dir, base_name + '.txt'))

    # Criar YAML para este fold
    fold_yaml = f'{split_dir}/data.yaml'
    with open(fold_yaml, 'w') as f:
        yaml.dump({
            'train': os.path.abspath(train_img_dir),
            'val': os.path.abspath(val_img_dir),
            'nc': 3,
            'names': ['sphere', 'cone','cylinder']
        }, f)

    # Treinar o modelo
    model = YOLO('yolov8n.pt')
    model.train(data=fold_yaml, epochs=50, imgsz=640, batch=16, name=f'yolov8_fold{fold}', classes = 3)

    # Avaliar o modelo
    metrics = model.val()

    fold_metrics.append({
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'f1': metrics.box.f1,
        'map50': metrics.box.map50,
        'map': metrics.box.map,
    })

# Média das métricas
print("\n📊 Resultados Médios dos Folds:")
avg_metrics = {k: sum(m[k] for m in fold_metrics) / k_folds for k in fold_metrics[0]}
for k, v in avg_metrics.items():
    print(f"{k}: {v:.4f}")