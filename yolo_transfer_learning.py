# yolo_transfer_learning.py

# 1. Load the model from 'yolo_complex_model'
# 2. Load the dataset_inicial 'complexEnv'
# 3. Choose which layers should be frozen
# 4. Retrain the model --> train and validation sets (validação automática durante o treino)
# 5. Test the model --> test set and robotic arm
#     5.1 arm input : prediction coordinates
#     5.2 predictive picking

from ultralytics import YOLO
import yaml
import torch
import os


# ----------------------- Configuração -----------------------
BASE_MODEL_PATH = 'best.pt'
DATASET_PATH = 'split_dataset_1000'
DATASET_TEST_PATH = 'dataset_test'
YAML_PATH = 'transferLearning.yaml'
OUTPUT_BASE_DIR = 'yolo_transfer_learning_1000'
NUM_CLASSES = 4
CLASS_NAMES = ['sphere', 'cone', 'cylinder', 'cube']
IMG_SIZE = 640
EPOCHS = 10
PATIENCE = 5
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------- Preparar Diretórios --------------------------
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
training_project_dir = os.path.join(OUTPUT_BASE_DIR, 'model_training_run')
evaluation_project_dir = os.path.join(OUTPUT_BASE_DIR, 'model_evaluation_run')


# ---------------------- Verificar Estrutura ---------------------------
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


# ------------------------ Criar arquivo YAML --------------------------
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


# ------------------- Carregar Modelo Pré-treinado ---------------------
model = YOLO(BASE_MODEL_PATH)

print("Estrutura do Modelo Do Ambiente Simples: ")
print(model.model)

# Atualizar manualmente o número de classes na camada de detecção (head)
# Esta versão é compatível com versões que não têm reset_class()
detect_layer = model.model.model[-1]  # última camada (Detect)
ch = detect_layer.cv2[0][0].conv.in_channels  # extrai canais de entrada corretamente
model.model.model[-1] = detect_layer.__class__(nc=NUM_CLASSES, ch=[ch, ch, ch])

# model.model = model.model.reset_class(NUM_CLASSES)

# --------------------- Congelar Camadas do Modelo ---------------------
def freeze_backbone_layers(model):
    # model.model refere-se à estrutura interna do modelo PyTorch dentro do objeto YOLO
    for name, module in model.model.named_children():
        # Condição para identificar o backbone:
        # 1. 'backbone' in name: Algumas arquiteturas nomeiam explicitamente a seção do backbone.
        # 2. name == '0': No YOLOv8, o backbone é tipicamente o primeiro grande bloco de camadas,
        #    acessível como model.model[0] (ou seja, o filho com nome '0').
        if 'backbone' in name or name == '0':
            for param in module.parameters():

                param.requires_grad = False
    print("Camadas do backbone congeladas.")

freeze_backbone_layers(model)


print("Estrutura do Modelo Depois do Cangelamento das Camadas de Classificação: ")
print(model.model)

# ------------------------- Treinar Modelo -----------------------------
print("Iniciando treino com Transfer Learning...")

train_results = model.train(
    data=yaml_file,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project=training_project_dir,
    device=DEVICE,
    verbose=True  # mostra progresso por época
)

print("Treino finalizado.")


# -------------------------- Avaliação Final ---------------------------
print("Avaliando o modelo no conjunto de TESTE...")
test_metrics = model.val(
    data=yaml_file,
    split='test',
    project=evaluation_project_dir,
    device=DEVICE
)
print("Avaliação no test set concluída.")


# ------------------- Predição em imagens reais de teste ---------------
# Esta secção simula o uso do modelo para o braço robótico:
#   - Localiza os objetos
#   - Calcula o centro (x, y) do bounding box
#   - (Simulado) envia coordenadas para o braço

print("\n--- Predição e picking simulado pelo braço robótico ---")
sample_images_dir = os.path.join(DATASET_PATH, 'test', 'images')
sample_images = sorted([f for f in os.listdir(sample_images_dir) if f.endswith('.jpg') or f.endswith('.png')])

# Exemplo com apenas a primeira imagem
sample_path = os.path.join(sample_images_dir, sample_images[0])
results = model(sample_path)

# Iterar sobre objetos detectados
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # centro do objeto detectado
        print(f"[Braço Robótico] Coordenadas previstas: ({cx:.1f}, {cy:.1f})")


"""
def move_arm_to(x, y):
    print(f"--> Braço movido para ({x:.1f}, {y:.1f}) e objeto recolhido.")

# Simular picking
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        move_arm_to(cx, cy)
"""
