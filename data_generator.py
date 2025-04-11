from controller import Supervisor
import numpy as np
import cv2
import os

# Pasta onde salvar as imagens e labels
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

# Inicializa o supervisor
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# Câmara do robô
camera = supervisor.getDevice("MultiSense S21 left camera")
camera.enable(timestep)
cam_width = camera.getWidth()
cam_height = camera.getHeight()

# Objeto que vamos seguir (com DEF="BOX1")
target = supervisor.getFromDef("RectangleArena")
print(target)
target_translation = target.getField("translation")
target_size = target.getField("size").getSFVec3f()  # largura, altura, profundidade

img_counter = 0

# Para 10 imagens
for i in range(10):
    # Define nova posição aleatória (dentro do campo de visão da câmara)
    new_x = np.random.uniform(-1.0, 1.0)
    new_z = np.random.uniform(0.5, 2.0)  # distancia da câmara
    target_translation.setSFVec3f([new_x, 0.05, new_z])

    for _ in range(5):  # espera alguns passos para estabilizar a cena
        supervisor.step(timestep)

    # Captura imagem
    raw_image = camera.getImage()
    img = np.frombuffer(raw_image, np.uint8).reshape((cam_height, cam_width, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # === Bounding Box YOLO ===
    # Usa a posição real do objeto (projetada na imagem)
    obj_pos = target.getPosition()  # posição global
    proj = camera.getRecognitionCamera().getCameraMatrix()
    cam_pos = supervisor.getFromDef("SUPERVISOR_CAMERA").getPosition()

    # Simples projeção 2D (estimada)
    # Estes valores são só aproximações para gerar os labels
    x_center = 0.5 + (obj_pos[0] / 2.0)
    y_center = 0.5 - (obj_pos[2] / 2.5)
    width = 0.1
    height = 0.1

    # Clamp entre 0 e 1
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)

    # Salva imagem
    img_path = os.path.join(save_dir, f"frame_{img_counter:04d}.png")
    cv2.imwrite(img_path, bgr)

    # Salva label YOLO
    label_path = os.path.join(save_dir, "labels", f"frame_{img_counter:04d}.txt")
    with open(label_path, "w") as f:
        # classe_id x_center y_center width height (tudo normalizado [0,1])
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"[{img_counter}] Salvo: {img_path}, Label: {label_path}")
    img_counter += 1
