import os
import numpy as np
import cv2
from controller import Supervisor

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

camera = supervisor.getDevice("camera")
camera.enable(timestep)

cam_width = camera.getWidth()
cam_height = camera.getHeight()

save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

# Faixas para posição aleatória (x, y, z)
x_range = (-0.5, 0.5)
y_range = (-0.5, 0.5)
z_range = (0, 3.5)

target1 = supervisor.getFromDef("SOLID")  # nome do DEF do novo objeto
target1_rotation = target1.getField("rotation")
target1_translation = target1.getField("translation")

img_counter = 0

for i in range(100):
    # Gerar posição aleatória
    pos = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    target1_translation.setSFVec3f(pos)

    # Aplicar rotação aleatória
    angle = np.random.uniform(0, 3.14)
    target1_rotation.setSFRotation([0, 1, 0, angle])

    # Esperar que o objeto caia
    for _ in range(int(3000 / timestep)):
        supervisor.step(timestep)

    # Captura imagem
    raw_image = camera.getImage()
    img = np.frombuffer(raw_image, np.uint8).reshape((cam_height, cam_width, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    img_path = os.path.join(save_dir, f"frame_{img_counter:04d}.png")
    cv2.imwrite(img_path, bgr)

    # Salva label YOLO (ficheiro vazio por agora)
    label_path = os.path.join(save_dir, "labels", f"frame_{img_counter:04d}.txt")
    open(label_path, "w").close()

    print(f"[{img_counter}] Salvo: {img_path}, Label: {label_path}")
    img_counter += 1
