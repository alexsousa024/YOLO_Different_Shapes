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

# Obtemos os dois objetos que queremos mover
target1 = supervisor.getFromDef("BALL")
target2 = supervisor.getFromDef("OIL_BARREL")

target1_rotation = target1.getField("rotation")
target2_rotation = target2.getField("rotation")

# Permitir que os objetos caiam naturalmente durante 5 segundos
print("A aguardar queda dos objetos...")
for _ in range(int(5000 / timestep)):
    supervisor.step(timestep)
print("Início da captura.")

img_counter = 0


# Para 10 imagens
for i in range(10):
    # Define nova posição aleatória (dentro do campo de visão da câmara)
    # Define nova rotação aleatória no eixo Y
    angle1 = np.random.uniform(0, 3.14)
    angle2 = np.random.uniform(0, 3.14)
    target1_rotation.setSFRotation([0, 1, 0, angle1])
    target2_rotation.setSFRotation([0, 1, 0, angle2])


    for _ in range(int(2000 / timestep)):  # espera 2 segundos para a gravidade atuar
        supervisor.step(timestep)

    # Captura imagem
    raw_image = camera.getImage()
    img = np.frombuffer(raw_image, np.uint8).reshape((cam_height, cam_width, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    print("ola")
    img_path = os.path.join(save_dir, f"frame_{img_counter:04d}.png")
    cv2.imwrite(img_path, bgr)

    # Salva label YOLO
    label_path = os.path.join(save_dir, "labels", f"frame_{img_counter:04d}.txt")

    print(f"[{img_counter}] Salvo: {img_path}, Label: {label_path}")
    img_counter += 1