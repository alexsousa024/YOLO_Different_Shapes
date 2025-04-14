from controller import Robot
import cv2
import numpy as np
import os

save_dir = "images"
# Inicializa o robô e obtém o timestep
robot = Robot()

timestep = int(robot.getBasicTimeStep())
print("Dispositivos disponíveis:")
for i in range(robot.getNumberOfDevices()):
    print("-", robot.getDeviceByIndex(i).getName())

# Obtém a referência à câmara (usa o nome exato do dispositivo)
camera = robot.getDevice("MultiSense S21 left camera")
camera.enable(timestep)

img_counter = 0

# Loop principal
while robot.step(timestep) != -1:
    # Captura a imagem da câmara
    # Aqui podes fazer o processamento da imagem ou apenas confirmar que ela está ativa
    print("Imagem capturada com resolução:", camera.getWidth(), "x", camera.getHeight())
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # Converter a imagem para formato OpenCV
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))  # RGBA
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    image_path = os.path.join(save_dir, f"frame_{img_counter:04d}.png")
    cv2.imwrite(image_path, bgr)
    print(f"Imagem salva em: {image_path}")
    img_counter += 1

    cv2.imshow("Imagem da Câmara", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



