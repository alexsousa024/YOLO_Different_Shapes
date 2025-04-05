from controller import Robot
import cv2
import numpy as np


# Inicializa o robô e obtém o timestep
robot = Robot()
timestep = int(robot.getBasicTimeStep())
print("Dispositivos disponíveis:")
for i in range(robot.getNumberOfDevices()):
    print("-", robot.getDeviceByIndex(i).getName())

# Obtém a referência à câmara (usa o nome exato do dispositivo)
camera = robot.getDevice("MultiSense S21 left camera")
camera.enable(timestep)

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

    cv2.imshow("Imagem da Câmara", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


