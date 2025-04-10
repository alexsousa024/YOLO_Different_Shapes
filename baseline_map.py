from controller import Robot, Camera, Supervisor
import cv2
import numpy as np
import os

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Pastas para guardar dataset
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

# Ativa a câmara
camera = robot.getDevice("MultiSense S21 left camera")
camera.enable(timestep)

print("Tipo do controlador:", type(robot).__name__)

# Suponhamos que os objetos têm DEFs: "SHAPE_1", "SHAPE_2", etc
shape_names = ["SHAPE_1", "SHAPE_2"]  # podes adicionar mais
shape_classes = {"SHAPE_1": 0, "SHAPE_2": 1}  # class_id por forma

img_count = 0

# Correr apenas durante 1 segundo
max_steps = int(1000 / timestep)
for step in range(max_steps):
    if robot.step(timestep) == -1:
        break
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))  # RGBA
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Guardar imagem
    img_filename = f"dataset/images/frame_{img_count}.jpg"
    cv2.imwrite(img_filename, bgr)

    # Criar ficheiro de labels
    label_filename = f"dataset/labels/frame_{img_count}.txt"
    with open(label_filename, "w") as label_file:
        for shape_def in shape_names:
            shape_node = robot.getFromDef(shape_def)
            if shape_node is None:
                continue
            pos = shape_node.getPosition()
            camera_position = camera.getPosition()
            relative_pos = [pos[i] - camera_position[i] for i in range(3)]

            # Projetar ponto 3D para 2D
            projection = camera.project(pos)
            if projection:
                x_img, y_img = projection
                # Normalizar
                x_center = x_img / width
                y_center = y_img / height

                bounding_object = shape_node.getBoundingObject()
                if bounding_object is not None:
                    extent = bounding_object.getExtent()
                    if extent is not None:
                        # Calcular largura e altura aproximadas a partir da extensão
                        x_size = extent[0]
                        y_size = extent[1]
                        z_size = extent[2]

                        # Estimar projeção da largura e altura no plano da imagem
                        p1 = camera.project([pos[0] - x_size/2, pos[1], pos[2]])
                        p2 = camera.project([pos[0] + x_size/2, pos[1], pos[2]])
                        p3 = camera.project([pos[0], pos[1] + y_size/2, pos[2]])
                        p4 = camera.project([pos[0], pos[1] - y_size/2, pos[2]])

                        if p1 and p2 and p3 and p4:
                            # box_width e box_height são calculados projetando os extremos do objeto na imagem:
                            # - p1 e p2 representam os extremos esquerdo e direito (em X)
                            # - p3 e p4 representam os extremos superior e inferior (em Y)
                            # A diferença entre p1 e p2 (e entre p3 e p4) dá-nos a dimensão projetada em píxeis,
                            # que é depois normalizada dividindo pela largura ou altura da imagem.
                            box_width = abs(p2[0] - p1[0]) / width
                            box_height = abs(p3[1] - p4[1]) / height

                            class_id = shape_classes[shape_def]
                            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"Imagem e label guardados: frame_{img_count}")
    img_count += 1

    # Mostrar imagem (opcional)
    cv2.imshow("Imagem da Câmara", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()