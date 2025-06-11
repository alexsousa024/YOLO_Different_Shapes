from controller import Robot, CameraRecognitionObject, Supervisor
import cv2
import numpy as np
import os


# --- Configurações ---
SAVE_DIR_IMAGES = "dataset_simples_cor/images"
SAVE_DIR_LABELS = "dataset_simples_cor/labels"
ANNOTATION_FILENAME_PREFIX = "frame" # Nome base para imagens e anotações

# Mapeamento: Nome do MODELO do objeto no Webots -> ID da classe YOLO (começando em 0)

CLASS_MAPPING = {
    "ball solid": 0,
    "cylinder": 1,
    "cone":2,
    "cube":3,
}

# ------------ Initialization ------------
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# ------------ Devices -------------
camera = supervisor.getDevice("camera(1)")

camera.enable(timestep)
camera.recognitionEnable(timestep)
# Devices dimensions
width_saving = camera.getWidth()
height_saving = camera.getHeight()
if width_saving == 0 or height_saving == 0:
    print("Error: Invalid camera dimensions - either height or width is equal to 0.")
    exit()

width_detection = camera.getWidth()
height_detection = camera.getHeight()
if width_detection == 0 or height_detection == 0:
    print("Error: Invalid detection camera dimensions - either height or width is equal to 0.")
    exit()

# Create output directories (if they don't exist)
os.makedirs(SAVE_DIR_IMAGES, exist_ok=True)
os.makedirs(SAVE_DIR_LABELS, exist_ok=True)
print(f"Images saved in: {os.path.abspath(SAVE_DIR_IMAGES)}")
print(f"Annotations saved: {os.path.abspath(SAVE_DIR_LABELS)}")

img_counter = 0
iteration_counter = 0

print("Main loop initializing. To quit, press 'q'")

# Faixas para posição aleatória (x, y, z)
x_range = (-0.4, 0.4)
y_range = (-0.4, 0.4)
z_range = (0, 1.2)

#Mudar o peso para que os objetos não caiam tanto


#Cilindro
target1 = supervisor.getFromDef("Cylinder")  # nome do DEF do novo objeto
target1_rotation = target1.getField("rotation")
target1_translation = target1.getField("translation")

#Ball
target2 = supervisor.getFromDef("Ball")  # nome do DEF do novo objeto
target2_rotation = target2.getField("rotation")
target2_translation = target2.getField("translation")

#Ball
target3 = supervisor.getFromDef("CONE")  # nome do DEF do novo objeto
target3_rotation = target3.getField("rotation")
target3_translation = target3.getField("translation")

"""
#Ball
target4 = supervisor.getFromDef("Cube")  # nome do DEF do novo objeto
target4_rotation = target4.getField("rotation")
target4_translation = target4.getField("translation")

#Ball
target5 = supervisor.getFromDef("Cube2")  # nome do DEF do novo objeto
target5_rotation = target5.getField("rotation")
target5_translation = target5.getField("translation")

#Ball
target6 = supervisor.getFromDef("CONE2")  # nome do DEF do novo objeto
target6_rotation = target6.getField("rotation")
target6_translation = target6.getField("translation")

#Ball
target7 = supervisor.getFromDef("Ball2")  # nome do DEF do novo objeto
target7_rotation = target7.getField("rotation")
target7_translation = target7.getField("translation")

#Ball
target8 = supervisor.getFromDef("Cylinder2")  # nome do DEF do novo objeto
target8_rotation = target8.getField("rotation")
target8_translation = target8.getField("translation")
"""

# --- Main Loop ---
while supervisor.step(timestep) != -1 and img_counter < 2000:

    # Gerar posição aleatória
    pos1 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos2 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos3 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos4 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos5 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos6 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos7 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]
    pos8 = [np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)]

    target1_translation.setSFVec3f(pos1)
    target2_translation.setSFVec3f(pos2)
    target3_translation.setSFVec3f(pos3)
    """

    target4_translation.setSFVec3f(pos4)
    target5_translation.setSFVec3f(pos5)
    target6_translation.setSFVec3f(pos6)
    target7_translation.setSFVec3f(pos7)
    target8_translation.setSFVec3f(pos8)
    """



    # Aplicar rotação aleatória
    angle = np.random.uniform(0, 3.14)
    target1_rotation.setSFRotation([0, 1, 0, angle])
    target2_rotation.setSFRotation([0, 1, 0, angle])
    target3_rotation.setSFRotation([0, 1, 0, angle])
    """
    target4_rotation.setSFRotation([0, 1, 0, angle])
    target5_rotation.setSFRotation([0, 1, 0, angle])
    target6_rotation.setSFRotation([0, 1, 0, angle])
    target7_rotation.setSFRotation([0, 1, 0, angle])
    target8_rotation.setSFRotation([0, 1, 0, angle])
    """

    # Esperar até que todos os objetos estejam parados
    # Função auxiliar para verificar se o objeto está praticamente parado
    def is_object_stationary(object_node, threshold=0.05):
        vel = object_node.getVelocity()
        if vel is None:
            return False
        linear_velocity = vel[:3]
        return np.linalg.norm(linear_velocity) < threshold

    # Timeout de 5 segundos (5000 ms)
    max_wait_time_ms = 5000
    elapsed_time_ms = 0
    while elapsed_time_ms < max_wait_time_ms:
        supervisor.step(timestep)
        elapsed_time_ms += timestep
        if all([
            is_object_stationary(target1),
            is_object_stationary(target2),
            is_object_stationary(target3)
        ]):
            break
    else:
        print("Aviso: Timeout de 5 segundos atingido antes de todos os objetos pararem.")

    # Pequeno atraso antes de capturar a imagem
    for _ in range(int(0.1 * 1000 / timestep)):
        supervisor.step(timestep)

    # 1. Capture Image to Save
    image_raw_saving = camera.getImage()
    if image_raw_saving is None:
        print("Warning: Fail to capture an image from camera.")
        continue # Next iteration

    # Convert to OpenCV format
    img_np = np.frombuffer(image_raw_saving, np.uint8).reshape((height_saving, width_saving, 4)) # RGBA
    bgr_img_saving = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # 2. Capture the image from Detection Camera - annotations
    image_raw_detection = camera.getImage()
    if image_raw_detection is None:
        print("Warning: Fail to capture an image from detection camera.")
        bgr_img_detection_viz = np.zeros((height_detection, width_detection, 3),
                                         dtype=np.uint8)  # Black image in case of failure

    else: # Convert the image to OpenCV format
        img_np_detection = np.frombuffer(image_raw_detection, np.uint8).reshape((height_detection, width_detection, 4)) # RGBA
        bgr_img_detection_viz = cv2.cvtColor(img_np_detection, cv2.COLOR_RGBA2BGR)  # Image to draw the Bounding Boxes



    # 3. Obtain recognized objects by the Detection Camera
    recognized_objects = camera.getRecognitionObjects()
    print(f"DEBUG [Frame {img_counter}]: Number of rec objs = {len(recognized_objects)}")

    yolo_annotations = []

    # 4. Process recognized objects
    for obj in recognized_objects:
        # Coordinates relative to Detection Camera
        position = obj.position_on_image
        size = obj.size_on_image
        model_name = obj.getModel()
        print(model_name)
        if model_name:
            pass
        else:
            print("Warning: Recognized object without model name")
            continue


        # Check if the model is in CLASS_MAPPING
        if model_name in CLASS_MAPPING:
            class_id = CLASS_MAPPING[model_name]

            # Compute YOLO coordinates (normalized) --> relative to Detection Camera
            center_x_norm = position[0] / width_detection
            center_y_norm = position[1] / height_detection
            width_norm = size[0] / width_detection
            height_norm = size[1] / height_detection

            center_x_norm = np.clip(center_x_norm, 0.0, 1.0)
            center_y_norm = np.clip(center_y_norm, 0.0, 1.0)
            width_norm = np.clip(width_norm, 0.0, 1.0)
            height_norm = np.clip(height_norm, 0.0, 1.0)

            # YOLO annotation line - Format
            yolo_line = f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_annotations.append(yolo_line)

            # Draw the BBox
            x_min = int((center_x_norm - width_norm / 2) * width_detection)
            y_min = int((center_y_norm - height_norm / 2) * height_detection)
            x_max = int((center_x_norm + width_norm / 2) * width_detection)
            y_max = int((center_y_norm + height_norm / 2) * height_detection)
            cv2.rectangle(bgr_img_detection_viz, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Verde
            cv2.putText(bgr_img_detection_viz, f"{model_name}:{class_id}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print(f"Warning: Recognized object '{model_name}' not in CLASS_MAPPING.")
            # Opcional: Desenhar caixa diferente para objetos não mapeados na imagem de visualização
            pos_obj = obj.position_on_image
            size_obj = obj.size_on_image
            x_min_unk = int(pos_obj[0] - size_obj[0] / 2)
            y_min_unk = int(pos_obj[1] - size_obj[1] / 2)
            x_max_unk = int(pos_obj[0] + size_obj[0] / 2)
            y_max_unk = int(pos_obj[1] + size_obj[1] / 2)
            cv2.rectangle(bgr_img_detection_viz, (x_min_unk, y_min_unk), (x_max_unk, y_max_unk), (0, 0, 255),
                          1)  # Vermelho
            cv2.putText(bgr_img_detection_viz, f"{model_name}: N/A", (x_min_unk, y_min_unk - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # 5. Save images (from CAMERA) and annotations (based on DETECTION_CAMERA)
    base_filename = f"{ANNOTATION_FILENAME_PREFIX}_{img_counter:06d}"
    image_path = os.path.join(SAVE_DIR_IMAGES, f"{base_filename}.png")
    label_path = os.path.join(SAVE_DIR_LABELS, f"{base_filename}.txt")

    save_success = cv2.imwrite(image_path, bgr_img_saving)
    if not save_success:
        print(f"Error when saving image: {image_path}")

    try:
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_annotations))
    except IOError as e:
        print(f"Erro ao salvar anotação em {label_path}: {e}")

    img_counter += 1
    iteration_counter += 1

    if iteration_counter % 10 == 0:
        def set_random_color(target):
            shape_node = target.getField("children").getMFNode(0).getField("appearance").getSFNode().getField("material").getSFNode()
            color_field = shape_node.getField("diffuseColor")
            random_color = [np.random.rand(), np.random.rand(), np.random.rand()]
            color_field.setSFColor(random_color)

        set_random_color(target1)
        set_random_color(target2)
        set_random_color(target3)
        """
        set_random_color(target4)
        set_random_color(target5)
        set_random_color(target6)
        set_random_color(target7)
        set_random_color(target8)
        """

    cv2.imshow("Câmera de Detecção com BBoxes (Pressione 'q' para sair)", bgr_img_detection_viz)

    # Opcional: Exibir também a imagem que está sendo salva
    # cv2.imshow("Câmera de Salvamento", bgr_img_saving)

    key = cv2.waitKey(1)  # Espera 1ms pela tecla
    if key & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Terminando...")
        break


cv2.destroyAllWindows()
# Adicionar um pequeno step final pode ajudar a garantir que a simulação termine limpa
supervisor.step(timestep)
print(f"Processo concluído. {img_counter} imagens salvas em '{SAVE_DIR_IMAGES}' e anotações em '{SAVE_DIR_LABELS}'.")