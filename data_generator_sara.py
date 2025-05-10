from controller import Robot, CameraRecognitionObject
import cv2
import numpy as np
import os

# --- Configurações ---
SAVE_DIR_IMAGES = "dataset/images"
SAVE_DIR_LABELS = "dataset/labels"
ANNOTATION_FILENAME_PREFIX = "frame" # Nome base para imagens e anotações

# Mapeamento: Nome do MODELO do objeto no Webots -> ID da classe YOLO (começando em 0)

CLASS_MAPPING = {
    "cube": 0,
    "oil barrel": 1,
}

# ------------ Initialization ------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ------------ Devices -------------
# 1. camera - used to capture the images of the world
camera = robot.getDevice("MultiSense S21 left camera")
if camera is None:
    print("Error: Camera 'MultiSense S21 left camera' not found.")
    exit()

camera.enable(timestep)
camera.recognitionEnable(timestep)


# Devices dimensions
width_saving = camera.getWidth()
height_saving = camera.getHeight()
if width_saving == 0 or height_saving == 0:
    print("Error: Invalid camera dimensions - either height or width is equal to 0.")
    exit()

width_detection = width_saving
height_detection = height_saving

# Create output directories (if they don't exist)
os.makedirs(SAVE_DIR_IMAGES, exist_ok=True)
os.makedirs(SAVE_DIR_LABELS, exist_ok=True)
print(f"Images saved in: {os.path.abspath(SAVE_DIR_IMAGES)}")
print(f"Annotations saved: {os.path.abspath(SAVE_DIR_LABELS)}")

img_counter = 0

print("Main loop initializing. To quit, press 'q'")

# --- Main Loop ---
while robot.step(timestep) != -1:
    # 1. Capture Image to Save
    image_raw_saving = camera.getImage()
    if image_raw_saving is None:
        print("Warning: Fail to capture an image from camera.")
        continue # Next iteration

    # Convert to OpenCV format
    img_np = np.frombuffer(image_raw_saving, np.uint8).reshape((height_saving, width_saving, 4)) # RGBA
    bgr_img_saving = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # 2. Capture the image from Detection Camera - annotations
    image_raw_detection = image_raw_saving
    bgr_img_detection_viz = bgr_img_saving.copy()

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

    cv2.imshow("Câmera de Detecção com BBoxes (Pressione 'q' para sair)", bgr_img_detection_viz)

    # Opcional: Exibir também a imagem que está sendo salva
    # cv2.imshow("Câmera de Salvamento", bgr_img_saving)

    key = cv2.waitKey(1)  # Espera 1ms pela tecla
    if key & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Terminando...")
        break


cv2.destroyAllWindows()
# Adicionar um pequeno step final pode ajudar a garantir que a simulação termine limpa
robot.step(timestep)
print(f"Processo concluído. {img_counter} imagens salvas em '{SAVE_DIR_IMAGES}' e anotações em '{SAVE_DIR_LABELS}'.")