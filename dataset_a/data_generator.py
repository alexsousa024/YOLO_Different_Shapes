from controller import Robot, CameraRecognitionObject, Supervisor
import cv2
import numpy as np
import os


# --- Configurations ---
SAVE_DIR_IMAGES = "dataset_baseEnv/images"
SAVE_DIR_LABELS = "dataset_baseEnv/labels"
ANNOTATION_FILENAME_PREFIX = "frame"

# Class mapping for the objects present in the environment
CLASS_MAPPING = {
    "ball solid": 0,
    "cylinder": 1,
    "cone":2,
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

x_range = (-0.4, 0.4)
y_range = (-0.4, 0.4)
z_range = (0, 1.2)




# Cylinder
target1 = supervisor.getFromDef("Cylinder")  # nome do DEF do novo objeto
target1_rotation = target1.getField("rotation")
target1_translation = target1.getField("translation")

# Ball
target2 = supervisor.getFromDef("Ball")  # nome do DEF do novo objeto
target2_rotation = target2.getField("rotation")
target2_translation = target2.getField("translation")

# Cone
target3 = supervisor.getFromDef("CONE")  # nome do DEF do novo objeto
target3_rotation = target3.getField("rotation")
target3_translation = target3.getField("translation")



# --- Main Loop ---
while supervisor.step(timestep) != -1 and img_counter < 2000:

    # Generate random position
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



    # Apply random rotation
    angle = np.random.uniform(0, 3.14)
    target1_rotation.setSFRotation([0, 1, 0, angle])
    target2_rotation.setSFRotation([0, 1, 0, angle])
    target3_rotation.setSFRotation([0, 1, 0, angle])


    # Wait until all objects are stopped --> angular velocity < 0.05
    def is_object_stationary(object_node, threshold=0.05):
        vel = object_node.getVelocity()
        if vel is None:
            return False
        linear_velocity = vel[:3]
        return np.linalg.norm(linear_velocity) < threshold

    # 5 seconds timeout (5000 ms) - if the images don't achieve the angular velocity threshold in 5 seconds, the frame is ignored
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
        print("Warning: 5 second timeout. The objects did not stop. Stepping to the next frame.")

    # Delay before capturing image
    for _ in range(int(0.1 * 1000 / timestep)):
        supervisor.step(timestep)

    # Capture Image to Save
    image_raw_saving = camera.getImage()
    if image_raw_saving is None:
        print("Warning: Fail to capture an image from camera.")
        continue # Next iteration

    # Convert to OpenCV format
    img_np = np.frombuffer(image_raw_saving, np.uint8).reshape((height_saving, width_saving, 4)) # RGBA
    bgr_img_saving = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # Capture the image from Detection Camera - annotations
    image_raw_detection = camera.getImage()
    if image_raw_detection is None:
        print("Warning: Fail to capture an image from detection camera.")
        bgr_img_detection_viz = np.zeros((height_detection, width_detection, 3),
                                         dtype=np.uint8)  # Black image in case of failure

    else: # Convert the image to OpenCV format
        img_np_detection = np.frombuffer(image_raw_detection, np.uint8).reshape((height_detection, width_detection, 4)) # RGBA
        bgr_img_detection_viz = cv2.cvtColor(img_np_detection, cv2.COLOR_RGBA2BGR)  # Image to draw the Bounding Boxes



    # Obtain recognized objects by the Detection Camera
    recognized_objects = camera.getRecognitionObjects()
    print(f"DEBUG [Frame {img_counter}]: Number of rec objs = {len(recognized_objects)}")

    yolo_annotations = []

    # Process recognized objects
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
            pos_obj = obj.position_on_image
            size_obj = obj.size_on_image
            x_min_unk = int(pos_obj[0] - size_obj[0] / 2)
            y_min_unk = int(pos_obj[1] - size_obj[1] / 2)
            x_max_unk = int(pos_obj[0] + size_obj[0] / 2)
            y_max_unk = int(pos_obj[1] + size_obj[1] / 2)
            cv2.rectangle(bgr_img_detection_viz, (x_min_unk, y_min_unk), (x_max_unk, y_max_unk), (0, 0, 255),
                          1)  # red
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
        print(f"Error saving annotation: {label_path}: {e}")

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


    cv2.imshow("Detection Camera with BBoxes (Press 'q' to exit)", bgr_img_detection_viz)


    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Keyboard 'q' pressed. Exiting...")
        break


cv2.destroyAllWindows()
supervisor.step(timestep)
print(f"Process concluded. {img_counter} images saved in '{SAVE_DIR_IMAGES}' and annotations in '{SAVE_DIR_LABELS}'.")