# Transfer Learning for Object Detection and Classification in Robot Perception
Pratical work of Inteligent Robotic's class in 3rd year of Bsc of Artificial Intelligence and Data Science.


Work developed by Alexandre Sousa, Marta Longo and Sara Táboas


### **Set Up**
 
- Start by installing all the libraries and dependencies needed for this project, by running the following command: 
```
pip install -r requirements.txt
```

### **Webots Maps**
- 'base.wbt' : map for the simple environment, consisting of 3 different objects (used for dataset (A))
- 'complex.wbt' : map for the complex environment, consisting of 8 different objects (used for dataset (B))

### **Datasets and Splits Generation**
**1. Dataset (A) :** 
- Generation : 
```
python dataset_a/data_generator.py
```
- Splitting (train, val and test sets) : 
```
python dataset_a/create_folders.py
```

**2. Dataset (B) :** 
- Generation : 
```
python dataset_b/data_generator_complex.py
```
- To generate the dataset for testing (containing 200 images), run the data_generator_complex.py file, replacing the line 101 by 'while supervisor.step(timestep) != -1 and img_counter < 200:'.


- Splitting (train, val and test sets) for 500 images: 
```
python dataset_b/create_folders_500.py
```
- To obtain the splits for both 1000 and 2000 images, replace in the command above the file by create_folders_1000.py and create_folders_2000.py, respectively.


### **Experiment #1**
YOLO training with the Dataset (A). The results of both training and test will be saved in the yolo_base_model folder (does not need to be generated).
```
python experiment1/yolo_training_base.py
```

### **Experiment #2**
YOLO training with the Dataset (B) - 500 images. The results of both training and test will be saved in the yolo_complex_model_500 folder (does not need to be generated).
```
python experiment2/yolo_training_complex.py
```
For the dataset splits with 1000 and 2000 images, run the command above, replacing the correct dataset split paths from the dataset_b folder and readjusting the output_dir name (yolo_complex_model_1000 and yolo_complex_model_2000).

### **Experiment #3**
YOLO training with the Dataset (B) - 500 images. The results of both training and test will be saved in the yolo_transfer_learning_500 folder (does not need to be generated).
```
python experiment3/yolo_transfer_learning.py
```
For the dataset splits with 1000 and 2000 images, run the command above, replacing the correct dataset split paths from the dataset_b folder and readjusting the output_dir name (yolo_transfer_learning_1000 and yolo_transfer_learning_2000).
