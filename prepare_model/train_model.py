import os
import numpy as np
import shutil
from ultralytics import YOLO  

def setup_directories(working_dir):
    train_dir = os.path.join(working_dir, 'train')
    val_dir = os.path.join(working_dir, 'val')
    test_dir = os.path.join(working_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, val_dir, test_dir

def split_data(input_dir, image_files, train_dir, val_dir, test_dir):
    train_end = int(0.6 * len(image_files))
    val_end = train_end + int(0.2 * len(image_files))

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    copy_files(train_files, input_dir, train_dir)
    copy_files(val_files, input_dir, val_dir)
    copy_files(test_files, input_dir, test_dir)

def copy_files(file_list, source_folder, target_folder):
    for f in file_list:
        shutil.copy(os.path.join(source_folder, f), target_folder)  
        shutil.copy(os.path.join(source_folder, f.replace('.jpg', '.txt')), target_folder)  

def write_dataset_yaml(working_dir):
    yaml_content = f"""
    path: {working_dir}
    train: train
    val: val
    test: test

    names:
    0: person
    1: car
    2: truck
    3: bus
    4: motorcycle
    """
    yaml_file = os.path.join(working_dir, 'dataset.yaml')
    
    with open(yaml_file, 'w') as file:
        file.write(yaml_content)
    return yaml_file

def train_model(yaml_file):
    os.environ['WANDB_DISABLED'] = 'true'
    model = YOLO('yolov8x.pt')
    model.train(
        data=yaml_file,
        epochs=35,
        imgsz=640,
        augment=True,
        project='runs',  
        name='my_yolov8_run'  
    )
    return model

def main():
    input_dir = '/kaggle/input/mega-set/Dataset/train/Person_Car_Bus_Truck_Motorcycle'
    working_dir = '/kaggle/working'

    train_dir, val_dir, test_dir = setup_directories(working_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    np.random.shuffle(image_files)
    #image_files = image_files[:len(image_files)//2]

    split_data(input_dir, image_files, train_dir, val_dir, test_dir)

    yaml_file = write_dataset_yaml(working_dir)

    model = train_model(yaml_file)

    metrics = model.val()
    path = model.export(format="onnx")
    print(metrics.box.map) 
    print(metrics.box.map50)

    metrics = model.val(split='test')
    print(metrics.box.map)
    print(metrics.box.map50)

if __name__ == "__main__":
    main()

