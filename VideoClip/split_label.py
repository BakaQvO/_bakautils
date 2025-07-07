import os
import random
import shutil

'''
可能的jpg路径
images/xxx.jpg
images/batch_000/xxx.jpg
images/side/xxx.jpg
images/side/batch_000/xxx.jpg
'''

def split(base_dir, dataset={"train":[], "val":[]}, ratio=0.8):
    """
    分割数据集
    
    Args:
        base_dir (str): 数据集路径 需要指向含有images/labels/json目录的目录
        ratio (float): 训练集和测试集的比例 默认0.8
    """
    base_dir_name = os.path.basename(base_dir)
    train_list_path = os.path.join(base_dir, f"train_{base_dir_name}.txt")
    val_list_path = os.path.join(base_dir, f"val_{base_dir_name}.txt")
    
    if os.path.exists(train_list_path):
        with open(train_list_path, 'r') as f:
            dataset["train"] = [line.strip() for line in f.readlines()]
    if os.path.exists(val_list_path):
        with open(val_list_path, 'r') as f:
            dataset["val"] = [line.strip() for line in f.readlines()]

    train_list_postive_path = os.path.join(base_dir, f"train_{base_dir_name}_postive.txt")
    val_list_postive_path = os.path.join(base_dir, f"val_{base_dir_name}_postive.txt")
    train_list_background_path = os.path.join(base_dir, f"train_{base_dir_name}_background.txt")
    val_list_background_path = os.path.join(base_dir, f"val_{base_dir_name}_background.txt")
    
    train_list = []
    val_list = []
    
    train_list_postive = []
    train_list_background = []
    
    if os.path.exists(train_list_postive_path):
        with open(train_list_postive_path, 'r') as f:
            train_list_postive = [line.strip() for line in f.readlines()]
    if os.path.exists(train_list_background_path):
        with open(train_list_background_path, 'r') as f:
            train_list_background = [line.strip() for line in f.readlines()]
    
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jpg"):
                POSTIVE_FLAG = False
                img_path = os.path.join(root, file)
                json_path = img_path.replace(".jpg", ".json").replace("images", "json")
                label_path = img_path.replace(".jpg", ".txt").replace("images", "labels")
                
                if not os.path.exists(label_path): # 图片未被标注
                    continue
                if os.path.exists(json_path):
                    POSTIVE_FLAG = True
                
                if img_path in dataset["train"] or img_path in dataset["val"]:
                    continue
                if random.random() < ratio:
                    # 训练集
                    train_list.append(img_path)
                    train_list_postive.append(img_path) if POSTIVE_FLAG else train_list_background.append(img_path)
                else:
                    # 测试集
                    val_list.append(img_path)
    # 将训练集和测试集的路径写入文件
    with open(train_list_path, 'w') as f:
        for item in train_list:
            f.write(f"{item}\n")
    with open(val_list_path, 'w') as f:
        for item in val_list:
            f.write(f"{item}\n")
    with open(train_list_postive_path, 'w') as f:
        for item in train_list_postive:
            f.write(f"{item}\n")
    with open(train_list_background_path, 'w') as f:
        for item in train_list_background:
            f.write(f"{item}\n")
    return train_list, val_list


if __name__ == "__main__":
    base_dir = r"F:\TrainingData\Human\Apex\20250627-BattleScene"
    
    
    split(base_dir)