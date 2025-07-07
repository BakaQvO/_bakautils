import os
import shutil


'''
目录规则
输入的目录为
F:\TrainingData\Human\COD\20250525-PVP
    images
        side
            xxx.jpg
            batch_000
                xxx.jpg
        xxx.jpg
        batch_000
            xxx.jpg
    labels
        classes.txt
        side
            classes.txt
            batch_000
                classes.txt
        batch_000
            classes.txt
        ....同上
    json
        ....同上

'''


def split_batch(base_dir, batch_size=2000):
    ext_list = [".jpg"]
    batch_dict = {}
    batch_dict_side = {}
    '''
    {0: {'path': 'F:\\TrainingData\\Human\\COD\\20250525-PVP\\images\\batch_000', 'num': 0}}
    '''
    
    for root, dirs, files in os.walk(base_dir):
        if "batch" in root:
            batch_id = int(root.split("batch_")[-1])
            batch_root = os.path.dirname(root)
            num = len([f for f in files if any(f.endswith(ext) for ext in ext_list)])
            if "side" in root:
                batch_dict_side[batch_id] = {"path": root, "num": num}
            else:
                batch_dict[batch_id] = {"path": root, "num": num}
    
    print(f"Batch dictionary: {batch_dict}")
    print(f"Batch side dictionary: {batch_dict_side}")
    
    for root, dirs, files in os.walk(base_dir):
        if "batch" in root:
            continue
        if "side" in root:
            batch_dict_ = batch_dict_side
        else:
            batch_dict_ = batch_dict
        for file in files:
            target_batch_id = None
            if any(file.endswith(ext) for ext in ext_list):
                file_path = os.path.join(root, file)
                if len(batch_dict_) == 0:
                    batch_id = 0
                    batch_path = os.path.join(root, f"batch_{batch_id:03d}")
                    os.makedirs(batch_path, exist_ok=True)
                    batch_dict_[batch_id] = {"path": batch_path, "num": 0}
                for batch_id in batch_dict_.keys():
                    if batch_dict_[batch_id]["num"] > batch_size:
                        raise ValueError(f"Batch {batch_id} exceeds the size limit of {batch_size}. Current size: {batch_dict_[batch_id]['num']}")
                    if batch_dict_[batch_id]["num"] == batch_size:
                        continue
                    if batch_dict_[batch_id]["num"] < batch_size:
                        batch_path = batch_dict_[batch_id]["path"]
                        target_batch_id = batch_id
                        new_file_path = os.path.join(batch_path, file)
                        shutil.move(file_path, new_file_path)
                        batch_dict_[batch_id]["num"] += 1
                        print(f"Moved {file_path} to {new_file_path} \t Batch ID: {batch_id} \t Current Size: {batch_dict_[batch_id]['num']}")
                        break
                if target_batch_id is None:
                    batch_id = max(batch_dict_.keys()) + 1
                    batch_path = os.path.join(root, f"batch_{batch_id:03d}")
                    # 等待按键确认
                    print(f"Creating new batch {batch_id} at {batch_path} file_path: {file_path}")
                    key = input("Press Enter to continue or q to quit: ")
                    if key.lower() == 'q':
                        print("Exiting...")
                        return
                    os.makedirs(batch_path, exist_ok=True)
                    batch_dict_[batch_id] = {"path": batch_path, "num": 1}
                    new_file_path = os.path.join(batch_path, file)
                    shutil.move(file_path, new_file_path)
                    print(f"Moved {file_path} to {new_file_path} \t Batch ID: {batch_id} \t Current Size: 1")
    for batch_dict_ in [batch_dict, batch_dict_side]:
        for batch_id, batch_info in batch_dict_.items():
            print(f"Batch {batch_id} has {batch_info['num']} files in {batch_info['path']}")
            img_path = batch_info["path"]
            label_path = img_path.replace("images", "labels")
            json_path = img_path.replace("images", "json")
            os.makedirs(label_path, exist_ok=True)
            os.makedirs(json_path, exist_ok=True)
        
                
        

if __name__ == "__main__":
    base_dir = r"F:\TrainingData\Human\Apex\20250627-BattleScene"
    split_batch(base_dir, batch_size=2000)