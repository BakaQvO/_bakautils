import os
import shutil

def move_side_images(base_dir):
    side_dir = os.path.join(base_dir, 'side')
    os.makedirs(side_dir, exist_ok=True)
    
    for file in os.listdir(base_dir):
        if file.endswith("_l.jpg") or file.endswith("_r.jpg"):
            src_path = os.path.join(base_dir, file)
            dst_path = os.path.join(side_dir, file)
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} to {dst_path}")
            
def move_side_img_labels(base_dir):
    """
    """
    ext_list = [
        "_l.jpg", "_r.jpg",
        "_l.txt", "_r.txt",
        "_l.json", "_r.json",
        ]
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.endswith(ext) for ext in ext_list) and "side" not in root:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root, "side", file)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
                print(f"Moved {src_path} to {dst_path}")
    classes = os.path.join(base_dir, "labels", "classes.txt")
    if os.path.exists(classes):
        shutil.copy(classes, os.path.join(base_dir, "labels", "side", "classes.txt"))
        

if __name__ == "__main__":
    base_dir = r"F:\TrainingData\Human\Apex\20250627-BattleScene"
    # move_side_images(base_dir)
    move_side_img_labels(base_dir)