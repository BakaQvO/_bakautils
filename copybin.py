import os
import shutil


import time

def safe(src,dst):
    while True:
        try:
            shutil.copy2(src, dst)
            break
        except Exception as e:
            time.sleep(0.2)


if __name__ == "__main__":
    desktop_path = r"F:\Desktop"
    dst_root = r"F:\Project\DMA\bins"
    start = time.time()
    end = time.time()
    while True:
        start = time.time()

        for root, dirs, files in os.walk(desktop_path):
            if "mods" in root or \
                "模拟器" in root or \
                "系统工具" in root or \
                "Game" in root or \
                "Editor" in root or \
                "CreamAPI" in root or \
                "融合器" in root or \
                "Tentacles" in root or \
                "FPGABit" in root:
                continue
            print(f"\r Scanning {root:<200}", end="")
            for file in files:
                if file.endswith('.bin') or file.endswith('.bit'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_root, file)
                    
                    if not os.path.exists(dst_path):
                        print("")
                        print(f"Copied {src_path} to {dst_path}")
                        safe(src_path, dst_path)
        end = time.time()
        print(f"loop cost {end-start}s", end="")