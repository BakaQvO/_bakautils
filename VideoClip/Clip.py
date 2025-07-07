import cv2
import os
import numpy as np
import random
import av
from PIL import Image
import time
import json
os.environ["PATH"] = r"C:\Libs\ffmpeg-7.1.1-full_build-shared\bin;" + os.environ["PATH"]

def clip_img(frame:np.ndarray, video_name, frame_id, output_dir,area=1280):
    """
    裁剪保存图片 自动判断图片宽高比 裁剪规则如下\n
    1. 16:9 裁剪中心1280x1280矩形区域\n
    2. 21:9 裁剪中心1280x1280矩形区域 以及左右两侧以图片高为边长的矩形\n
    3. 此外的图片裁剪中心1280x1280矩形区域\n
    
    Args:
        frame (ndarray): 视频帧
        video_name (str): 视频文件名
        frame_id (int): 帧编号
        output_dir (str): 输出目录
    """
    img_w = frame.shape[1]
    img_h = frame.shape[0]
    
    
    left = max(int((img_w - area) / 2), 0) # 防止负数
    right = min(int((img_w + area) / 2), img_w) # 防止超出边界
    top = max(int((img_h - area) / 2), 0)
    bottom = min(int((img_h + area) / 2), img_h)
    
    img_m = frame[top:bottom, left:right]
    cv2.imwrite(os.path.join(output_dir, f"{video_name}_{frame_id}_m.jpg"), img_m)
    
    black_border_flag = False
    
    check_point_l = [150, img_h/2] # 左侧检查点
    check_point_r = [img_w - 150, img_h/2] # 右侧检查点
    
    check_point_l_clr = frame[int(check_point_l[1]), int(check_point_l[0])]
    check_point_r_clr = frame[int(check_point_r[1]), int(check_point_r[0])]
    if sum(check_point_l_clr) < 30 and sum(check_point_r_clr) < 30:
        black_border_flag = True
    
    if img_w / img_h > 2.3 and not black_border_flag: # 21:9
        img_l = frame[0:img_h, 0:img_h]
        img_r = frame[0:img_h, img_w - img_h:img_w]
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_{frame_id}_l.jpg"), img_l)
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_{frame_id}_r.jpg"), img_r)
    
    return

def clip_video(video_path, output_dir, interval):
    """
    将视频随机跳帧截图保存
    
    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        interval (list): 截图间隔 单位帧 对应FPS=30 如果FPS更高则interval=FPS/30*1.7
    
    Returns:
        int: 处理结果 0: 处理失败 -1: 已被处理 1: 处理成功
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if video_name in file:
                print(f"[INFO] {video_name} already processed")
                return -1
    start_time = time.time()
    with av.open(video_path) as container:
        for stream in container.streams:
            if stream.type == 'video':
                video_stream = stream
                break
        else:
            print(f"[ERROR] No video stream found in {video_path}")
            return 0
        FPS = float(video_stream.average_rate or video_stream.guessed_rate)
        # cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        # FPS = cap.get(cv2.CAP_PROP_FPS)
        if FPS is None or FPS == 0:
            print(f"[ERROR] Unable to get FPS from video file {video_path}")
            FPS = 30
        FPS = round(FPS)
        duration = video_stream.duration * float(video_stream.time_base)
        if duration and duration > 1250:
            interval = [interval[0] * 2, interval[1] * 5]
        interval = [round(FPS / 30 * 1.7/2 * i) for i in interval] # FPS30->15-45帧间隔 60->25-76
        print(f"[INFO] {video_path} FPS: {FPS} interval: {interval} duration: {duration}")
        
        
        clip_interval = random.randint(interval[0], interval[1])
        last_clip_frame_id = 0
        
        
        try:
            for packet in container.demux(video_stream):
                try:
                    frames = packet.decode()
                except Exception as e:
                    print(f"[ERROR] Error decoding packet: {e}")
                    continue
                for frame in frames:
                    frame_id = int(frame.pts * float(video_stream.time_base) * float(video_stream.average_rate or video_stream.guessed_rate))
                    # print(f"\r[INFO] Frameid {frame_id} = {frame.pts} pts, rate {float(video_stream.average_rate or video_stream.guessed_rate)} time_base {float(video_stream.time_base)}", end="")
                    print(f"\r[INFO] {video_path} \t {frame_id:8d} / {int(duration * FPS)}\t{frame_id/(duration*FPS)*100:5.2f}%", end="")
                    if clip_interval == 0:
                        img = frame.to_ndarray(format='bgr24')
                        clip_img(img, video_name, frame_id, output_dir, area=960)
                        clip_interval = random.randint(interval[0], interval[1])
                        last_clip_frame_id = frame_id
                    else:
                        clip_interval -= 1
        except Exception as e:
            print(f"[ERROR] Error processing video file {video_path}: {e}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[INFO] Processed {video_path} in {elapsed_time:.2f} seconds")
            return 0
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n[INFO] Processed {video_path} in {elapsed_time:.2f} seconds")
    return 1

def clip_video_dir(video_dir, output_dir, interval=[15, 45]):
    """
    遍历视频目录，处理所有视频文件
    
    Args:
        video_dir (str): 视频目录
        output_dir (str): 输出目录
        interval (list): 截图间隔 单位帧 对应FPS=30 如果FPS更高则interval=FPS/30*1.7
    """
    output_root = os.path.dirname(output_dir)
    json_file = os.path.join(output_root, "processed_list.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            processed_list = json.load(f)
    processed_list = get_processed_list(output_root)
    for v in processed_list:
        print(f"[INFO] {v} already processed")
    
    with open(json_file, "w") as f:
        json.dump(processed_list, f, indent=2)
    
    
    if "images" not in output_dir:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    output_dir = os.path.join(output_dir, "images")
    
    
    
    for file in os.listdir(video_dir):
        if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mkv"):
            video_path = os.path.join(video_dir, file)
            video_filename = os.path.basename(video_path)
            video_name, _ = os.path.splitext(video_filename)
            if video_name in processed_list:
                print(f"[INFO] {video_name} already processed")
                continue
            print(f"[INFO] {video_path} Processing")
            
            result = clip_video(video_path, output_dir, interval)
            
            if result == 0:
                print(f"[ERROR] Failed to process {video_path}")
            elif result == -1:
                print(f"[INFO] {video_path} already processed")
            else:
                print(f"[INFO] {video_path} Completed\n")

def get_processed_list(output_root):
    """
    获取已处理的视频列表
    
    Args:
        output_root (str): 输出目录根目录
    
    Returns:
        list: 已处理的视频列表
    """
    processed_list = []
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith(".jpg"):
                video_name = file.rsplit("_", 2)[0]
                if video_name not in processed_list:
                    processed_list.append(video_name)
    return processed_list

if __name__ == "__main__":
    # video_path = r"G:\Videos\Call of Duty  Black Ops 6\PVP\Desktop 2025.04.08 - 21.39.09.01.mp4"
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Clip.py <batch name>")
        print("Example: python Clip.py 20250523")
        sys.exit(1)
    batch_date = sys.argv[1]
    
    game = "Call of Duty  Black Ops 6"
    
    # video_dir = rF"M:\Videos\Call of Duty  Black Ops 6\PVP"
    # output_dir = rf"F:\TrainingData\Human\COD\{batch}"
    # os.makedirs(output_dir, exist_ok=True)
    # GameMode = os.path.basename(video_dir)
    # batch = f"{batch}-{GameMode}"
    # clip_video_dir(video_dir, output_dir, [2, 40])
    
    # video_dir = r"M:\Videos\Call of Duty  Black Ops 6\Zombie"
    # output_dir = r"F:\TrainingData\Human\COD\20250518-Zombie"
    # GameMode = os.path.basename(video_dir)
    # batch = f"{batch}-{GameMode}"
    # clip_video_dir(video_dir, output_dir, [30, 90])
    
    game = "Apex Legends"
    
    video_dir = rF"M:\Videos\Apex Legends\BigMap"
    GameMode = os.path.basename(video_dir)
    batch = f"{batch_date}-{GameMode}"
    output_dir = rf"F:\TrainingData\Human\Apex\{batch}"
    os.makedirs(output_dir, exist_ok=True)
    clip_video_dir(video_dir, output_dir, [10, 60])
    
    video_dir = rF"M:\Videos\Apex Legends\SmallMap"
    GameMode = os.path.basename(video_dir)
    batch = f"{batch_date}-{GameMode}"
    output_dir = rf"F:\TrainingData\Human\Apex\{batch}"
    os.makedirs(output_dir, exist_ok=True)
    clip_video_dir(video_dir, output_dir, [2, 30])
    
    video_dir = rF"M:\Videos\Apex Legends\TestingField"
    GameMode = os.path.basename(video_dir)
    batch = f"{batch_date}-{GameMode}"
    output_dir = rf"F:\TrainingData\Human\Apex\{batch}"
    os.makedirs(output_dir, exist_ok=True)
    clip_video_dir(video_dir, output_dir, [2, 45])
    
    video_dir = rF"M:\Videos\Apex Legends\BattleScene"
    GameMode = os.path.basename(video_dir)
    batch = f"{batch_date}-{GameMode}"
    output_dir = rf"F:\TrainingData\Human\Apex\{batch}"
    os.makedirs(output_dir, exist_ok=True)
    clip_video_dir(video_dir, output_dir, [0, 12])
    
    
    
    
    
    
