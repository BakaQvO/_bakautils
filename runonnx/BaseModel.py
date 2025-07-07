import onnxruntime as ort
import os
import numpy as np
import json
from PIL import Image
import cv2
import scipy


"""
Json Format
{
    "model_path": "path",
    "model_format": "onnx",
    "head_type": "dflobj", # or "v5", "dfl", "dflobj"
    "reg_max": 16,
    "platform": "cpu", # or "cuda", "gpu"
    "input":{
        "input_name_0": [shape],
    },
    "output":{
        "output_name_0": [shape],
        ...
    }
    "class_map": [
        "class_name_0",
    ]
}

"""

class BaseModel():
    def __init__(self, config_path: str=None, debug: bool=False):
        self.debug = debug
        if config_path is None:
            raise ValueError(f"Config Not Found {config_path}")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        input = self.config.get("input", {})
        if not input:
            raise ValueError("Input not found in config")
        self.input_names = list(input.keys())
        self.input_shapes = input
        
        output = self.config.get("output", {})
        if not output:
            raise ValueError("Output not found in config")
        self.output_names = list(output.keys())
        self.output_shapes = output
        
        self.model_path = self.config.get("model_path", "")
        self.platform = self.config.get("platform", "gpu")
        
        self.head_type = self.config.get("head_type", "dfl") #v5, dfl, dflobj
        if "dfl" in self.head_type:
            self.reg_max = self.config.get("reg_max", 16)
        
        self.class_map = self.config.get("class_map", [])
        
        self.__init_onnxruntime__()
        


    def __init_onnxruntime__(self):
        platform = self.platform
        assert self.config.get("model_path", "").endswith(".onnx")
        if platform == "cpu":
            self.session = ort.InferenceSession(self.model_path, providers = ["CPUExecutionProvider"])
        else:
            so = ort.SessionOptions()
            if platform == "cuda" or platform == "gpu":
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    })
                ]
            self.session = ort.InferenceSession(self.model_path, so, providers = providers)
            print(self.session.get_providers())
            if not self.session.get_providers()[0] in ["CUDAExecutionProvider", "DmlExecutionProvider"]:
                print("warning: not run in GPU")
    
        
    def _preprocess(self, input, dst_shape: tuple=None):
        """
        预处理输入数据，将其转换为模型所需的格式
        Args:
            input: 输入数据，可以是文件路径、PIL图像或numpy数组(RGB)
            dst_shape: 目标形状，默认为模型输入形状(H,W)
        Returns:
            img: 原始图像数据
            img_: 预处理后的图像数据，形状为 (1, C, H, W)
        """
        if isinstance(input, str):
            img = cv2.imread(input)
        elif isinstance(input, Image.Image):
            img = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
        elif isinstance(input, np.ndarray):
            if input.ndim == 2:
                img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            elif input.ndim == 3 and input.shape[2] == 3:
                img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input must be a file path or PIL Image")
        
        if dst_shape is None:
            print(f"[INFO] [BaseModel] [_preprocess] dst_shape is None, using model input shape {self.input_shapes[self.input_names[0]]}")
            dst_shape = (self.input_shapes[self.input_names[0]][2], self.input_shapes[self.input_names[0]][1])  # (W, H)
        img_ = img.copy()
        self.input_img = img.copy()  # 保存原始图像
        img_ = cv2.resize(img_, dst_shape)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        img_ = np.expand_dims(img_, 0).astype(np.float32) / 255.0 # 1 h w c
        img_ = np.transpose(img_, (0, 3, 1, 2)) # N C H W
        return img, img_
    
    def preprocess(self, input):
        # dst_shape = [self.input_shapes[self.input_names[0]][1], self.input_shapes[self.input_names[0]][2]]
        # if self.debug:
        #     print(f"[INFO] [BaseModel] [preprocess] dst_shape: {dst_shape}")
        img, img_ = self._preprocess(input, dst_shape=None)
        return img, img_
    
    def draw_bbox(self, img, bbox, color): # bbox = [idx, class, conf, x1, y1, x2, y2]
        cls = int(bbox[1])
        pt1 = (int(bbox[3] * img.shape[1]), int(bbox[4] * img.shape[0]))
        pt2 = (int(bbox[5] * img.shape[1]), int(bbox[6] * img.shape[0]))
        clr = (int((color[0]+cls*67)%200), int((color[1]+cls*134)%200), int((color[2]+cls*181)%200))
        cv2.rectangle(img, pt1, pt2, clr, 2)
        if self.class_map:
            cls = self.class_map[cls]
        cv2.putText(img, f"{cls} {bbox[2]:.2f}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        return img
    
    def area(self, bbox): # x1, y1, x2, y2
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return 0

    def bbox_iou(self, bbox1, bbox2): # bbox = [idx, class, conf, x1, y1, x2, y2]
        x1 = max(bbox1[3], bbox2[3])
        y1 = max(bbox1[4], bbox2[4])
        x2 = min(bbox1[5], bbox2[5])
        y2 = min(bbox1[6], bbox2[6])
        if x1 >= x2 or y1 >= y2:
            return 0
        inter = self.area([x1, y1, x2, y2])
        union = self.area(bbox1[3:]) + self.area(bbox2[3:]) - inter
        if inter > 0:
            return inter / union
        else:
            assert inter == 0
            return 0
    
    def bbox_ioA(self, bbox1, bbox2):
        x1 = max(bbox1[3], bbox2[3])
        y1 = max(bbox1[4], bbox2[4])
        x2 = min(bbox1[5], bbox2[5])
        y2 = min(bbox1[6], bbox2[6])
        if x1 >= x2 or y1 >= y2:
            return 0
        area_i = self.area([x1, y1, x2, y2])
        area_a = self.area(bbox1[3:])
        if area_a > 0:
            return area_i / area_a
        else:
            assert area_a == 0
            return 0
    
    def nms(self, bboxs, iou = 0.45):
        # bbox = [idx, class, conf, x1, y1, x2, y2]
        bboxs = sorted(bboxs, key=lambda x: x[2], reverse=True)  # Sort by confidence
        bboxs_ = bboxs.copy()
        bboxs = []
        for bbox in bboxs_:
            flag = True
            for i in bboxs:
                if i[1] != bbox[1]:
                    continue
                if self.bbox_iou(bbox, i) > iou:
                    flag = False
                    break
            if flag:
                bboxs.append(bbox)
        return bboxs
    
    def cls_based_t_iou(self, cls, t_iou):
        if (cls == 0 or cls == 1 or
            cls == 10 or cls == 11 or cls == 12):
            return 1.4*t_iou
        else:
            return t_iou
    
    def softnms(self, bboxs:list[list[float]], t_iou=0.45, t_det=0.25, sigma=0.5):
        bboxs = sorted(bboxs, key=lambda x: x[2], reverse=True)  # Sort by confidence
        bboxs_ = bboxs.copy()
        bboxs = []
        i = 0
        while i < len(bboxs_):
            bbox_select = bboxs_[i]
            if bbox_select[2] == 0: break
            j = i+1
            while j < len(bboxs_):
                bbox_view = bboxs_[j]
                if bbox_view[1] != bbox_select[1]:
                    j += 1
                    continue
                ioa = self.bbox_ioA(bbox_select, bbox_view)
                if ioa <= 0:
                    j += 1
                    continue
                
                if ioa > self.cls_based_t_iou(bbox_view[1], t_iou):
                    del bboxs_[j]
                    continue
                bbox_view[2] *= np.exp(-(ioa*ioa)/(2*sigma*sigma))
                # if bbox_view[2] < t_det:
                #     bbox_view[2] = 0
                #     del bboxs_[j]
                #     continue
                bboxs_[j] = bbox_view
                j += 1
            bboxs_ = bboxs_[:i] + sorted(bboxs_[i:], key=lambda x: x[2], reverse=True)
            i += 1
        # bboxs = [bbox for bbox in bboxs_ if bbox[2] >= t_det]
        return bboxs_
    
    
        
    
    def ltrb2xyxy(self, dbox, anchor_points=None):
        """
        将ltrb格式的边界框转换为xyxy格式
        Args:
            dbox: 边界框数据，格式为 [left, top, right, bottom] 或者 (N,H,W,4)
            anchor_points: (2,) 或 (H,W,2) 的anchor点坐标，表示gridcell的中心点
        Returns:
            Decoded bounding box in the format [x1, y1, x2, y2] or (N, H, W, 4) tensor.
        """
        if anchor_points is None:
            raise ValueError("anchor_points must be provided for decoding boxes")
        dbox = np.array(dbox, dtype=np.float32)
        anchor_points = np.array(anchor_points, dtype=np.float32)
        
        dbox_shape = dbox.shape
        
        lt, rb = dbox[..., :2], dbox[..., 2:]
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        
        xyxy = np.concatenate([x1y1, x2y2], axis=-1)  # (N, H, W, 4) or (4,)
        return xyxy.reshape(dbox_shape)
        
        
    
    def generate_anchor_points(self, f_h:int, f_w:int, grid_cell_offset=0.5):
        anchor_points_x = np.arange(f_w) + grid_cell_offset
        anchor_points_y = np.arange(f_h) + grid_cell_offset
        anchor_points_y, anchor_points_x = np.meshgrid(anchor_points_y, anchor_points_x, indexing='ij')
        anchor_point = np.stack([anchor_points_x, anchor_points_y], axis=-1)
        # print(f"anchor_point shape : {anchor_point.shape}")
        # print(f"anchor_point : {anchor_point}")
        return anchor_point # (h, w, 2)
    
    def split_output_dflobj(self, outputs, dims=-1):
        """
        在dims维度上分割输出张量
        Args:
            outputs: The raw output from the model, (N, H, W, C) tensor.
        Returns:
            [reg_max*4],[obj],[nc] : (N, H, W, reg_max*4), (N, H, W, 1), (N, H, W, nc) tensors.
        """
        nbox = self.reg_max * 4
        nc = len(self.class_map)
        
        if not isinstance(outputs, np.ndarray):
            raise TypeError(f"Expected outputs to be a numpy array, got {type(outputs)}")
        if outputs.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {outputs.ndim}D tensor")
        if outputs.shape[-1] != nbox + 1 + nc:
            raise ValueError(f"Expected last dimension to be {nbox + 1 + nc}, got {outputs.shape[-1]}")
        
        dboxes, obj, cls = np.split(outputs, [nbox, nbox + 1], axis=dims)
        return dboxes, obj, cls
    
    def decode_bbox_dfl(self, data, anchor_points=None):
        """
        Decode DFL (Distribute Focal Loss) bounding boxes.
        Args:
            data: The raw output from the model, typically a tensor of shape (N, H, W, reg_max * 4).
            anchor_points: Optional, precomputed anchor points for the feature map.
        Returns:
            Decoded bounding boxes in the format [x1, y1, x2, y2].
        """
    
    def decode_single_box_dfl(self, dbox, anchor_points=None):
        """
        解码单个reg_max * 4格式的边界框
        Args:
            dbox: 单个边界框数据，形状为 (reg_max * 4, )
            anchor_points: 单个gridcell对应的anchor (2, )
        Returns:
            Decoded bounding box in the format [x1, y1, x2, y2].
        """
        dbox = np.array(dbox, dtype=np.float32).reshape(4, self.reg_max)
        if anchor_points is None:
            raise ValueError("anchor_points must be provided for decoding DFL boxes")
        anchor_points = np.array(anchor_points, dtype=np.float32)
        
        dbox = scipy.special.softmax(dbox, axis=-1)
        dbox = np.sum(dbox * np.arange(self.reg_max, dtype=np.float32), axis=-1)  # (4, )
        
        bbox = self.ltrb2xyxy(dbox, anchor_points)
        
        return bbox
        
    
    def _postprocess(self, outputs, threshold=0.25, iou=0.45, do_nms=True):
        """
        解码模型原始输出
        Args:
            outputs: 模型输出 3个尺度 3x(N, H, W, C) tensor 或者 3x(NHWC, )
        Returns:
            bboxs: 解码后的边界框列表，每个边界框格式为 [idx, class, conf, x1, y1, x2, y2]
        """
        threshold_logit = scipy.special.logit(threshold)
        print("[DEBUG] [BaseModel] [_postprocess] threshold/logit:", threshold, threshold_logit)
        bboxs = []
        anchors = [[],[],[]]
        for idx, output in enumerate(outputs):
            output = np.asarray(output, dtype=np.float32)
            output_name = self.output_names[idx]
            output_shape = self.output_shapes[output_name]
            
            if anchors[idx] == []:
                f_h, f_w = output_shape[2], output_shape[3]
                anchors[idx] = self.generate_anchor_points(f_h, f_w)
            
            # reshape
            if output.ndim != 4:
                try:
                    output = output.reshape(output_shape)
                except Exception as e:
                    raise ValueError(f"Failed to reshape output {output_name} to {output_shape}: {e}")
            # if tuple(output.shape) != tuple(output_shape):
            #     raise ValueError(f"Output shape mismatch for {output_name}: expected {output_shape}, got {output.shape}")
            
            # split
            dboxes_o, obj_o, cls_o = self.split_output_dflobj(output)
            
            mask = obj_o > threshold_logit
            mask = np.asarray(mask).squeeze(-1)
            if self.debug:
                print(f"Processing output {output_name} with shape {output.shape}, Mask sum: {np.sum(mask)}")
            if not np.any(mask):
                continue
            
            dboxes_m = dboxes_o[mask]
            cls_m = cls_o[mask]
            obj_m = obj_o[mask]
            anchors_m = anchors[idx][mask[0]]
            
            for i in range(obj_m.shape[0]):
                cls_conf = np.max(cls_m[i])
                if cls_conf < threshold_logit:
                    if self.debug:
                        # print(f"[INFO] [BaseModel] [_postprocess] Skipping box {i} with low conf obj/cls/threshold_logit: {obj_m[i][0]:.4f}/{cls_conf:.4f}/{threshold_logit:.4f}")
                        pass
                    continue
                cls_conf = scipy.special.expit(cls_conf)
                cls_idx = np.argmax(cls_m[i])
                dbox = self.decode_single_box_dfl(dboxes_m[i], anchors_m[i])
                
                bboxs.append([
                    len(bboxs),  # idx
                    cls_idx,  # class
                    cls_conf,  # conf
                    dbox[0] / output.shape[2],
                    dbox[1] / output.shape[1],
                    dbox[2] / output.shape[2],
                    dbox[3] / output.shape[1],
                ])
        if do_nms:
            bboxs = self.nms(bboxs, iou)
            # bboxs = self.softnms(bboxs, iou)
        
        bboxs = np.array(bboxs, dtype=np.float32)
        
        if self.debug:
            print(f"[INFO] [BaseModel] [_postprocess] Detected {len(bboxs)} bounding boxes after NMS")
            # print(f"[INFO] [BaseModel] [_postprocess] Bounding boxes: {bboxs}")
        
        return bboxs
    
    def postprocess(self, outputs, threshold=0.25, iou=0.45):
        """
        解码模型原始输出
        Args:
            outputs: 模型输出 3个尺度 3x(N, H, W, C) tensor 或者 3x(NHWC, )
        Returns:
            bboxs: 解码后的边界框列表，每个边界框格式为 [idx, class, conf, x1, y1, x2, y2]
        """
        bboxs = self._postprocess(outputs, threshold, iou)
        return bboxs
    
    def draw_bboxs(self, img, bboxs,color=(0,0,255)):
        for idx, bbox in enumerate(bboxs):
            pt1 = (int(bbox[3] * img.shape[1]), int(bbox[4] * img.shape[0]))
            pt2 = (int(bbox[5] * img.shape[1]), int(bbox[6] * img.shape[0]))
            clr = (int((color[0]+int(bbox[1])*67)%200), int((color[1]+int(bbox[1])*134)%200), int((color[2]+int(bbox[1])*181)%200))
            # cv2.rectangle(img, (pt1[0]+1, pt1[1]+1), (pt2[0]-1, pt2[1]-1), (255,255,255), 1)
            cv2.rectangle(img, pt1, pt2, clr, 1)
            cls = int(bbox[1])
            conf = bbox[2]
            if self.class_map:
                cls = self.class_map[cls]
            cv2.putText(img, f"{cls} {conf:.2f}", (pt1[0]+1, pt1[1] - 10+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(img, f"{cls} {conf:.2f}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
        return img
    
    def predict(self, img):
        """
        预测函数
        Args:
            img: 输入数据 经过preprocess处理后的图像数据
        Returns:
            outputs: 模型输出 3x(N,H,W,C)
        """
        inputs = {self.input_names[0]: img}
        
        outputs = self.session.run(self.output_names, inputs) # [3x(N C H W)]
        # transpose
        outputs = [np.transpose(output, (0, 2, 3, 1)) for output in outputs]
        
        return outputs
    
    def run(self, input, output=None, threshold=0.25, iou=0.45):
        """
        运行模型预测并解码输出
        Args:
            input: 输入数据，可以是文件路径、PIL图像或numpy数组(RGB)
            output: 输出路径，如果不是"bboxs"，则保存处理后的图像到指定路径
        Returns:
            img: 处理后的图像数据，或者解码后的边界框列表
            bboxs: 解码后的边界框列表，每个边界框格式为 [idx, class, conf, x1, y1, x2, y2]
        """
        self.t_det = threshold
        img, img_ = self.preprocess(input)
        if self.debug:
            print(f"Input shape: {img_.shape}")
        
        outputs = self.predict(img_)
        if self.debug:
            print(f"Outputs shape: {[output.shape for output in outputs]}")
        
        bboxs = self.postprocess(outputs, threshold, iou)
        if self.debug:
            print(f"Detected {len(bboxs)} bounding boxes")
        
        if output == "bboxs":
            return bboxs
        
        img = self.input_img.copy()
        img = self.draw_bboxs(img, bboxs)
        
        if isinstance(output, str):
            output_dir = os.path.dirname(output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(output, img)
        
        return img
            
            