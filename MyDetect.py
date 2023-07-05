import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


@smart_inference_mode()
def run(
        weights=ROOT / 'best.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'target/target.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    im = letterbox(source, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    
    # imread
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=augment, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1        
        im0 = source.copy()
        # p = Path("title")  # to Path

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
        Antenna_xyxy = None
        Antenna_conf = 0
        Note_xyxy = None
        Note_conf = 0
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if c == 0 and conf > Antenna_conf:
                    Antenna_xyxy = xyxy
                elif c == 1 and conf > Note_conf:
                    Note_xyxy = xyxy                    
                
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()                
                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                # annotator.box_label(xyxy, label, color=colors(c, True))
                
        # Stream results
        # im0 = annotator.result()
        # if view_img:
        #     if platform.system() == 'Linux' and p not in windows:
        #         windows.append(p)
        #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(0)  # 1 millisecond
        #     cv2.imwrite('result.jpg', im0)

    # Print time
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        
    return Antenna_xyxy, Note_xyxy

# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))


def MyDetect(img):
    Antenna_xyxy, Note_xyxy = run(source = img, imgsz=(640,640), update=False)
    
    crop_img = None
    Antenna_pos = None
    Note_pos = None

    if Antenna_xyxy is not None:
        
        img_height, img_width, _ = img.shape
        x_min = int(np.array(Antenna_xyxy[0]))
        y_min = int(np.array(Antenna_xyxy[1]))
        x_max = int(np.array(Antenna_xyxy[2]))
        y_max = int(np.array(Antenna_xyxy[3]))
        delta_x = int((x_max - x_min) * 0.1)
        delta_y = int((y_max - y_min) * 0.1)
        
        # 确定切割区域
        x_min = np.max((x_min - delta_x, 0))
        x_max = np.min((x_max + delta_x, img_width))
        y_min = np.max((y_min - delta_y, 0))
        y_max = np.min((y_max + delta_y, img_height))
        crop_img = img[y_min:y_max, x_min:x_max]
                
        # 显示切割后的图片
        # cv2.imwrite('crop0.jpg', crop_img)
        # print([x_min, x_max, y_min, y_max])
        Antenna_pos =  [x_min, x_max, y_min, y_max]

        if Note_xyxy is not None:
            # Note_pos = [int(np.array(Note_xyxy[0])), int(np.array(Note_xyxy[2])), int(np.array(Note_xyxy[1])), int(np.array(Note_xyxy[3]))]
            Note_x_min = np.max((int(np.array(Note_xyxy[0])) - x_min, 0))
            Note_x_max = np.min((int(np.array(Note_xyxy[2])) - x_min, x_max - x_min))
            Note_y_min = np.max((int(np.array(Note_xyxy[1])) - y_min, 0))
            Note_y_max = np.min((int(np.array(Note_xyxy[3])) - y_min, y_max - y_min))
            
            Note_pos = [Note_x_min, Note_x_max, Note_y_min, Note_y_max]
        
    return crop_img, Antenna_pos, Note_pos
    
if __name__ == '__main__':
    img = cv2.imread('./data/Antenna/images/train/1.jpg')
    crop_img, Antenna_pos, Note_pos = MyDetect(img)
    cv2.imwrite('crop.jpg', crop_img)
    print(Antenna_pos, Note_pos)