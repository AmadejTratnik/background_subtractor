#!/usr/bin/python

from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A  # we need this for data augmentation
import torchvision.transforms.functional as F
import time
import psutil #type: ignore
import onnxruntime
import torch.onnx
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

transform = A.Compose(
    [
        A.Resize(height=192, width=320),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


background_img_path = "images/background_image.png"
cap= cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,IMAGE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,60)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

ort_session = onnxruntime.InferenceSession("model/optimized_person_segmentation_180x320.onnx")
print("starting onnxruntime inference session...")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

mean_frames = 0
num_frames = 0


img_background = cv2.imread(background_img_path, cv2.IMREAD_COLOR)
i=0
all_frames = []
time_mean = 0

while True:
    ret, img = cap.read()    
    num_frames += 1
    # Capture frame-by-frame
    

    new_frame_time = time.time()
    im_pil = np.array(Image.fromarray(img).convert("RGB"))
    augmentation = transform(image=im_pil)
    image = augmentation["image"]
    image = torch.unsqueeze(image, 0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    preds = torch.from_numpy(img_out_y)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.7).float()
    # preds = np.array(preds)

    preds = torch.squeeze(preds, 0)
    preds = F.to_pil_image(preds)

    open_cv_mask = np.array(preds)

    fps = 1 / (new_frame_time - prev_frame_time)
    all_frames.append(fps)
    t = new_frame_time - prev_frame_time
    prev_frame_time = new_frame_time
    mean_frames += int(fps)
    fin_fps = int(fps)
    fps = "FPS:" + str(int(fps))

    open_cv_mask = cv2.resize(open_cv_mask, (IMAGE_WIDTH,IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

    dim = (open_cv_mask.shape[1], open_cv_mask.shape[0])
    img_f = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img_background = cv2.resize(img_background, dim, interpolation=cv2.INTER_AREA)
    mapping = cv2.cvtColor(open_cv_mask, cv2.COLOR_GRAY2RGB)
    layered_image = np.where(mapping != (0, 0, 0), img_f,img_background )
    cv2.putText(layered_image, fps, (7, 70), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
    final = cv2.hconcat([img_f, mapping, layered_image])
    cv2.imshow("ORIGINAL | MAPPING | LAYERED IMAGE", final)
    number = "{:04d}".format(i)

    cpu_per = psutil.cpu_percent()
    print(f'FPS: [{mean_frames//num_frames}] | TIME: [{(int(t*1000))}ms] | CPU: [{psutil.cpu_percent()}%]',)
    data = [i+1000,fin_fps,cpu_per]
    i+=1
    if(i == 1000):
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
