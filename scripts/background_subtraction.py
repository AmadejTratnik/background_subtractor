#!/usr/bin/python

import threading
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2
import numpy as np
import albumentations as A
import torchvision.transforms.functional as F
import time
import onnxruntime

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
THRESHOLD = 0.7
N_OF_FRAMES = 1000
model_cache = {}
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
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

ort_session = onnxruntime.InferenceSession("model/optimized_person_segmentation_180x320.onnx")
print("starting onnxruntime inference session...")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def process_frame(img, img_background):
    augmentation = transform(image=np.array(img))
    image = augmentation["image"]
    image = torch.unsqueeze(image, 0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    preds = (torch.sigmoid(torch.from_numpy(img_out_y)) > THRESHOLD).to(torch.float32)
    preds = torch.squeeze(preds, 0)
    preds = F.to_pil_image(preds)

    open_cv_mask = cv2.resize(np.array(preds), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

    dim = (open_cv_mask.shape[1], open_cv_mask.shape[0])
    img_f = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img_background = cv2.resize(img_background, dim, interpolation=cv2.INTER_AREA)
    mapping = cv2.cvtColor(open_cv_mask, cv2.COLOR_GRAY2RGB)
    layered_image = np.where(mapping != (0, 0, 0), img_f, img_background)

    return layered_image,mapping

def display_frame():
    global cap, N_OF_FRAMES
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_background = cv2.imread(background_img_path, cv2.IMREAD_COLOR)
    
    prev_frame_time = 0
    i = 0
    while i < N_OF_FRAMES:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        _, img = cap.read()
        new_frame_time = time.time()

        layered_image,mapping = process_frame(img, img_background)
        
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = "FPS:" + str(int(fps))

        cv2.putText(layered_image, fps, (7, 70), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        final = cv2.hconcat([img, mapping, layered_image])
        cv2.imshow("ORIGINAL | MAPPING | LAYERED IMAGE", final)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    image_processing_thread = threading.Thread(target=display_frame)
    image_processing_thread.start()
    image_processing_thread.join() 

if __name__ == "__main__":
    main()
