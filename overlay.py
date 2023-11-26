#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import random
import numpy as np

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
            

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO("/home/lab/yolo_ws/src/Yolov8_ros/yolov8_ros/weights/seg_0924.pt")
# print(model)
class_names = model.names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

def overlay_seg(image):
    img = cv2.imread(image)
    h, w, _ = img.shape
    results = model.predict(img, stream=True)
    # print(results)
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    if masks is not None:
        masks = masks.data.cpu()
    for seg, box in zip(masks.data.cpu().numpy(), boxes):
        seg = cv2.resize(seg, (w, h))
        img = overlay(img, seg, colors[int(box.cls)], 0.4)
        
        xmin = int(box.data[0][0])
        ymin = int(box.data[0][1])
        xmax = int(box.data[0][2])
        ymax = int(box.data[0][3])
        cls = 'Unknown class'
        if int(box.cls) == 0:
            cls = 'gap'
            left, right = find_top_points(seg)
            cv2.circle(img, left, radius=1, color=(0, 255, 0), thickness=-1)  # Green circle for top-left
            cv2.circle(img, right, radius=1, color=(0, 0, 255), thickness=-1)  # Red circle for top-right
        elif int(box.cls) == 1:
            cls =  'pc'
        else:
            cls = f'Unknown class {int(box.cls)}'  # Optional, for other class indices

        plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{cls} {float(box.conf):.3}')
    
    new_string = image.replace('.png', '')
    cv2.imwrite(f'{new_string}_seg.png',img)

def find_top_points(mask):
    # Step 1: Find the top row that contains the mask
    top_row_index = np.min(np.where(mask > 0)[0])
    
    # Step 2: Extract the top row from the mask
    top_row = mask[top_row_index, :]
    
    # Step 3: Find the top-left and top-right x coordinates
    top_left_x = np.min(np.where(top_row > 0))
    top_right_x = np.max(np.where(top_row > 0))
    
    # Step 4: Construct and return the points
    top_left_point = (top_left_x, top_row_index)
    top_right_point = (top_right_x, top_row_index)
    
    print(f"Top Left Point: {top_left_point}")
    print(f"Top Right Point: {top_right_point}")
    
    return top_left_point, top_right_point


if __name__ == '__main__':
    #pass
    overlay_seg("test001.png")