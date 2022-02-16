# RoadDamageDetector_YOLOv5_videoRealtime
YOLOv5 model for Road Damage Detector using OAK-D DepthAI camera
https://shop.luxonis.com/products/1098obcenclosure

## Training models

The models that we have trained use a private dataset about Vietnamese streets, and contain 4 labels :
- D00 
- D01
- D02
- D03

Trained model (contain YOLOv5s and YOLOv5x):
https://drive.google.com/drive/folders/1R07RYF7ZFBByc_3cRjY1-PUhHtiQiXCs?usp=sharing

## Export your model 

As the models have to be exported to OpenVINO IR in a certain way, we provide the tutorial on training and exporting:
- YoloV5: YoloV5_training.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb)

### Things that you need to adjust 
1. Size of images in training and exporting to ONNX model should be the same.
- Example : In this guide, we training YOLOv5s  model (you can pick whatever YOLOv5 version you want)

    - Training YOLOv5s model
    ```
    !python train.py --img 416 --batch 32 --epochs 100 --data data/custom.yaml --weights yolov5s.pt --cache 
    ```
    - Exporting to ONNX model
    ```
    !python export.py --weights $weights_path --img 416 --batch 1 --device cpu --include "onnx" --simplify
    ```
2. During training, YOLOv5 might sometimes automatically compute anchors. Because dataset (road damages) is different from COCO on with original YOLOv5 is trained.
If you see lines similar to this in log file, that mean the anchors are change
```
AutoAnchor: 3.86 anchors/target, 0.847 Best Possible Recall (BPR). Anchors are a poor fit to dataset ⚠️, attempting to improve...
AutoAnchor: WARNING: Extremely small objects found. 14 of 3499 labels are < 3 pixels in size.
AutoAnchor: Running kmeans for 9 anchors on 3499 points...
AutoAnchor: Evolving anchors with Genetic Algorithm: fitness = 0.7225: 100% 1000/1000 [00:01<00:00, 767.78it/s]
AutoAnchor: thr=0.25: 0.9991 best possible recall, 4.71 anchors past thr
AutoAnchor: n=9, img_size=416, metric_all=0.319/0.723-mean/best, past_thr=0.474-mean: 77,6, 23,26, 66,21, 156,11, 32,53, 296,9, 58,51, 343,20, 169,60
AutoAnchor: New anchors saved to model. Update model *.yaml to use these anchors in the future.
```
3. Another way to know masks of the trained model
```
%cd yolov5/
```
```
import torch
import numpy as np
from models.experimental import attempt_load


model = attempt_load(weights_path_2, map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
for i in range(3):
  for j in range(3):
    print(np.round(m.anchor_grid[i][0, j, 0, 0].numpy()))
```

## Usage
1. Running this following command in the terminal 
```
https://github.com/tiendatAI/RoadDamageDetector_YOLO_realtime
```
2. Install requirements
```
python3 -m pip install -r requirements.txt
```
3. Run the script
```
python3 realtime_camera_YOLO.py <video_path> <model_path> <config_json> --verbose
``` 

## JSONs
We already provide some JSONs for common Yolo versions. You can edit them and set them up for your model, as described in the next steps section in the mentioned tutorials. In case you are changing some of the parameters in the tutorial, you should edit the corresponding parameters. In general, the settings in the JSON should follow the settings in the CFG of the model. For YoloV5, the default settings should be the same as for YoloV3.

Note: Values must match the values set in the CFG during training. If you use a different input width, you should also change side8 to sideX, side16 to sideY and side32 to sideZ, where X = width/8, Y = width/16 and Z = width/32.

Example : With 416x416 images, we have side52, side26, side13

You can also change IOU and confidence thresholds. Increase the IOU threshold if the same object is getting detected multiple times. Decrease confidence threshold if not enough objects are detected. Note that this will not magically improve your object detector, but might help if some objects are filtered out due to the threshold being too high.[1]

In particular, you should edit anchors of your model in config file.

## Reference
1. https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo/device-decoding
2. https://docs.luxonis.com/projects/api/en/latest/samples/MobileNet/video_mobilenet/

Special thanks to @Waterfool in Luxonis Community
