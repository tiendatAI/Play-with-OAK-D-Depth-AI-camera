# RoadDamageDetector_YOLO_realtime
YOLOv5 model for Road Damage Detector using OAK-D DepthAI camera

## Export your model 

As the models have to be exported to OpenVINO IR in a certain way, we provide the tutorial on training and exporting:
- YoloV5: YoloV5_training.ipynb [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb)

### Things that you need to adjust 
1. Size of images in training and exporting to ONNX model should be the same.
- Example : When training with 416x416 images


