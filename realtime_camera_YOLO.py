#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
        

def main(args):
    if args.verbose:
        print("Input arguments: ", args)
    
    # check input exists
    videoPath = args.input_video
    if not Path(videoPath).exists():
        raise ValueError("Path {} does not exist!".format(videoPath))

    modelPath = args.model
    if not Path(modelPath).exists():
        raise ValueError("Path {} does not exist!".format(modelPath))
    
    #parse config 
    configPath = Path(args.config)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    with configPath.open() as f:
        config = json.load(f)
    nnConfig = config.get("nn_config", {})

    # parse input shape
    if "input_size" in nnConfig:
        W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

    # extract metadata
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    if args.verbose:
        print(metadata)

    # parse labels
    nnMappings = config.get("mappings", {})
    labels = nnMappings.get("labels", {})

    #create pipeline
    pipeline = dai.Pipeline()

    #define sources and outputs
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xinFrame = pipeline.create(dai.node.XLinkIn)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xinFrame.setStreamName("inputFrame")
    nnOut.setStreamName("nn")

    # Network specific settings
    nn.setConfidenceThreshold(confidenceThreshold)
    nn.setNumClasses(classes)
    nn.setCoordinateSize(coordinates)
    nn.setAnchors(anchors)
    nn.setAnchorMasks(anchorMasks)
    nn.setIouThreshold(iouThreshold)
    nn.setBlobPath(modelPath)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    #linking
    xinFrame.out.link(nn.input)
    nn.out.link(nnOut.input)

    #connect to device and start pipeline 
    with dai.Device(pipeline) as device:
        # Input queue will be used to send video frames to the device.
        qIn = device.getInputQueue(name="inputFrame")
        # Output queue will be used to get nn data from the video frames.
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []

        def displayFrame(name, frame):
            color = (0, 255, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                #print out label 
                if args.verbose:
                    print("Label detected: " + labels[detection.label] + " - " + str(detection.confidence))
                    print("Coordinate:" + str(detection.xmin) + " "
                                        + str(detection.ymin) + " " 
                                        + str(detection.xmax) + " "
                                        + str(detection.ymax))

                cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Show the frame
            frame = cv2.resize(frame, (1028, 640))
            cv2.imshow(name, frame)
        
        #display on screen
        cap = cv2.VideoCapture(args.input_video)

        # used to record the time when we processed last and current frame
        prev_frame_time = 0
        new_frame_time = 0

        while cap.isOpened():
            read_correctly, frame = cap.read()
            if not read_correctly:
                break
            
            #calculate and put fps 
            new_frame_time = time.time()
            fps = int(1/(new_frame_time-prev_frame_time))
            prev_frame_time = new_frame_time
            cv2.putText(frame, "FPS: " + str(fps), (20, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 3)
            
            #send frame to camera
            in_frame = to_planar(frame, (416, 416))
            img = dai.ImgFrame()
            img.setData(in_frame)
            img.setTimestamp(time.monotonic())
            img.setWidth(416)
            img.setHeight(416)
            qIn.send(img)

            inDet = qDet.tryGet()

            if inDet is not None:
                detections = inDet.detections

            if frame is not None:
                displayFrame("Realtime", frame)

            if cv2.waitKey(1) == ord('q'):
                break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Realtime OAK-D")
    parser.add_argument("input_video", metavar="<input_video_path>", help="Path to input realtime video")
    parser.add_argument("model", metavar="<input_model_path>", help="Path to .blob file model")
    parser.add_argument("config", metavar="<config_path>", help="Path to .json config file")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print verbose output")

    args = parser.parse_args()
    ret = main(args)
    print("Closed realtime video")
    exit(ret)