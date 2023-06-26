import cv2
thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set width
cap.set(4, 480)  # set height
cap.set(10, 128)  # set default brightness value

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = 'UNKNOWN'
            if classId - 1 < len(classNames):
                className = classNames[classId - 1]
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Add text box with object name
            cv2.rectangle(img, (0, 0), (300, 50), (0, 0, 0), -1)
            cv2.putText(img, className.upper(), (10, 35),
                        cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

            break  # add break statement to exit loop after the first detection

    cv2.imshow("Output", img)
    cv2.waitKey(1)