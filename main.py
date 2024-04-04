import cv2

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = ['Car', 'Person', 'WheelChair']  # Your remaining class labels

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)

# Open the video file
video_file = 'vid.mp4'
cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            # Ensure ClassInd is within the range of classLabels
            if ClassInd < len(classLabels):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                label = classLabels[ClassInd]
                cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
