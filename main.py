import cv2
import tkinter as tk
from PIL import Image, ImageTk

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = ['Car', 'Person', 'WheelChair']  # Your remaining class labels

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)

# Function to check if the video file path is valid
def is_valid_video(video_file):
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Open the video file if it's valid
video_file = 'vid1.mp4'
if not is_valid_video(video_file):
    print("Invalid video file path.")
    exit()

cap = cv2.VideoCapture(video_file)

paused = False

person_counter = 0
wheelchair_counter = 0

def toggle_pause():
    global paused
    paused = not paused

def video_stream():
    global paused, person_counter, wheelchair_counter
    if not paused:
        ret, frame = cap.read()
        if not ret:
            return
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
        # Reset counters for each frame
        person_counter = 0
        wheelchair_counter = 0
        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd < len(classLabels):
                    label = classLabels[ClassInd]
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if label == 'Person':
                        person_counter += 1
                    elif label == 'WheelChair':
                        wheelchair_counter += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))  # Resize the frame
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        panel.img = img
        panel.config(image=img)
        person_counter_label.config(text=f"Person Counter: {person_counter}")
        wheelchair_counter_label.config(text=f"Wheelchair Counter: {wheelchair_counter}")
    panel.after(10, video_stream)  # Call this function again after 10 ms

root = tk.Tk()
root.title("Video Player")

# Create buttons
btn_start = tk.Button(root, text="Start", command=video_stream)
btn_stop = tk.Button(root, text="Stop", command=root.quit)
btn_pause_resume = tk.Button(root, text="Pause/Resume", command=toggle_pause)

# Create labels for counters
person_counter_label = tk.Label(root, text="Person Counter: 0")
wheelchair_counter_label = tk.Label(root, text="Wheelchair Counter: 0")

# Layout buttons and labels
btn_start.grid(row=0, column=0, pady=5)
btn_stop.grid(row=1, column=0, pady=5)
btn_pause_resume.grid(row=2, column=0, pady=5)
person_counter_label.grid(row=3, column=0, pady=5)
wheelchair_counter_label.grid(row=4, column=0, pady=5)

# Create label for displaying video
panel = tk.Label(root, width=640, height=480)  # Adjust width and height as needed
panel.grid(row=0, column=1, rowspan=5, padx=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
