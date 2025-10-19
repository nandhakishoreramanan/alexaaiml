from ultralytics import YOLO
import cv2
import math
detected_obj_list = []
unique_obj_list = []
inp = input("Type source if you want to use a file or type webcam if you want to use webcam: ")
if inp == "source":
    typee = input("Enter source type(photo or video):")
    cap = cv2.VideoCapture(input("Enter the source name(should be in the same folder): "))
else:
 cap = cv2.VideoCapture(0)
 cap.set(3, 640)
 cap.set(4, 480)
model = YOLO("yolo11m.pt")
number_detected_objects = 0
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True,)
    for r in results:

        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values


            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            detected_obj_list.append(classNames[cls])
            for var in detected_obj_list:
                if var not in unique_obj_list:
                    unique_obj_list.append(classNames[cls])

            number_detected_objects += len(unique_obj_list)

            # object details
            org = [x1, y1]
            conf = [x1+150, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, str(confidence), conf, font, fontScale, color, thickness)
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    if inp == "source":
      if typee =="photo":
        print("number of unique detected objects -->", len(unique_obj_list))
        print("name of unique detected objects -->", unique_obj_list)
        cv2.imshow("Source", img)
        cv2.waitKey(0)

      else:
       cv2.imshow('Webcam', img)

       if cv2.waitKey(1) == ord('q'):
        print("number of unique detected objects -->", len(unique_obj_list))
        print("name of unique detected objects -->", unique_obj_list)

        break
    else:
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            print("number of unique detected objects -->", len(unique_obj_list))
            print("unique object list -->", unique_obj_list)
            break

cap.release()
cv2.destroyAllWindows()