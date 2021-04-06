import cv2
from tracker import *

tracker = EuclideanDistTracker()

videoFile = "test_22.mp4"
cap = cv2.VideoCapture(videoFile)

startX = 0

startY = 0

prevPoint = [0,0]

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200)

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def makeImage(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file

    for item in trackedPoints:
        cv2.drawMarker(image, (startX+int(item[0]), startY+int(item[1])),(0,255,0), markerType=cv2.MARKER_DIAMOND, 
        markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        prevPoint = [startX+int(item[0]), startY+int(item[1])]
        

    print(int(item[0]), int(item[1]))

    cv2.imwrite("final_frame.jpg",image)


trackedPoints = []

while (cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    roi = frame[startY:2000,startX:2000]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 10:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+w), (0,255,0), 3)
            detections.append([x,y,w,h])
            trackedPoints.append([x+w/2,y+h/2])

    if(len(detections) != 0):
        print(detections[-1][0]+detections[-1][2]/2,',',cap.get(4)-detections[-1][1]-detections[-1][3]/2)
        #trackedPoints.append([detections[-1][0]+detections[-1][2]/2,detections[-1][1]+detections[-1][3]/2])
    #cv2.imshow("Frame", frame)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("ROI", rescale_frame(roi))
    cv2.imshow("Mask", rescale_frame(mask))

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

makeImage(videoFile)

cap.release()
cv2.destroyAllWindows()
