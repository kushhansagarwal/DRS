import cv2
import numpy as np

frontFile = "test_24.mp4"
perpFile = "test_22.mp4"

frontCap = cv2.VideoCapture(frontFile)
perpCap = cv2.VideoCapture(perpFile)

frontStartX = 800
frontStartY = 0
perpStartX = 0
perpStartY = 0
frontEndX = 1200
frontEndY = 2000
perpEndX = 2000
perpEndY = 2000  

frontDetector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200)
perpDetector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200)

frontPoints = []
perpPoints = []

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def makeFrontImage():
    vidcap = cv2.VideoCapture(frontFile)
    success, image = vidcap.read()
    image = cv2.subtract(image,np.ones(image.shape, dtype="uint8") * 100)
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file

    for item in frontPoints:
        cv2.drawMarker(image, (frontStartX+int(item[0]), frontStartY+int(item[1])),(0,255,0), markerType=cv2.MARKER_DIAMOND, 
        markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        prevPoint = [frontStartX+int(item[0]), frontStartY+int(item[1])]
        

    print(int(item[0]), int(item[1]))

    cv2.imwrite("front_final_frame.jpg",image)

def makePerpImage():
    vidcap = cv2.VideoCapture(perpFile)
    success, image = vidcap.read()
    image = cv2.subtract(image,np.ones(image.shape, dtype="uint8") * 100)
    if success:
        cv2.imwrite("first_frame.jpg", image)  # save frame as JPEG file

    for item in perpPoints:
        cv2.drawMarker(image, (perpStartX+int(item[0]), perpStartY+int(item[1])),(0,255,0), markerType=cv2.MARKER_DIAMOND, 
        markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        prevPoint = [perpStartX+int(item[0]), perpStartY+int(item[1])]
        

    print(int(item[0]), int(item[1]))

    cv2.imwrite("perp_final_frame.jpg",image)

while (frontCap.isOpened()):
    retFront, frameFront = frontCap.read()

    if not retFront:
        break

    heightFront, widthFront, _ = frameFront.shape

    roiFront = frameFront[frontStartY:frontEndY,frontStartX:frontEndX]

    maskFront = frontDetector.apply(roiFront)
    _, maskFront = cv2.threshold(maskFront, 254, 255, cv2.THRESH_BINARY)
    contoursFront,_ = cv2.findContours(maskFront,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detectionsFront = []

    for cntFront in contoursFront:
        area = cv2.contourArea(cntFront)
        if (area > 10) & (area < 50):
            x, y, w, h = cv2.boundingRect(cntFront)
            cv2.rectangle(roiFront, (x,y), (x+w, y+w), (0,255,0), 3)
            detectionsFront.append([x,y,w,h])
            frontPoints.append([x+w/2,y+h/2])

    cv2.imshow("ROI", rescale_frame(roiFront))
    cv2.imshow("Mask", rescale_frame(maskFront))

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

frontCap.release()
cv2.destroyAllWindows()

while (perpCap.isOpened()):
    retperp, frameperp = perpCap.read()

    if not retperp:
        break

    heightperp, widthperp, _ = frameperp.shape

    roiperp = frameperp[perpStartY:perpEndY,perpStartX:perpEndX]

    maskperp = perpDetector.apply(roiperp)
    _, maskperp = cv2.threshold(maskperp, 254, 255, cv2.THRESH_BINARY)
    contoursperp,_ = cv2.findContours(maskperp,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detectionsperp = []

    for cntperp in contoursperp:
        area = cv2.contourArea(cntperp)
        if area > 10:
            x, y, w, h = cv2.boundingRect(cntperp)
            cv2.rectangle(roiperp, (x,y), (x+w, y+w), (0,255,0), 3)
            detectionsperp.append([x,y,w,h])
            perpPoints.append([x+w/2,y+h/2])

    cv2.imshow("ROI", rescale_frame(roiperp))
    cv2.imshow("Mask", rescale_frame(maskperp))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

perpCap.release()
cv2.destroyAllWindows()

makeFrontImage()
makePerpImage()