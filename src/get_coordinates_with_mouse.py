import cv2
import numpy as np
 
# Create a black image and a window
windowName = 'MouseCallback'
img = cv2.imread("frame_from_video.jpg")
cv2.namedWindow(windowName)
 
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
    elif event == cv2.EVENT_MBUTTONDOWN:
        print("Middle button of the mouse is clicked - position (", x, ", ", y, ")")
 
# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)
 
def main():
    while (True):
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == 27:
            break
 
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()