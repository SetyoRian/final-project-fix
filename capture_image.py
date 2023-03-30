import cv2

cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-31-Juli-2022\PL-3-80 (2).mp4")
# cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-2022-Alfany\PL-3-200-1.mp4")

if not cap.isOpened():
    print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the "
          "same location as this script/notebook")

# While the video is opened
while cap.isOpened():
    # Read the video file.
    ret, frame = cap.read()
    # If we got frames show them.
    if ret:
        # time.sleep(1/fps)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite("benur.jpg", frame)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Or automatically break this whole loop if the video is over.
    else:
        break

cap.release()
cv2.destroyAllWindows()