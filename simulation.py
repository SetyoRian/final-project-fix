import cv2
import image_slicer
from cv2 import dnn_superres
import torch
import os


def deleteAll():
    firstIndex = 5
    secondIndex = 5
    for first in range(firstIndex):
        for second in range(secondIndex):
            # Read image
            image_name = 'benur_0' + str(first + 1) + '_0' + str(second + 1) + '.png'
            os.remove(image_name)


def upscale():
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read the desired model
    path = "ESPCN_x4.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("espcn", 4)

    firstIndex = 5
    secondIndex = 5
    for first in range(firstIndex):
        for second in range(secondIndex):
            # Read image
            image_name = 'benur_0' + str(first + 1) + '_0' + str(second + 1) + '.png'
            image = cv2.imread(image_name)

            # Upscale the image
            result = sr.upsample(image)
            cv2.imwrite(image_name, result)
            print(image_name)


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
            frame = frame[64:964, 507:1407, :]
            cv2.imwrite("benur.jpg", frame)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Or automatically break this whole loop if the video is over.
    else:
        break

cap.release()
cv2.destroyAllWindows()

tiles = image_slicer.slice('benur.jpg', 25)

print("Upscaling...")
upscale()
print("Upscaling Done")

# Model
model = torch.hub.load('.', 'custom',
                       path='/Users/asus/PycharmProjects/finalProject-training/yolov5-master/runs/train/YOLOv5s/weights/best.pt',
                       source='local')

batchImg = ('benur_01_01.png', 'benur_01_02.png', 'benur_01_03.png', 'benur_01_04.png', 'benur_01_05.png',
            'benur_02_01.png', 'benur_02_02.png', 'benur_02_03.png', 'benur_02_04.png', 'benur_02_05.png',
            'benur_03_01.png', 'benur_03_02.png', 'benur_03_03.png', 'benur_03_04.png', 'benur_03_05.png',
            'benur_04_01.png', 'benur_04_02.png', 'benur_04_03.png', 'benur_04_04.png', 'benur_04_05.png',
            'benur_05_01.png', 'benur_05_02.png', 'benur_05_03.png', 'benur_05_04.png', 'benur_05_05.png',
            )

path = '/Users/asus/PycharmProjects/finalProject-training/yolov5-master/'
imgs = [path + f for f in batchImg]  # batch of images
# Inference
results = model(imgs, size=720)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.
# results.save()

# Count
jum = 0
for index in range(0, 25):
    try:
        count = results.pandas().xyxy[index].value_counts('name').values[0]  # class counts (pandas)
    except:
        continue
    else:
        jum += count

print("Jumlah Benur = ", jum)

deleteAll()
