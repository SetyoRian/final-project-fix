import cv2
import image_slicer
from cv2 import dnn_superres
import torch
import os

indexing = '5'


def deleteAll():
    firstIndex = 5
    secondIndex = 5
    for first in range(firstIndex):
        for second in range(secondIndex):
            # Read image
            image_name = 'DataSelect/PL-3-260/benur' + indexing + '_0' + str(first + 1) + '_0' + str(
                second + 1) + '.png'
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
            image_name = 'DataSelect/PL-3-260/benur' + indexing + '_0' + str(first + 1) + '_0' + str(
                second + 1) + '.png'
            image = cv2.imread(image_name)

            scale_percent = 400  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Upscale the image
            result = sr.upsample(image)
            # result = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(image_name, result)
            print(image_name)


cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-31-Juli-2022\PL-3-80 (2).mp4")
# cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-2022-Alfany\PL-3-200-1.mp4")

# if not cap.isOpened():
#     print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the "
#           "same location as this script/notebook")
#
# # While the video is opened
# while cap.isOpened():
#     # Read the video file.
#     ret, frame = cap.read()
#     # If we got frames show them.
#     if ret:
#         # time.sleep(1/fps)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('c'):
#             frame = frame[64:964, 507:1407, :]
#             cv2.imwrite("benur.jpg", frame)
#             break
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     # Or automatically break this whole loop if the video is over.
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

tiles = image_slicer.slice('DataSelect/PL-3-260/benur' + indexing + '.jpg', 25)

print("Upscaling...")
upscale()
print("Upscaling Done")

# Model
model = torch.hub.load('.', 'custom',
                       path='/Users/asus/PycharmProjects/finalProject-training/yolov5-master/runs/train/YOLOv5s/weights/best.pt',
                       source='local')

batchImg = (
    'benur' + indexing + '_01_01.png', 'benur' + indexing + '_01_02.png', 'benur' + indexing + '_01_03.png', 'benur' + indexing + '_01_04.png', 'benur' + indexing + '_01_05.png',
    'benur' + indexing + '_02_01.png', 'benur' + indexing + '_02_02.png', 'benur' + indexing + '_02_03.png', 'benur' + indexing + '_02_04.png', 'benur' + indexing + '_02_05.png',
    'benur' + indexing + '_03_01.png', 'benur' + indexing + '_03_02.png', 'benur' + indexing + '_03_03.png', 'benur' + indexing + '_03_04.png', 'benur' + indexing + '_03_05.png',
    'benur' + indexing + '_04_01.png', 'benur' + indexing + '_04_02.png', 'benur' + indexing + '_04_03.png', 'benur' + indexing + '_04_04.png', 'benur' + indexing + '_04_05.png',
    'benur' + indexing + '_05_01.png', 'benur' + indexing + '_05_02.png', 'benur' + indexing + '_05_03.png', 'benur' + indexing + '_05_04.png', 'benur' + indexing + '_05_05.png',
)

path = '/Users/asus/PycharmProjects/finalProject-training/yolov5-master/DataSelect/PL-3-260/'
path2 = 'E:\\PENS\\PA\\Dataset-31-Juli-2022\\DataSelect\\PL-3-80\\benur'
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
