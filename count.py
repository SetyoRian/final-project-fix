import torch

# Model
model = torch.hub.load('.', 'custom',
                       path='/Users/asus/PycharmProjects/finalProject-training/yolov5-master/runs/train/exp/weights/best.pt',
                       source='local')  # yolov5n - yolov5x6 official model
#                                            'custom', 'path/to/best.pt')  # custom model

# Images
im = '/Users/asus/PycharmProjects/finalProject-training/yolov5-master/data/testing/PL3_1_03_02_12.jpg'
# or file, Path, URL, PIL, OpenCV, numpy, list
dir = '/Users/asus/PycharmProjects/finalProject-training/yolov5-master/data/testing/'
imgs = [dir + f for f in ('PL3_1_01_01_1.jpg', 'PL3_1_01_02_2.jpg', 'PL3_1_01_03_3.jpg',
                          'PL3_1_01_04_4.jpg', 'PL3_1_01_05_5.jpg', 'PL3_1_02_01_6.jpg',
                          'PL3_1_02_02_7.jpg', 'PL3_1_02_03_8.jpg', 'PL3_1_02_04_9.jpg',
                          'PL3_1_02_05_10.jpg', 'PL3_1_03_01_11.jpg', 'PL3_1_03_02_12.jpg',
                          'PL3_1_03_03_13.jpg', 'PL3_1_03_04_14.jpg', 'PL3_1_03_05_15.jpg',
                          'PL3_1_04_01_16.jpg', 'PL3_1_04_02_17.jpg', 'PL3_1_04_03_18.jpg',
                          'PL3_1_04_04_19.jpg', 'PL3_1_04_05_20.jpg', 'PL3_1_05_01_21.jpg',
                          'PL3_1_05_02_22.jpg', 'PL3_1_05_03_23.jpg', 'PL3_1_05_04_24.jpg',
                          'PL3_1_05_05_25.jpg'
                          )]  # batch of images
# Inference
results = model(imgs)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# print(results.xyxy[1])  # im predictions (tensor)

# results.pandas().xyxy[0]  # im predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
#
jum = 0
for index in range(0, 25):
    try:
        count = results.pandas().xyxy[index].value_counts('name').values[0]  # class counts (pandas)
    except:
        continue
    else:
        jum += count

# jum = results.pandas().xyxy[1].value_counts('name').values[0]  # class counts (pandas)
# print(type(jum))
print("Jumlah Benur = ", jum)
# person    2
# tie       1

# hasil = results.print()
# print(hasil)
