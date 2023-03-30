import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
from datetime import datetime
import subprocess
import time
import RPi.GPIO as GPIO
import cv2
import image_slicer
from cv2 import dnn_superres
import torch
import os

import serial

ser = serial.Serial(port="/dev/ttyS0",
                    baudrate=38400,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=1)

GPIO.setmode(GPIO.BCM)
# The GPIO.BCM option means that we are referring to the pins by the “Broadcom SOC channel” number, these are the numbers after “GPIO”

GPIO.setwarnings(False)
# We use this line of code to avoid warning messages because we don’t end the GPIO connection properly while interrupting the program

# DEFINE GPIO Relay Pins
relay = 21
relayState = False
GPIO.setup(relay, GPIO.OUT)
GPIO.output(relay, relayState)

# make white color Backlight as default
light_default = ["@255$\n", "#255$\n", "&255$\n"]
for send in light_default:
    print(send)
    ser.write(send.encode())


contrastValue = 30  # in precentages


# For preprocessing image
def make_image_square(filename):
    img = cv2.imread(filename)
    # Size of the image
    # s = 640

    # Creating a dark square with NUMPY
    # f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    # ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    # f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img

    # resize to 640x640
    f = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

    cv2.imwrite(filename, f)


def increaseContrast(img, precentage):
    start = time.perf_counter()
    image = img
    contrast = precentage / 100

    image = cv2.addWeighted(image, 1, image, contrast, 0)

    stop = time.perf_counter()
    print("finish adding contrast in " + str(round(stop - start, 2)) + " seconds")
    return image

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


def crop_image2():
    # cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-31-Juli-2022\PL-3-80 (2).mp4")
    # # cap = cv2.VideoCapture(r"E:\PENS\PA\Dataset-2022-Alfany\PL-3-200-1.mp4")
    #
    # if not cap.isOpened():
    #     print(
    #         "Error opening the video file. Please double check your file path for typos. Or move the movie file to the "
    #         "same location as this script/notebook")
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

    tiles = image_slicer.slice('benur.jpg', 25)

    print("Upscaling...")
    upscale()
    print("Upscaling Done")


def detect_image2():
    # Model
    model = torch.hub.load('.', 'custom',
                           path='best.pt',
                           source='local')

    batchImg = ('benur_01_01.png', 'benur_01_02.png', 'benur_01_03.png', 'benur_01_04.png', 'benur_01_05.png',
                'benur_02_01.png', 'benur_02_02.png', 'benur_02_03.png', 'benur_02_04.png', 'benur_02_05.png',
                'benur_03_01.png', 'benur_03_02.png', 'benur_03_03.png', 'benur_03_04.png', 'benur_03_05.png',
                'benur_04_01.png', 'benur_04_02.png', 'benur_04_03.png', 'benur_04_04.png', 'benur_04_05.png',
                'benur_05_01.png', 'benur_05_02.png', 'benur_05_03.png', 'benur_05_04.png', 'benur_05_05.png',
                )

    path = 'C:/Users/asus/PycharmProjects/finalProject-training/yolov5-master/'
    imgs = [f for f in batchImg]  # batch of images
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
    return jum

    deleteAll()


def calculate():
    # =========== Detection Program here ========#
    crop_image2()
    # =========== Should return the result ======#
    detected_result = detect_image2()

    return detected_result


# ------------------------------------------------------------------------------------------
# additional function
def checkBacklight(frame):
    # backlight check wether the led is on or not, light intensity threshold - 50 per pixel
    check = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for row in range(118, 138):
        for column in range(118, 138):
            check += gray[row][column]
    return check


# ------------------------------------------------------------------------------------------
# --------------------------------------------- GUI Section --------------------------------

white = "#ffffff"
BlackSolid = "#000000"
font = "Constantia"
fontButtons = (font, 12)
maxWidth = 1024
maxHeight = 768
colorChoice = {'putih': '$255,255,255$\n',
               'kuning': '$255,255,0$\n',
               'hijau': '$0,255,0$\n',
               'biru': '$0,255,255$\n',
               'merah': '$255,0,0$\n'}


def _from_rgb(rgb):
    """translate an rgb tuple to hex"""
    return "#%02x%02x%02x" % rgb


class buttonL:
    def __init__(self, obj, size, position, text, font, fontSize, hoverColor, command=None):
        self.obj = obj
        self.size = size
        self.position = position
        self.font = font
        self.fontSize = fontSize
        self.hoverColor = hoverColor
        self.text = text
        self.command = command
        self.state = True
        self.Button_ = None

    def myfunc(self):
        print("Hello size :", self.size)
        print("Hello position :", self.position)
        print("Hello font :", self.font)
        print("Hello fontSize :", self.fontSize)
        print("Hello hoverState :", self.hoverColor)

    def changeOnHover(self, obj, colorOnHover, colorOnLeave):
        obj.bind("<Enter>", func=lambda e: obj.config(
            background=colorOnHover))

        obj.bind("<Leave>", func=lambda e: obj.config(
            background=colorOnLeave))

    def buttonShow(self):
        fontStyle = tkFont.Font(family=self.font, size=self.fontSize, weight="bold")
        self.Button_ = Button(self.obj, text=self.text, font=fontStyle, width=self.size[0], height=self.size[1],
                              bg=self.hoverColor[1] if isinstance(self.hoverColor, list) == True else self.hoverColor,
                              compound=TOP, command=self.command)
        self.Button_.place(x=self.position[0], y=self.position[1])

        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)

    def stateButton(self, st):
        self.st = st
        if not self.Button_ == None:
            self.Button_["state"] = self.st

    def buttonUpdate(self, textUpdate="", colorUpdate="#fff"):
        temp = [self.hoverColor[0], colorUpdate]
        self.hoverColor = temp
        self.Button_.config(text=textUpdate,
                            bg=self.hoverColor[1] if isinstance(self.hoverColor, list) == True else self.hoverColor)
        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)


class buttonImg:
    def __init__(self, obj, imgDir, size, position, hoverColor, command=None):
        self.obj = obj
        self.imgDir = imgDir
        self.size = size
        self.position = position
        self.hoverColor = hoverColor
        self.command = command
        self.state = True
        self.Button_ = None

    def changeOnHover(self, obj, colorOnHover, colorOnLeave):
        obj.bind("<Enter>", func=lambda e: obj.config(
            background=colorOnHover))

        obj.bind("<Leave>", func=lambda e: obj.config(
            background=colorOnLeave))

    def buttonShow(self):
        self.Button_ = Button(self.obj, width=self.size[0], height=self.size[1],
                              bg=self.hoverColor[1] if isinstance(self.hoverColor, list) == True else self.hoverColor,
                              bd=10, highlightthickness=4, highlightcolor="#000", highlightbackground="#000",
                              borderwidth=4, compound=TOP, command=self.command)
        self.Button_.place(x=self.position[0], y=self.position[1])
        self.imageOpen = Image.open(self.imgDir)
        self.imageOpen = self.imageOpen.resize((self.size[0], self.size[1]), Image.ANTIALIAS)
        self.imageOpen = ImageTk.PhotoImage(self.imageOpen)
        self.Button_.config(image=self.imageOpen)

        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)

    def stateButton(self, st):
        self.st = st
        if not self.Button_ == None:
            self.Button_["state"] = self.st

    def buttonUpdate(self, colorUpdate="#fff"):
        temp = [self.hoverColor[0], colorUpdate]
        self.hoverColor = temp
        self.Button_.config(bg=self.hoverColor[1] if isinstance(self.hoverColor, list) == True else self.hoverColor)
        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)


class sliderLabel:
    def __init__(self, obj, labelText, bgColor, labelPosition, labelFont, labelFontSize):
        fontStyleLabel = tkFont.Font(family=labelFont, size=labelFontSize, weight="bold")
        redLabel = Label(obj, text=labelText, bg=bgColor, fg="#fff", font=fontStyleLabel)
        redLabel.pack()
        redLabel.place(x=labelPosition[0], y=labelPosition[1])


class logo:
    def __init__(self, obj, imgDir, size, position, bg, command=None):
        self.obj = obj
        self.imgDir = imgDir
        self.size = size
        self.position = position
        self.bg = bg
        self.command = command
        self.state = True
        self.Button_ = None

    def show(self):
        self.logo = Button(self.obj, width=self.size[0], height=self.size[1], bg=self.bg, borderwidth=0)
        self.logo.place(x=self.position[0], y=self.position[1])
        self.img = Image.open(self.imgDir)
        self.img = self.img.resize((self.size[0], self.size[1]), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)
        self.logo.config(image=self.img)


class framecontroller(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Graphics window
        self.mainWindow = self
        self.mainWindow.configure(bg=BlackSolid)
        self.mainWindow.geometry('%dx%d+%d+%d' % (maxWidth, maxHeight, 0, 0))
        self.mainWindow.resizable(0, 0)
        self.mainWindow.title("SHRICO")
        self.mainWindow.attributes("-fullscreen", True)

        # # creating a container
        container = tk.Frame(self.mainWindow)
        container.configure(bg=BlackSolid)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, Page1):
            frame = F(container, self.mainWindow)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.relayState = False
        self.relay = relay

        self.ratarata = 0

        # backgroud
        self.bg = Button(self, width=1024, height=768, fg='#000', bg='#000', borderwidth=0)
        self.bg.place(x=-2, y=0)
        self.imageOpen = ImageTk.PhotoImage(Image.open('icon/tambak.jpg'))
        self.bg.config(image=self.imageOpen)

        # canvas
        self.canvas = logo(self, 'icon/back.png', [975, 520], [20, 70], bg='#fff')
        self.canvas.show()

        # contain
        fontStyleLabel = tkFont.Font(family="Arial", size=80)
        self.label1 = Label(self, text="Jumlah Benih", bg='#c9e4e9', fg='#08272b', font=fontStyleLabel)
        self.label1.pack()
        self.label1.place(x=210, y=130)

        self.shrico = logo(self, 'icon/logo.png', [220, 60], [0, 0], bg='#d4e4e8')
        self.shrico.show()

        self.pens = logo(self, 'icon/pens.png', [65, 55], [932, 0], bg='#d4e4e8')
        self.pens.show()

        self.sky = logo(self, 'icon/penssky.png', [215, 55], [714, 0], bg='#d4e4e8')
        self.sky.show()

        fontStyleLabel = tkFont.Font(family="Arial", size=180)
        self.label2 = Label(self, text="        0", bg='#c9e4e9', fg='#08272b', font=fontStyleLabel)
        self.label2.pack()
        self.label2.place(x=210, y=250)

        fontStyle = tkFont.Font(family="Arial", size=40, weight="bold")
        self.button1 = buttonL(self, [15, 2], [20, 690], "Kalibrasi", fontStyle, 15, ["yellow", '#ddd'],
                               lambda: [controller.show_frame(Page1)])
        self.button1.buttonShow()

        self.button2 = buttonL(self, [50, 2], [20, 602], "Hitung Benih", fontStyle, 20, ["#000", '#8ef695'],
                               self.Waitcalculate)
        self.button2.buttonShow()

        self.button3 = buttonImg(self, 'icon/exit.png', [60, 60], [920, 685], ["#000", "#fff"], lambda: self.close())
        self.button3.buttonShow()

    def Waitcalculate(self):
        self.relayState = not self.relayState
        print(self.relayState)
        GPIO.output(self.relay, self.relayState)
        fontStyleLabel = tkFont.Font(family="Arial", size=20)
        self.label3 = Label(self, text="Proses Sedang Berlangsung...", bg='#c9e4e9', fg='#08272b', font=fontStyleLabel)
        self.label3.pack()
        self.label3.place(x=50, y=90)

        self.label2.configure(text="        ~")

        fontStyleLabel = tkFont.Font(family="Arial", size=15)
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%B %d, %Y %H:%M:%S")
        self.label4 = Label(self, bg='#c9e4e9', fg='#08272b', font=fontStyleLabel)
        self.label4.configure(text="Waktu:\n" + self.dt_string, justify="left")
        self.label4.pack()
        self.label4.place(x=50, y=530)

        self.button1.stateButton("disabled")
        self.button2.stateButton("disabled")
        self.button3.stateButton("disabled")

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.tensorflow)
        self.thread.start()

    def tensorflow(self):
        # ================ Process ===================#
        start = time.perf_counter()
        value = calculate()
        stop = time.perf_counter()
        print("\n----Total calculation time : " + str(round(stop - start, 2)) + " seconds----\n")
        self.ratarata = value
        # ============================================#
        self.stopEvent.set()
        self.Resultcalculate(self.ratarata)

    def Resultcalculate(self, ratarata):
        self.label3.configure(text="Proses Selesai...")
        self.label2.configure(text="    " + str(ratarata))

        self.button1.stateButton("active")
        self.button1.buttonShow()
        self.button2.stateButton("active")
        self.button2.buttonShow()
        self.button3.stateButton("active")
        time.sleep(3)
        self.label3.configure(text="")
        self.relayState = not self.relayState
        print(self.relayState)
        GPIO.output(self.relay, self.relayState)

    def close(self):
        subprocess.run('sudo shutdown -h now', shell=True)


class Page1(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.videoObj = None
        self.colorSelected = ''
        self.backColor = "0,0,0"
        self.cameraFlag = False
        self.ledFlag = False
        self.count = 0
        self.relayState = False
        self.relay = relay

        self.configure(bg="#444")
        fontStyle = tkFont.Font(family="Arial", size=28, weight="bold")
        fontStyleLabel = tkFont.Font(family="Arial", size=19)
        # ---------------------- comboBox -------------------------
        # self.selectColor = tk.StringVar()
        # self.selectColor.set('putih')

        # colorBox = ttk.Combobox(self, textvariable=self.selectColor, state = 'readonly', width = 27, font = 'Arial 25')

        # colorBox['values'] = ('putih', 'kuning', 'hijau', 'biru', 'merah')

        # colorBox.pack(ipadx = 15, pady = 25)
        # colorBox.place(x= 10, y = 90)

        # colorBox.bind('<<ComboboxSelected>>', self.colorSelect)

        # ----------------------- slider ---------------------------
        redLabel = sliderLabel(self, labelText="Merah", bgColor='#444', labelPosition=[255, 50], labelFont="Arial",
                               labelFontSize=21)
        greenLabel = sliderLabel(self, labelText="Hijau", bgColor='#444', labelPosition=[550, 50], labelFont="Arial",
                                 labelFontSize=21)
        blueLabel = sliderLabel(self, labelText="Biru", bgColor='#444', labelPosition=[845, 50], labelFont="Arial",
                                labelFontSize=21)

        self.redNow = tk.DoubleVar()
        self.greenNow = tk.DoubleVar()
        self.blueNow = tk.DoubleVar()

        self.red = ttk.Scale(self, from_=0, length=250, to=255, orient='horizontal', command=self.getRedColor,
                             variable=self.redNow)
        self.red.place(x=170, y=90)
        self.green = ttk.Scale(self, from_=0, length=250, to=255, orient='horizontal', command=self.getGreenColor,
                               variable=self.greenNow)
        self.green.place(x=460, y=90)
        self.blue = ttk.Scale(self, from_=0, length=250, to=255, orient='horizontal', command=self.getBlueColor,
                              variable=self.blueNow)
        self.blue.place(x=750, y=90)

        self.redValue = Label(self, text="0", bg='#444', fg='#fff', font=fontStyleLabel)
        self.redValue.place(x=270, y=120)

        self.greenValue = Label(self, text="0", bg='#444', fg='#fff', font=fontStyleLabel)
        self.greenValue.place(x=560, y=120)

        self.blueValue = Label(self, text="0", bg='#444', fg='#fff', font=fontStyleLabel)
        self.blueValue.place(x=850, y=120)

        label1 = Label(self, text="Pastikan Wadah Benih\nUdang Terlihat Jelas\nMelalui Kamera", bg='#444', fg='#fff',
                       font=fontStyleLabel)
        label1.pack()
        label1.place(x=730, y=180)

        self.back = buttonImg(self, 'icon/home.png', [130, 130], [780, 570], ["#000", "#fff"],
                              lambda: [controller.show_frame(StartPage), videoStream.onClose(self.videoObj)])
        self.back.buttonShow()

        self.ledButton = buttonImg(self, 'icon/sun.png', [80, 80], [35, 50], [BlackSolid, "#fff"],
                                   lambda: [self.ledState()])
        self.ledButton.buttonShow()

        self.button2 = buttonL(self, [16, 2], [710, 310], "Camera On", fontStyle, 18,
                               [BlackSolid, _from_rgb((244, 239, 140))], lambda: [self.cameraState()])
        self.button2.buttonShow()

        button3 = buttonL(self, [16, 2], [710, 450], "Record", fontStyle, 18, [BlackSolid, _from_rgb((255, 190, 100))],
                          lambda: [self.startRecord()])
        button3.buttonShow()

        self.videoObj = videoStream()

    # If you use combobox
    def colorSelect(self, event=None):
        self.colorSelected = self.selectColor.get()
        # print(self.colorSelected)
        for color in colorChoice.keys():
            if (self.colorSelected == color):
                self.backColor = colorChoice.get(self.colorSelected)
                print(str(colorChoice.get(self.colorSelected)))
        ser.write(self.backColor.encode())

    # If you use trackbar
    def getRedColor(self, event=None):
        self.redValue.configure(text='{:d}'.format(round(self.redNow.get())))
        self.backColor = str(round(self.redNow.get())) + "," + str(round(self.greenNow.get())) + "," + str(
            round(self.blueNow.get()))
        sendRed = "@" + str(round(self.redNow.get())) + "$\n"
        print(sendRed)
        ser.write(sendRed.encode())

    def getGreenColor(self, event=None):
        self.greenValue.configure(text='{:d}'.format(round(self.greenNow.get())))
        self.backColor = str(round(self.redNow.get())) + "," + str(round(self.greenNow.get())) + "," + str(
            round(self.blueNow.get()))
        sendGreen = "#" + str(round(self.greenNow.get())) + "$\n"
        print(sendGreen)
        ser.write(sendGreen.encode())

    def getBlueColor(self, event=None):
        self.blueValue.configure(text='{:d}'.format(round(self.blueNow.get())))
        self.backColor = str(round(self.redNow.get())) + "," + str(round(self.greenNow.get())) + "," + str(
            round(self.blueNow.get()))
        sendBlue = "&" + str(round(self.blueNow.get())) + "$\n"
        print(sendBlue)
        ser.write(sendBlue.encode())

    def cameraState(self, event=None):
        self.cameraFlag = not (self.cameraFlag)
        print(self.cameraFlag)
        if (self.cameraFlag):
            self.button2.buttonUpdate("Camera Off", _from_rgb((255, 150, 150)))
            videoStream.onStart(self.videoObj, cameraFlag=self.cameraFlag)
        else:
            self.button2.buttonUpdate("Camera On", _from_rgb((244, 239, 140)))
            videoStream.onStart(self.videoObj, cameraFlag=self.cameraFlag)

    def ledState(self, event=None):
        self.ledFlag = not (self.ledFlag)
        print(self.ledFlag)
        if (self.ledFlag):
            self.relayState = not self.relayState
            print(self.relayState)
            GPIO.output(self.relay, self.relayState)
            self.ledButton.buttonUpdate(_from_rgb((244, 239, 140)))
        else:
            self.relayState = not self.relayState
            print(self.relayState)
            GPIO.output(self.relay, self.relayState)
            self.ledButton.buttonUpdate("#fff")

    def startRecord(self, event=None):
        self.count += 1
        videoStream.onStart(self.videoObj, bgColor=self.backColor, recordCount=self.count, recordTime=15, record="yes")


class videoStream(tk.Frame):
    def __init__(self):

        self.ret = None
        self.frame = None

        self.thread = None
        self.stopEvent = None
        self.capWebcam = None
        self.fourcc = None
        self.out = None

        self.time = datetime.now()
        self.timeString = self.time.strftime("%d-%B-%Y %H:%M:%S")
        self.now = None
        self.check = 0
        self.count = 0

        self.panel = None

    def onStart(self, bgColor='255,255,255', recordCount=0, recordTime=1, record="no", cameraFlag=False):
        self.record = record
        self.recordCount = recordCount
        self.recordTime = recordTime * 60 * 1000  # minutes from ms
        self.outVideo = "record/" + bgColor + " - " + self.timeString + "(" + str(self.recordCount) + ").mp4"
        self.message = "perekaman video berdurasi " + str(recordTime) + " menit dimulai ..."

        if ((self.record == "no") and (cameraFlag == True)):
            self.capWebcam = cv2.VideoCapture(0)
            # if not self.capWebcam.isOpened():
            #    messagebox.showerror("Error !", "Kamera tidak terhubung ! Harap memeriksa koneksi kamera ...")
            #    raise Exception("Could not open video device")
            self.capWebcam.set(3, 656)
            self.capWebcam.set(4, 600)
            # self.capWebcam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            self.stopEvent = threading.Event()
            self.thread = threading.Thread(target=self.videoLoop)
            self.thread.start()
        elif (self.record == "yes"):
            self.capWebcam = cv2.VideoCapture(0)
            if not self.capWebcam.isOpened():
                messagebox.showerror("Error !", "Kamera tidak terhubung ! Harap memeriksa koneksi kamera ...")
                raise Exception("Could not open video device")
            messagebox.showinfo("notification", self.message)
            self.capWebcam.set(3, 1920)
            self.capWebcam.set(4, 1080)
            # self.capWebcam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.outVideo, self.fourcc, 5.0, (1920, 1080))
            self.prev = int(round(time.time() * 1000))

            self.stopEvent = threading.Event()
            self.thread = threading.Thread(target=self.recordVideo)
            self.thread.start()
        else:
            self.capWebcam.release()

    def onClose(self):
        print("[INFO] closing...")
        if not self.panel == None:
            self.panel.destroy()
            self.stopEvent.set()
            self.capWebcam.release()

    def videoLoop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():

                self.ret, self.frame = self.capWebcam.read()
                print(self.ret)

                if (self.ret == True):
                    image = cv2.flip(self.frame, 1)

                    # backlight check wether the led is on or not, light intensity threshold - 30 per pixel
                    # self.check = checkBacklight(self.frame)
                    # if(self.check < 16000):
                    #    messagebox.showerror("Error !", "Backlight tidak menyala ! Harap memeriksa sambungan backlight ...")
                    #    self.capWebcam.release()
                    #    break

                    # self.check = 0
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                        self.panel = Label(image=image, width=660, height=550)
                        self.panel.image = image
                        self.panel.place(x=35, y=160)

                    # otherwise, simply update the panel
                    else:
                        if (not self.panel == None):
                            self.panel.configure(image=image)
                            self.panel.image = image
                else:
                    self.panel.destroy()
                    self.capWebcam.release()
                    self.panel = None

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def recordVideo(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():

                self.ret, self.frame = self.capWebcam.read()

                if (self.ret == True):
                    image = cv2.flip(self.frame, 1)

                    # backlight check wether the led is on or not, light intensity threshold - 30 per pixel
                    # self.check = checkBacklight(self.frame)
                    # if(self.check < 16000):
                    #    messagebox.showerror("Error !", "Backlight tidak menyala ! Harap memeriksa sambungan backlight ...")
                    #    self.capWebcam.release()
                    #    break

                    # self.check = 0
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    self.now = int(round(time.time() * 1000))
                    if (self.now - self.prev <= self.recordTime):
                        self.out.write(self.frame)

                        if self.panel is None:
                            self.panel = Label(image=image, width=660, height=550)
                            self.panel.image = image
                            self.panel.place(x=35, y=160)

                        # otherwise, simply update the panel
                        else:
                            if (not self.panel == None):
                                self.panel.configure(image=image)
                                self.panel.image = image
                    else:
                        self.now = 0
                        self.panel.destroy()
                        self.capWebcam.release()
                        self.panel = None
                        self.message = "Perekaman Selesai ..."
                        messagebox.showinfo("notification", self.message)
                        break

        except RuntimeError:
            print("[INFO] caught a RuntimeError")


app = framecontroller()
app.mainloop()
