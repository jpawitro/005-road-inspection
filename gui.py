import os
import pickle
from tqdm import tqdm

import tkinter as tk
from tkinter.constants import ANCHOR, BOTTOM, CENTER, INSERT, LEFT, RIGHT, S, TOP, X
from tkinter import ttk, filedialog

import numpy as np
import pandas as pd
from pandastable import Table

import cv2
import pywt
from PIL import Image, ImageTk

from tensorflow import keras

# ==== Constants ==== #
BACKGROUND = "#4bacc6"
ALTBACKGROUND = "#4bacb0"
shape = (512,512)
le = pickle.load(open(os.path.join("model","le.sav"),"rb"))
model = keras.models.load_model('model')
transdict = {
    "retak buaya": "Area Crack",
    "retak garis": "Line Crack",
    "tidak retak": "Good Condition"
}
# ==== Constants ==== #

# ==== GUI ==== #
class root(tk.Tk):
    def __init__(self):
        # head
        super(root, self).__init__()
        self.title("Road Inspection")
        self.geometry("1024x768")
        self['background'] = BACKGROUND
        # top frame
        self.header = tk.Frame(self, background=BACKGROUND)
        self.header.pack(side=TOP)
        self.headerL = tk.Frame(self.header, background=BACKGROUND)
        self.headerL.pack(side=LEFT)
        self.headerR = tk.Frame(self.header, background=BACKGROUND)
        self.headerR.pack(side=LEFT)
        self.top = tk.Frame(self.headerR, background=BACKGROUND)
        self.top.pack(side=TOP)

        self.label = tk.Label(self.top, text='Road Inspection', fg='black', bg='#4bacc6', padx=5, pady=5)
        self.label.pack(anchor='n', side='top', padx=20, pady=3)
        self.label.config(font=("Comic Sans", 32, 'bold'))

        # define frame
        self.sub = tk.Frame(self.headerR, background=BACKGROUND, padx=10, pady=3)
        self.sub.pack(side=TOP, pady=3)
        self.subL = tk.Frame(self.sub, background=BACKGROUND, padx=10, pady=3)
        self.subL.pack(side=LEFT, pady=3)
        self.subC = tk.Frame(self.sub, background=BACKGROUND, padx=10, pady=3)
        self.subC.pack(side=LEFT, pady=3)
        self.subR = tk.Frame(self.sub, background=BACKGROUND, padx=10, pady=3)
        self.subR.pack(side=LEFT, pady=3)
        self.bottom = tk.Frame(self, background=BACKGROUND, padx=10, pady=3)
        self.bottom.pack(side=TOP)
        self.frame = tk.Frame(self, background=BACKGROUND, padx=10, pady=3)
        self.frame.pack(side=TOP, pady=3)
        self.frameL = tk.Frame(self.frame, background=BACKGROUND, padx=10, pady=3)
        self.frameL.pack(side=LEFT, pady=3)
        self.frameR = tk.Frame(self.frame, background=BACKGROUND, padx=10, pady=3)
        self.frameR.pack(side=LEFT, pady=3)
        self.frTop = tk.Frame(self.frameR, background=BACKGROUND, padx=10, pady=3)
        self.frTop.pack(side=TOP, pady=3)
        self.frMid = tk.Frame(self.frameR, background=BACKGROUND, padx=10, pady=3)
        self.frMid.pack(side=TOP, pady=3)
        self.frBtm = tk.Frame(self.frameR, background=BACKGROUND, padx=10, pady=3)
        self.frBtm.pack(side=BOTTOM, pady=3)

        self.folderPath = tk.StringVar()
        self.inputlabel = tk.Label(self.subL ,text="Input Folder")
        self.inputlabel.pack(side=LEFT, pady=10)
        self.inputtext = tk.Entry(self.subC,textvariable=self.folderPath, width=50)
        self.inputtext.pack(side=LEFT, pady=10)
        self.btnFind = ttk.Button(self.subR, text="Browse Folder",command=self.folderpath)
        self.btnFind.pack(side=LEFT, pady=10)

        self.options_list = ["Approximation", "Horizontal Detail", "Vertical Detail", "Diagonal Detail"]
        self.value_inside = tk.StringVar()
        self.value_inside.set("Choose Transformation")

        self.button1 = tk.Button(self.bottom, text="Quit", command=self._quit)
        self.button1.pack(side=RIGHT, pady=2, padx=10)
        self.button2 = tk.Button(self.bottom, text="Proceed", command=self.doStuff)
        self.button2.pack(side=LEFT, pady=2, padx=10)
        self.question_menu = tk.OptionMenu(self.bottom, self.value_inside, *self.options_list)
        self.question_menu.pack(side=TOP, pady=2, padx=10)

        self.imgfilename = tk.StringVar()

        self.num = tk.IntVar()
    def _quit(self):
        self.quit()
        self.destroy()

    def folderpath(self):
        self.folder_selected = filedialog.askdirectory()
        self.folderPath.set(self.folder_selected)

    def doStuff(self):
        if isinstance(self.num,int):
            self.filename.destroy()
            self.resultlabel.destroy()
            self.image_frame.destroy()
            self.prev.destroy()
            self.next.destroy()
        root_folder = self.folderPath.get()
        transform = self.value_inside.get()
        print("Doing stuff with folder", root_folder, transform)
        self.image_path = []
        for img in os.listdir(root_folder):
            self.image_path.append(os.path.join(root_folder,img))
        self.image = []
        self.imageH = []
        self.imageV = []
        self.imageD = []
        for img in tqdm(self.image_path):
            arr = cv2.imread(img)
            arr = cv2.resize(arr,shape)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(arr)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            arr = cv2.merge((cl,a,b))
            arr = cv2.cvtColor(arr, cv2.COLOR_LAB2BGR)
            arr = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
            coeffs2 = pywt.dwt2(arr, 'bior1.3')
            LL, (LH, HL, HH) = coeffs2
            self.image.append(LL.tolist())
            self.imageH.append(LH.tolist())
            self.imageV.append(HL.tolist())
            self.imageD.append(HH.tolist())
        if transform == "Horizontal Detail":
            X = np.array(self.imageH)/255
        elif transform == "Vertical Detail":
            X = np.array(self.imageV)/255
        elif transform == "Diagonal Detail":
            X = np.array(self.imageD)/255
        else:
            X = np.array(self.image)/255
        input_shape = (X.shape[1],X.shape[2])
        X = X.reshape(-1, input_shape[0], input_shape[1], 1)
        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=-1)
        self.results = [transdict[p] for p in le.inverse_transform(predictions)]
        df = pd.DataFrame({
            "image":self.image_path,
            "prediction":self.results})
        df['image'] = df['image'].str.split("/",expand=True).iloc[:,-1]
        self.table = pt = Table(self.frameL, dataframe=df,
                                    showtoolbar=True, showstatusbar=True, width=500)
        pt.show()
        self.num = 0
        self.imgfilename = self.image_path[self.num].split("/")[-1]
        self.filename = tk.Label(self.frTop, text=self.imgfilename, fg='black', bg='white', padx=5, pady=5)
        self.filename.pack(side=TOP, pady=2, padx=10)
        self.resultlabel = tk.Label(self.frTop, text=self.results[self.num], fg='black', bg='white', padx=5, pady=5)
        self.resultlabel.pack(side=TOP, pady=2, padx=10)
        self.img = Image.open(self.image_path[self.num]).resize((300,300))
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.image_frame = tk.Label(self.frMid, image= self.imgtk)
        self.image_frame.pack(side=LEFT)
        self.prev = tk.Button(self.frBtm, text="Previous", command=self._previous)
        self.prev.pack(side=LEFT, pady=2, padx=10)
        self.next = tk.Button(self.frBtm, text="Next", command=self._next)
        self.next.pack(side=LEFT, pady=2, padx=10)
    def _next(self):
        if self.num == len(self.image_path):
            self.num = self.num
        else:
            self.num+=1
        self.filename.destroy()
        self.resultlabel.destroy()
        self.image_frame.destroy()
        self.imgfilename = self.image_path[self.num].split("/")[-1]
        self.filename = tk.Label(self.frTop, text=self.imgfilename, fg='black', bg='white', padx=5, pady=5)
        self.filename.pack(side=TOP, pady=2, padx=10)
        self.resultlabel = tk.Label(self.frTop, text=self.results[self.num], fg='black', bg='white', padx=5, pady=5)
        self.resultlabel.pack(side=TOP, pady=2, padx=10)
        self.img = Image.open(self.image_path[self.num]).resize((300,300))
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.image_frame = tk.Label(self.frMid, image= self.imgtk)
        self.image_frame.pack(side=LEFT)
    def _previous(self):
        if self.num == 0:
            self.num = len(self.image_path)-1
        else:
            self.num-=1
        self.filename.destroy()
        self.resultlabel.destroy()
        self.image_frame.destroy()
        self.imgfilename = self.image_path[self.num].split("/")[-1]
        self.filename = tk.Label(self.frTop, text=self.imgfilename, fg='black', bg='white', padx=5, pady=5)
        self.filename.pack(side=TOP, pady=2, padx=10)
        self.resultlabel = tk.Label(self.frTop, text=self.results[self.num], fg='black', bg='white', padx=5, pady=5)
        self.resultlabel.pack(side=TOP, pady=2, padx=10)
        self.img = Image.open(self.image_path[self.num]).resize((300,300))
        self.imgtk = ImageTk.PhotoImage(image=self.img)
        self.image_frame1 = tk.Label(self.frMid, image= self.imgtk)
        self.image_frame1.pack(side=LEFT)


road = root()
road.mainloop()
# ==== GUI ==== #