import cv2
import os
import threading
from PIL import Image
from PIL import ImageTk
from tkinter import Label, Canvas, Frame, Button, Entry, X, NW
from tkinter import BOTH, PanedWindow, VERTICAL, TRUE, StringVar, Tk


class Capture:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)

    def create_dataset(self, name, count):

        detector = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')
        user_dir_path = 'dataset/' + name + '/'

        if not os.path.exists(user_dir_path):
            os.mkdir(user_dir_path)

        ret, img = self.cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            img = gray[y:y + h, x:x + w]
            img = cv2.resize(img, (112, 92))
            cv2.imwrite(user_dir_path + str(count) + '.jpg', img)

        return


class App:
    def __init__(self, win):
        self.video = Capture()
        self.win = win
        self.imgcount = 1
        self.win['bg'] = 'white'
        Label(self.win, bg='white', fg='black', text="Attendance Registration",
              font=('times new roman', 25)).pack()
        # master pane contains the seperation
        self.m1 = PanedWindow(self.win, bd=5, background="black")
        self.m1.pack(fill=BOTH, expand=1)

        # leftpane
        self.leftpane = PanedWindow(self.m1, orient=VERTICAL, bd=1, background="white")
        self.m1.add(self.leftpane)

        pic = Image.open('resources/emptyFace.jpg')
        self.ima = ImageTk.PhotoImage(pic)

        self.P1 = Label(self.leftpane, image=self.ima, borderwidth=2, relief="solid", height=150, width=180)
        self.P1.pack(padx=(2, 2), pady=(2, 2))
        self.P1.image = self.ima

        self.P2 = Label(self.leftpane, image=self.ima, borderwidth=2, relief="solid", height=150, width=180)
        self.P2.pack(padx=(2, 2), pady=(2, 2))
        self.P2.image = self.ima

        self.P3 = Label(self.leftpane, image=self.ima,borderwidth=2, relief="solid", height=150, width=180)
        self.P3.pack(padx=(2, 2), pady=(2, 2))
        self.P3.image = self.ima

        # rightpane
        self.rightpane = PanedWindow(self.m1, orient=VERTICAL, bd=2, background="white")
        self.m1.add(self.rightpane)
        h, w = 480, 640
        self.frame2 = Frame(self.rightpane, bg='cyan')
        self.frame2.pack(expand=TRUE)
        self.canvas = Canvas(self.frame2, width=w, height=h, bd=2)
        self.canvas.grid(row=0, column=0)

        self.frame3 = Frame(self.rightpane, bg='white')
        self.frame3.pack(fill=X)

        # inside frame 3
        self.suid = StringVar(value="Enter UID")

        self.Uid = Entry(self.frame3, textvariable=self.suid, font=('times new roman', 13), borderwidth=2, relief="solid", fg='black').grid(
            row=0, column=1, padx=80, pady=5)

        Button(self.frame3, text="CAPTURE", font=('times new roman', 12), fg='black', bg='green', height=1, width=10,
               command=lambda: [self.activate_threads()]).grid(row=0, column=2, padx=10, pady=5)

        Button(self.frame3, text="SUBMIT", font=('times new roaman', 12), fg='black', bg='red', height=1, width=10,
               command=self.details).grid(row=0, column=3,padx=10, pady=5)

        self.updates()

        self.win.mainloop()

    def preview(self):
        s = self.suid.get()

        list_of_faces = os.listdir("dataset/" + s + "/")
        size_of_faces = len(list_of_faces)

        if size_of_faces >= 3:
            temp_list = list_of_faces[size_of_faces - 3:]
        else:
            temp_list = list_of_faces

        path = 'dataset/' + s + '/' + temp_list[-1]

        if os.path.exists(path):
            image1 = ImageTk.PhotoImage(Image.open(path))
            self.P1.config(image=image1)
            self.P1.image = image1

        if len(temp_list) > 1:
            path = 'dataset/' + s + '/' + temp_list[-2]
            if os.path.exists(path):
                image2 = ImageTk.PhotoImage(Image.open(path))
                self.P2.config(image=image2)
                self.P2.image = image2

        if len(temp_list) > 2:
            path = 'dataset/' + s + '/' + temp_list[-3]
            if os.path.exists(path):
                image3 = ImageTk.PhotoImage(Image.open(path))
                self.P3.config(image=image3)
                self.P3.image = image3

    def activate_threads(self):
        self.thread_var = True
        self.threads()

    def updates(self):

        r, f = self.video.get_frame()
        if r:
            self.img = ImageTk.PhotoImage(image=Image.fromarray(f))
            self.canvas.create_image(0, 0, image=self.img, anchor=NW)
        self.win.after(6, self.updates)

    def threads(self):
        if self.thread_var:
            if self.check_fields():
                print("threads started..")
                self.video.create_dataset(self.suid.get(), self.imgcount)
                self.imgcount += 1

            else:
                from tkinter import messagebox
                messagebox.showerror("Error", "Enter valid details! and try again.")

            t3 = threading.Thread(target=self.preview)
            t3.start()

    def check_fields(self):
        uid = self.suid.get()
        if uid == "Enter UID":
            return False
        else:
            return True

    def details(self):
        if not self.check_fields:
            from tkinter import messagebox
            messagebox.showerror("Error", "Enter valid UID!")
        else:
            # call for post
            print(self.suid.get())

    def close_btn(self):
        try:
            self.win.destroy()
        except:
            pass


App(Tk())
