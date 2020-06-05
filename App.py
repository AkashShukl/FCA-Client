import requests
import cv2
import pickle
import os
import csv
from numpy import expand_dims, asarray
from datetime import datetime
from PIL import Image
from PIL import ImageTk
from tkinter import Tk, Button, Canvas, Label, LEFT, TRUE, PhotoImage, RIGHT, NW
from tkinter import X, BOTH, Frame
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from datetime import date


class Initialise(object):
    def __init__(self):
        self.detector = MTCNN()
        self.model = load_model('resources/facenet_keras.h5')
        self.prediction_model = pickle.load(open('resources/face_model.sav', 'rb'))
        self.names1 = pickle.load(open('resources/all_names.pkl', 'rb'))
        self.queue = list()
        print("INITIALISATION COMPLETE")

    def post_attendance(self, student_id):
        today = date.today().strftime("%m-%d-%Y")
        r = requests.post("https://fca-btech-cse.herokuapp.com/attendance",
                          json={'date': today, 'uid': student_id, 'name': student_id})

        print("request for student id : {} , status : {} , message : {}".format(student_id, r.status_code, r.reason))

    def active_learning(self, name, img2):
        if name in os.listdir('Active_learning'):
            path = 'Active_learning/' + name + '/'
            filename = name + '-' + str(datetime.now().date()) + '.jpg'
            cv2.imwrite(path + filename, img2)
        else:
            path = 'Active_learning'
            if os.path.isdir(path):
                path += '/' + name
                os.mkdir(path)
                path += '/'
                filename = name + '-' + str(datetime.now().date()) + '.jpg'
                cv2.imwrite(path + filename, img2)


class Capture:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        while not self.cap.isOpened():
            print("Camera not found! Trying Again..")
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None


class App(Initialise):
    def __init__(self, win):
        super().__init__()
        self.video = Capture()
        self.win = win
        self.win.title("FCA")
        Label(self.win, bg='white', fg='black', text="--UIT ATTENDANCE--", font=('times new roman', 25)).pack(fill=X)
        self.canvas = Canvas(self.win, bd=2, height=480, width=640)
        self.canvas.pack(fill=BOTH, expand='yes')

        self.frame3 = Frame(self.win, bg='white')
        self.frame3.pack(fill=X)

        self.marked_label = Label(self.frame3, text="", wraplength=500, font=('times new roman', 12, "bold"))
        self.marked_label.pack(fill=X, expand='yes')

        self.label1 = Label(self.frame3, text="Look in to the Camera :)", bg='white', fg="green",
                            font=('times new roman', 16))

        self.label1.pack(side=LEFT, expand=TRUE)
        photo = PhotoImage(file=r"resources/close.png")
        photoimage = photo.subsample(2, 2)
        Button(self.frame3, bg="white", image=photoimage, command=self.close).pack(side=RIGHT)

        userpic = ImageTk.PhotoImage(Image.open("resources/emptyFace.jpg"))
        self.image_label = Label(self.frame3, image=userpic, height=200, width=200)
        self.image_label.pack()

        self.updates()
        self.win.mainloop()

    def close(self):
        self.video.cap.release()
        cv2.destroyAllWindows()
        self.win.destroy()

    def recognise(self, image):
        imgcpy = image.copy()
        if image is not None:
            scale_percent = 80  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            pixels = asarray(resized_image)
            # using detect faces function to retrive box, confidence and landmarks of faces
            results = self.detector.detect_faces(pixels)
            # if face not detected just skip the image
            if len(results):
                # print("1. Face Detected form image")
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = pixels[y1:y2, x1:x2]
                # print("2. Face extracted")
                # resized for FaceNet model
                face_pixels = cv2.resize(face, (160, 160))
                face_pixels = face_pixels.astype('float32')
                mean, std = face_pixels.mean(), face_pixels.std()
                face_pixels = (face_pixels - mean) / std

                samples = expand_dims(face_pixels, axis=0)
                # Face embeddings collected
                yhat = self.model.predict(samples)

                # print("3. Face Embeddings Collected")

                # comparing the embeddings
                yhat_class = self.prediction_model.predict(yhat)

                # Retrieving the probability of the prediction
                yhat_prob = self.prediction_model.predict_proba(yhat)
                # print("4. Predicting class and probability done")

                class_index = yhat_class[0]
                class_probability = yhat_prob[0, class_index] * 100

                print('Prediction Probablity:%.3f' % class_probability)
                # setting threshold based on probability

                if class_probability > 95:
                    name = str(self.names1[class_index])
                    if name not in self.queue:
                        self.queue.append(name)
                        self.label1.configure(text=name)
                        self.write_to_csv(name)
                        self.update_image(name)
                        self.active_learning(name, imgcpy)
                        self.post_attendance(name)
                else:
                    self.label1.configure(text="Underconfident-Matching")

                temp_names = self.queue[-5:]
                names_string = ' | '.join(temp_names)
                self.marked_label.configure(text=names_string)

        return

    def updates(self):
        r, f = self.video.get_frame()
        if r:
            x = cv2.flip(f, 1)
            self.recognise(x)
            self.img = ImageTk.PhotoImage(image=Image.fromarray(x))
            self.canvas.create_image(0, 0, image=self.img, anchor=NW)
        self.win.after(3, self.updates)

    def write_to_csv(self, name):
        with open('Attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, str(datetime.now())])

    def update_image(self, name):
        t_path = "./resources/data/" + name + '/'
        if os.path.exists(t_path):
            path = t_path + os.listdir(t_path)[0]
            if os.path.exists(path):
                path = path
            else:
                path = "resources/emptyFace.jpg"
        else:
            path = "resources/emptyFace.jpg"

        t = Image.open(path)
        image = t.resize((200, 200), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(image)
        self.image_label.configure(image=img2)
        self.image_label.image = img2


App(Tk())
