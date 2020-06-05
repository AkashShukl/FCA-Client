from numpy import load
from numpy import expand_dims
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import time

start_time = datetime.now()

detector = MTCNN()


def extract_face(filename, required_size=(160, 160)):
    print(filename)
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d pictures for USER: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)


# load train dataset
trainX, trainy = load_dataset('dataset/Train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('dataset/Test/')
# save arrays to one file in compressed format
savez_compressed('faces.npz', trainX, trainy, testX, testy)
print("/n/n ---->FEATURE EXTRACTION COMPLETED<----")

#######################################################

print("hold")
time.sleep(2)
print("released")

# calculate a face embedding for each face in the dataset using facenet



# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# load the face dataset
data = load('faces.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
model = load_model('resources/facenet_keras.h5')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)

# save arrays to one file in compressed format
savez_compressed('embeddings.npz', newTrainX, trainy, newTestX, testy)

print("\n\n--->face embeddings Collected<---")

############################################

print("hold")
time.sleep(2)
print("released")

############################################


# load dataset
data = load('embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
names_label = list(out_encoder.classes_)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# Saving Model
filename = 'resources/face_model.sav'
pickle.dump(model, open(filename, 'wb'))
# saving names in a pickle file for future use
tempname = 'resources/all_names.pkl'
pickle.dump(names_label, open(tempname, 'wb'))
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
end_time = datetime.now()

print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
print("Training Completed! Model File Name: face_model")
print("Duration : ", end_time - start_time)
