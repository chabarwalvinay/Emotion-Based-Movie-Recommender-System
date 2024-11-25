import cv2
import keras
from deepface.detectors import FaceDetector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from keras.models import load_model
import pickle
import pandas as pd


with open('file/sim1.pkl', 'rb') as file:
    sim1 = pickle.load(file)
with open('file/sim2.pkl', 'rb') as file:
    sim2 = pickle.load(file)
with open('file/sim3.pkl', 'rb') as file:
    sim3 = pickle.load(file)
with open('file/sim4.pkl', 'rb') as file:
    sim4 = pickle.load(file)
with open('file/sim5.pkl', 'rb') as file:
    sim5 = pickle.load(file)
with open('file/sim5.pkl', 'rb') as file:
    sim6 = pickle.load(file)


rd1 = pd.read_csv('file/rd1.csv')
rd2 = pd.read_csv('file/rd2.csv')
rd3 = pd.read_csv('file/rd3.csv')
rd4 = pd.read_csv('file/rd4.csv')
rd5 = pd.read_csv('file/rd5.csv')
rd6 = pd.read_csv('file/rd6.csv')

action = pd.read_csv('file/action.csv')
horror = pd.read_csv('file/horror.csv')
romance = pd.read_csv('file/romance.csv')
crime = pd.read_csv('file/crime.csv')
family = pd.read_csv('file/family.csv')
scifi = pd.read_csv('file/scifi.csv')

model = keras.models.load_model('saved_model/model_t2')
# model = keras.models.load_model(r"D:\emotion_recognition-main\emotion_recognition-main\model.h5")
def get_class(argument):
    return ["Fear", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"][argument]

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
prev_mood = []
mode_mood = "Happy"

emotions = []
l1 = []

def rd_movies(n):
    if(n == 4):
        sampled_titles = rd4['title'].sample(n=5)
    if(n == 3):
       sampled_titles = rd3['title'].sample(n=5)
    if(n == 2):
        sampled_titles = rd2['title'].sample(n=5)
    if(n == 1):
       sampled_titles = rd1['title'].sample(n=5)
    if(n == 5):
       sampled_titles = rd5['title'].sample(n=5)
    if(n == 6):
       sampled_titles = rd6['title'].sample(n=5)
    for title in sampled_titles:
        print(title)


def recommend(movie, n):
    if(n == 6):
        movie_index=scifi[scifi['title']==movie].index[0]
        distances=sim4[movie_index]
    if(n == 5):
        movie_index=family[family['title']==movie].index[0]
        distances=sim4[movie_index]
    if(n == 4):
        movie_index=crime[crime['title']==movie].index[0]
        distances=sim4[movie_index]
    if(n == 3):
        movie_index=romance[romance['title']==movie].index[0]
        distances=sim3[movie_index]
    if(n == 2):
        movie_index=horror[horror['title']==movie].index[0]
        distances=sim2[movie_index]
    if(n == 1):
        movie_index=action[action['title']==movie].index[0]
        distances=sim1[movie_index]

    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        if(n == 6): print(crime.iloc[i[0]].title)
        if(n == 5): print(crime.iloc[i[0]].title)
        if(n == 4): print(crime.iloc[i[0]].title)
        if(n == 3): print(romance.iloc[i[0]].title)
        if(n == 2): print(horror.iloc[i[0]].title)
        if(n == 1): print(action.iloc[i[0]].title)

def emotion(x):
    if x=='Angry':
        rd_movies(1)
        y=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
        recommend(y, 1)
    elif x=='Fear':
        h2=input('Do you want to watch a movie that resonates with your current emotion(1) or do you want to improve your current mood(2)')
        if h2=="1":
            rd_movies(2)
            y2=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
            recommend(y2, 2)
        elif h2=="2":
            rd_movies(5)
            y5=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
            recommend(y5, 5)

    elif x=='Sad':
        h2=input('Do you want to watch a movie that resonates with your current emotion(1) or do you want to improve your current mood(2)')
        if h2=="1":
            rd_movies(3)
            y3=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
            recommend(y3, 3)
        elif h2=="2":
            rd_movies(5)
            y5=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
            recommend(y5, 5)
    elif x=='Surprise':
        rd_movies(4)
        y4=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
        recommend(y4, 4)
    elif x=='Neutral':
        rd_movies(6)
        y6=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
        recommend(y6, 6)
    elif x=='Happy':
        rd_movies(5)
        y5=input('Enter the movie from above you want to be recommended similar movie on the basis of: ')
        recommend(y5, 5)
    
for i in range(200):
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imwrite("Face.jpg", frame)
    # try:
    detector = FaceDetector.build_model('opencv')
    faces_1 = FaceDetector.detect_faces(detector, 'opencv', frame)
    try:
        dim = faces_1[0][1]
    except:
        cv2.imshow("cam", frame)
        cv2.waitKey(10)
        continue
    cv2.rectangle(frame, (dim[0], dim[1]), (dim[0] + dim[2], dim[1] + dim[3]), (140, 140, 0), 2)
    roi = frame[dim[1]:dim[1] + dim[3], dim[0]:dim[0] + dim[2]]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(roi, (48, 48))
    img = img / 255.0
    pred = model.predict(np.array([img]), verbose = 0)
    # print(pred)
    pred[0][0]=pred[0][0]*1.7
    pred[0][1]
    pred[0][2]=pred[0][2]*0.8
    pred[0][3]
    pred[0][4]=pred[0][4]*0.65
    pred[0][5]=pred[0][5]*2.7
    pred[0][6]=0

    emotions.append(pred[0][:])
    # print(emotions)
    print(np.argmax(sum(emotions[-10:])))
    x = ["Fear", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"][np.argmax(sum(emotions[-10:]))]

   
    pred_string = get_class(np.argmax(pred))

    l1.append(pred[0])
    # print(pred_string)

    cv2.imshow("cam", frame)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
print(["Fear", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"][np.argmax(sum(emotions[-10:]))])

emotion(x)

plt.plot(l1)
plt.legend(["Fear", "Disgust", "Sad", "Surprise", "Angry", "Happy", "Neutral"])
plt.grid()
plt.show()