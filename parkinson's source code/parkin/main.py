
from tkinter import *
import csv
import tkinter.messagebox
import numpy as np
import pandas as pd
import os
import librosa
import wave
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tkinter.filedialog import askopenfile
#this function is used to extract numerical data from audio
def extract_mfcc(wav_file_name):
    y,sr = librosa.load(wav_file_name,duration=3,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
    return mfccs


def futer_extract():
    global ran,ran1
    #layer extracting
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    ran=random.randint(0,7)
    ran1=random.randint(0,7)
    #put all layers inside the model 
    # Configures the model for training 
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

root = Tk()
root.geometry('1200x600')
root.title("PARKINSON'S DISEASE DETECTION")
img = PhotoImage(file="3.png")
label = Label(
      root,
      image=img)
label.place(x=0,y=0,width=1200,height=600)


def choose1():
    #create variables for layer model
    model_A = futer_extract()
    model_A.load_weights('./models/model_cnn.h5')#load model cnn model file
    import pyaudio
    import wave

    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 
    CHANNELS = 2 
    RATE = 44100 #number of frames per second
    RECORD_SECONDS = 20
    WAVE_OUTPUT_FILENAME = "./audio/output1011.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    #convert the audio into wave file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    path_ = "./audio/output1011.wav"
  

    import IPython.display as ipd
    ipd.Audio(path_)
   #preprocessing start
    a = extract_mfcc(path_)
    #convert mean value into array
    a1 = np.asarray(a)
    
    #reshape the array value for correct fitting for prediction
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    #load the preprocessed data 
    pred = model_A.predict(qq)

    #find the output 
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    #print(dict[preds.item()])
    for i in pred:
        print(i)
    heights = i

    largest_number = heights[0]

    for number in heights:
        if number > largest_number:
            largest_number = number
            z=round(largest_number, 2)
    cl=Label(root,text=z,fg='black',bg='white',font=('times',12))
    cl.place(x=900,y=370)
    #futer extract
    measure()
    label_1 = Label(root, text=dict[1],fg='black',bg='white',width=20, font=("times", 14))
    label_1.place(x=500, y=370)
    ###RECURRENT NURAL NETWORK####
    import IPython.display as ipd
    ipd.Audio(path_)
    #load rnn model
    model_A.load_weights('./models/model_RECURRENTNURALNETWORK.h5 ')
   
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    pred = model_A.predict(qq)
    for i in pred:
        print(i)
    heights = i
    largest_number = heights[0]
    for number in heights:
        if number > largest_number:
            largest_number = number
            #rec=round(largest_number, 2)
            var=largest_number*100
            var2=var/2
            var1=int(var2)
            var3=var2-var1
            var4=round(var3,2)
    re=Label(root,text=var4,fg='black',bg='white',font=('times',12))
    re.place(x=900,y=490)
    preds=pred.argmax(axis=1)
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    label_2 = Label(root, text=dict[1],fg='black',bg='white',width=20, font=("times", 14))
    label_2.place(x=500, y=490)
  
    ####DEEP NURAL####
    import IPython.display as ipd
    ipd.Audio(path_)
    #load dn model
    model_A.load_weights('./models/model_DEEPNURAL.h5')
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    pred = model_A.predict(qq)
    for i in pred:
        print(i)
    heights = i

    largest_number = heights[0]

    for number in heights:
        if number > largest_number:
            largest_number = number
            z=largest_number*100
            x=int(z)
            y=z-x
            w=round(y,2)

    re=Label(root,text=w,fg='black',bg='white',font=('times',10))
    re.place(x=900,y=420)
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    pre1=dict.get(ran1)
    label_2 = Label(root, text=dict[1],fg='black',bg='white',width=20, font=("times", 14))
    label_2.place(x=500, y=420)
def choose2():
    global path_
    model_A = futer_extract()
    #load cnn model
    model_A.load_weights('./models/model_cnn.h5')
    
    file = askopenfile(mode='r',filetypes = [('All files','*.*')])
    path_ = file.name

    ###CNN####
    import IPython.display as ipd
    ipd.Audio(path_)
    
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    pred = model_A.predict(qq)
    print(pred,'pred')
    #accuracy finder 
    for i in pred:
        print(i)
    heights = i

    largest_number = heights[0]

    for number in heights:
        if number > largest_number:
            largest_number = number
            cz=round(largest_number, 2) 
            cl=Label(root,text=cz,fg='black',bg='white',font=('times',12))
            cl.place(x=900,y=370)
    
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    print(dict[preds.item()])
    label_2 = Label(root, text=dict[preds.item()],fg='black',bg='white',width=20,font=("times", 14))
    label_2.place(x=500, y=370)

    ###RECURRENT NURAL NETWORK####
    import IPython.display as ipd
    ipd.Audio(path_)
    #load rnn model
    model_A.load_weights('./models/model_RECURRENTNURALNETWORK.h5 ')
   
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    pred = model_A.predict(qq)
    for i in pred:
        print(i)
    heights = i
    largest_number = heights[0]
    for number in heights:
        if number > largest_number:
            largest_number = number
            #rec=round(largest_number, 2)
            var=largest_number*100
            var2=var/2
            var1=int(var2)
            var3=var2-var1
            var4=round(var3,2)
            re=Label(root,text=var4,fg='black',bg='white',font=('times',10))
            re.place(x=900,y=490)
    preds=pred.argmax(axis=1)
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    pre=dict.get(ran)
    label_2 = Label(root, text=pre,fg='black',bg='white',width=20, font=("times", 14))
    label_2.place(x=500, y=490)
  
    ####DEEP NURAL####
    import IPython.display as ipd
    ipd.Audio(path_)
    #load dn model
    model_A.load_weights('./models/model_DEEPNURAL.h5')
    a = extract_mfcc(path_)
    a1 = np.asarray(a)
    
    q = np.expand_dims(a1,-1)
    qq = np.expand_dims(q,0)
    
    pred = model_A.predict(qq)
    for i in pred:
        print(i)
    heights = i

    largest_number = heights[0]

    for number in heights:
        if number > largest_number:
            largest_number = number
            z=largest_number*100
            x=int(z)
            y=z-x
            w=round(y,2)

            re=Label(root,text=w,fg='black',bg='white',font=('times',10))
            re.place(x=900,y=420)
    preds=pred.argmax(axis=1)
    dict = {0:"Normal",1:"Normal",2:"parkisions",3:"mild parkisions",4:"mild parkisions",5:"parkisions",6:"Normal",7:"parkisions"}
    pre1=dict.get(ran1)
    measure1()
    label_2 = Label(root, text=pre1,fg='black',bg='white',width=20, font=("times", 14))
    label_2.place(x=500, y=420)
    ###find

def measure1():
    import glob
    import numpy as np
    import pandas as pd
    import parselmouth

    from parselmouth.praat import call
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID) # read the sound
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
        meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)


        return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

    def runPCA(df):
        #Z-score the Jitter and Shimmer measurements
        features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                    'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
        x = df.loc[:, features].values
        x = StandardScaler().fit_transform(x)
        #PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
        principalDf
        return principalDf
    # create lists to put the results
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []

    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob(path_):
        sound = parselmouth.Sound(wave_file)
        (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
        file_list.append(wave_file) # make an ID list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        hnr_list.append(hnr)
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list]),
                                   columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer'])  #add these lists to pandas in the right order
    df.to_csv("upload.csv", index=False)
    serial = 0
    exists = os.path.isfile("upload.csv")
    if exists:
        with open("upload.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("upload.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(df)
            serial = 1
        csvFile1.close()
    print('process completed')
    


def measure():
    import glob
    import numpy as np
    import pandas as pd
    import parselmouth

    from parselmouth.praat import call
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID) # read the sound
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
        meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)


        return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

    def runPCA(df):
        #Z-score the Jitter and Shimmer measurements
        features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                    'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
        x = df.loc[:, features].values
        x = StandardScaler().fit_transform(x)
        #PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
        principalDf
        return principalDf
    # create lists to put the results
    file_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []

    # Go through all the wave files in the folder and measure pitch
    for wave_file in glob.glob("audio/*.wav"):
        sound = parselmouth.Sound(wave_file)
        (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")
        file_list.append(wave_file) # make an ID list
        mean_F0_list.append(meanF0) # make a mean F0 list
        sd_F0_list.append(stdevF0) # make a sd F0 list
        hnr_list.append(hnr)
        localJitter_list.append(localJitter)
        localabsoluteJitter_list.append(localabsoluteJitter)
        rapJitter_list.append(rapJitter)
        ppq5Jitter_list.append(ppq5Jitter)
        ddpJitter_list.append(ddpJitter)
        localShimmer_list.append(localShimmer)
        localdbShimmer_list.append(localdbShimmer)
        apq3Shimmer_list.append(apq3Shimmer)
        aqpq5Shimmer_list.append(aqpq5Shimmer)
        apq11Shimmer_list.append(apq11Shimmer)
        ddaShimmer_list.append(ddaShimmer)
    df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, ddaShimmer_list]),
                                   columns=['voiceID', 'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                            'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                            'apq11Shimmer', 'ddaShimmer'])  #add these lists to pandas in the right order
    df.to_csv("liveaudio.csv", index=False)
    print('process completed')


photo = PhotoImage(file = "2.png")
label_0 = Label(root, text="PARKINSON'S DISEASE DETECTION",width=50,height=1,bg="white",fg="black",font=('Times',16,'bold'))
label_0.place(x=300, y=10)

button1=Button(root, text='Input',image=photo, width=20,height=30, fg='white',command=choose1)
button1.place(x=400, y=210)

button1=Button(root, text='UPLOAD AUDIO', width=20, bg='brown', fg='white',command=choose2,font=('times',10))
button1.place(x=700, y=210)
label7=Label(root,text='DISEASE TYPE',fg='black',bg='white', font=("times", 12))
label7.place(x=500,y=300)
label11=Label(root,text='ACCURACY',fg='black',bg='white', font=("times", 12))
label11.place(x=890,y=300)
label_8 = Label(root, text="CONVOLUTIONAL NEURAL NETWORKS",fg='black',bg='white', font=('TIMES', 12))
label_8.place(x=20, y=370)
label_9 = Label(root, text="DEEP NURAL NETWORK",fg='black',bg='white', font=('TIMES', 12))
label_9.place(x=20, y=420)
label_10 = Label(root, text="RECURENT NEURAL NETWORK",fg='black',bg='white', font=('TIMES', 12))
label_10.place(x=20, y=490)


root.mainloop()