import tensorflow as tf
from tensorflow.contrib.keras.models import Sequential
from tensorflow.contrib.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D
import numpy as np
import os
import random
import cv2
import pickle

def save_output(array):
    out = open('out.pickle', 'wb')
    pickle.dump(array, out)
    out.close()

def load_file(file):
    reader = open(file, 'rb')
    return pickle.load(reader)


DATADIR = './PetImages' # Isso supoe que Dataset esteja no mesmo diretório que o arquivo
CATEGORIES = ['Cat', 'Dog']
t_data = []
X = []
y = []

'''
Esse código é responsável pela criação dos dados de treino
do modelo, basicamente iteramos sobre os arquivos de dentro
das subpastas do dataset, normalizamos elas deixando-as preto
e branco, utilizamos o resize para deixar todas com 50x50 e
fazemos um append de uma tupla com (imagem, categoria) para
t_data.
'''
for item in CATEGORIES:
    path = os.path.join(DATADIR, item)
    for image in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (50, 50))
            t_data.append((new_array, CATEGORIES.index(item)))
        except Exception as e:
            pass

random.shuffle(t_data)
for image, label in t_data:
    X.append(image)
    y.append(label)

'''    
Salva os arquivos localmente
'''
save_output(X)
save_output(y)

'''
Construção do Modelo
'''
X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('sigmoid'))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, validation_split=0.1, epochs=10)
