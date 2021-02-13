
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm
from extract_bottleneck_features import *
import cv2 as cv
import matplotlib.pyplot as plt



#funzione per creare il grafico dei valori di accuratezza
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#funzione per creare il grafico dei valori persi
def plot_losses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


#Funzione principale del programma(o main) essa richiama la perdizione e stampa l'immagine con il nome della sua predizione con un accuratezza dell' 81%
def Dog_Recognize_app(imgpath, model, dog_names):
    predict = Resnet50_prediction_breed(imgpath, model)
    img = cv.imread(imgpath)
    cv.putText(img, 'Predizione razza: {}'.format(dog_names[predict]), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 0), 2)
    cv.imshow("Predizione ", img)
    cv.waitKey(0)


#funione che calcola la predizione di Resnet50
def Resnet50_prediction_breed(img_path, model):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return np.argmax(predicted_vector,-1)

#funzione che estrae le feature a collo di bottiglia
def get_bottleneck_features(path):
    bottleneck_features = np.load(path)
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    return train, valid, test

#fuzione utilizzata per poter utilizzare le immagini per keras
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

#fuzione utilizzata per poter utilizzare i file per keras
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def test_model(model, test_tensors, test_targets, name):
    # prendere il valore di perdizione per ogni immagine del test set
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # accuratezza del test
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
    print(f'Accuratezza test {name}: {round(test_accuracy, 4)}%')


# definisco la funzione per caricare train, test e validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets
