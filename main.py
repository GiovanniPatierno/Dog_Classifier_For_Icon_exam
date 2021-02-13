from glob import glob
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
import def_function



# carico train, test, e validation datasets
train_files, train_targets = def_function.load_dataset('dogImages/train')
valid_files, valid_targets = def_function.load_dataset('dogImages/valid')
test_files, test_targets = def_function.load_dataset('dogImages/test')

# carico la lista dei nomi delle razze di cane
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
dog_names = np.array(dog_names)

# stampo delle statistiche sul dataset come per esempio il numero di categorie di cani presenti
print('Ci sono %d categorie totali.' % len(dog_names))
print('Ci sono %s immagini totali.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('Ci sono %d immagini di addestramento.' % len(train_files))
print('Ci sono %d immagini di validazione.' % len(valid_files))
print('Ci sono %d immagini di test.'% len(test_files))

#assegno le bottleneck features pesenti nel file DogResnet50
train_Resnet50, valid_Resnet50, test_Resnet50 = def_function.get_bottleneck_features('DogResnet50Data.npz')

#inizializzo il modello di Resnet50
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(train_Resnet50.shape[1:])))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.summary()

#compilo il suddetto modell0
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#addresto un moello creando prima però un checkpointer che identifica i migliori pesi riscontrati
checkpointer = ModelCheckpoint(filepath='weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)
#history = Resnet50_model.fit(train_Resnet50, train_targets, validation_data=(valid_Resnet50, valid_targets),epochs=100, batch_size=20, callbacks=[checkpointer], verbose=1)

#carico i pesi salvati nel file weights.best.Resnet50.hdf5
Resnet50_model.load_weights('weights.best.Resnet50.hdf5')

#testo l accuratezza del moello
def_function.test_model(Resnet50_model,test_Resnet50, test_targets, 'Resnet50')

#qui utilizzo le funzioni atte a stampare i grafici
#plot_accuracy(history)
#plot_losses(history)


#Qui eseguo i test per riscontrate se effettivamente la macchina è accurata,controllando l output ricevuto dalla mia funziode di gestione Dog_Recognize_app
def_function.Dog_Recognize_app('immagini per test/golden.jpg', Resnet50_model, dog_names)
def_function.Dog_Recognize_app('immagini per test/newfoundland.jpg', Resnet50_model,  dog_names)
def_function.Dog_Recognize_app('immagini per test/poodle.jpg', Resnet50_model,  dog_names)
def_function.Dog_Recognize_app('immagini per test/pug.jpg', Resnet50_model,  dog_names)
def_function.Dog_Recognize_app('immagini per test/Shetland.jpg', Resnet50_model,  dog_names)

