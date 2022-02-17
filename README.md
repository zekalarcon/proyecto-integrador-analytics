# Proyecto Integrador Python Analytics

Detector de emociones

Se tomo un dataset de emociones de kaggle (https://www.kaggle.com/ananthu017/emotion-detection-fer), al cual se le elimino la carpeta "disgusted" tanto en train, como en test.
De esta manera quedan 35,340 imagenes y 6 clases.

Al crear los generadores train y test, se hizo data augmentation y se fueron probando diferentes batch_size, rotation_range, zoom_range, horizontal_flip. 
Los resultados fueron insignificatentes,

Se probaron 4 modelos de Transfer Learnig y un Sequential.
En todos se fue alternando la cantidad de units(outputs), como tambien se quitaron y agregaron capas densas.
Resultados similares, y en algunos casos overfitting desde el comienzo al entrenar el modelo.

# Modelo Sequential + Xception

- model = Sequential()
- model.add(xception_base)
- Aplanamos la salida.
- model.add(Flatten()) 
- Última capa totalmente conectada con activación de softmax para la deteccion de emociones.
- model.add(Dense(units=out_shape, activation='softmax'))

Con una accuracy entre los 40. Overfitting

![Xception-accuracy-46%](https://user-images.githubusercontent.com/67808305/154382274-4edae351-1ce9-429c-b5e4-823a995f0711.jpg)

# Modelo Sequential + InceptionV3

- model = Sequential()
- model.add(inception_v3_base)
- Aplanamos la salida.
- model.add(Flatten())
- Primera capa completamente conectada con activacion relu.
- model.add(Dense(units=512, activation='relu'))
- model.add(Dropout(rate=0.8))
- model.add(Dense(units=256, activation='relu'))
- model.add(Dropout(rate=0.5))
- model.add(Dense(128, activation='relu'))
- model.add(Dropout(rate=0.2))
- model.add(Dense(64, activation='relu'))
- model.add(Dropout(rate=0.2))
- Última capa totalmente conectada con activación de softmax para la deteccion de emociones.
- model.add(Dense(units=out_shape, activation='softmax'))

Con una accuracy de 40%. Overfitting

![InceptionV3-accuracy-40%](https://user-images.githubusercontent.com/67808305/154382407-dea41e15-3bfb-44e5-8bc3-fb776f7f4202.jpg)

# Modelo Sequential + ResNet50

- model = Sequential()
- model.add(xception_base)
- Aplanamos la salida.
- model.add(Flatten()) 
- Última capa totalmente conectada con activación de softmax para la deteccion de emociones.
- model.add(Dense(units=out_shape, activation='softmax'))

Con una accuracy entre los 30. Overfitting

![ResNet50-accuracy-38%](https://user-images.githubusercontent.com/67808305/154382422-3ef0ca47-9636-4c43-b224-305b21c1090b.jpg)

# Modelo Sequential + VGG16

- model = Sequential()
- model.add(inception_v3_base)
- Aplanamos la salida.
- model.add(Flatten())
- Primera capa completamente conectada con activacion relu.
- model.add(Dense(units=512, activation='relu'))
- model.add(Dropout(rate=0.8))
- model.add(Dense(units=256, activation='relu'))
- model.add(Dropout(rate=0.5))
- model.add(Dense(128, activation='relu'))
- model.add(Dropout(rate=0.2))
- model.add(Dense(64, activation='relu'))
- model.add(Dropout(rate=0.2))
- Última capa totalmente conectada con activación de softmax para la deteccion de emociones.
- model.add(Dense(units=out_shape, activation='softmax'))

Con una accuracy entre los 40. Overfitting

![VGG16-accuracy-42%](https://user-images.githubusercontent.com/67808305/154382448-11af6af3-e8b1-4096-85d6-398fd8c1b681.jpg)

# Modelo Sequential

Con este modelo y capas se obtiene el mejor accuraccy 67%.
Seguimos teniendo overfitting.

- model = Sequential()
- model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu", input_shape=(48, 48, 1)))
- model.add(BatchNormalization())
- model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu"))
- model.add(BatchNormalization())
- model.add(MaxPooling2D(pool_size=(2,2)))
- model.add(Dropout(0.5))
- model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu"))
- model.add(BatchNormalization())
- model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same', activation="relu"))
- model.add(BatchNormalization())
- model.add(MaxPooling2D(pool_size=(2,2)))
- model.add(Dropout(0.5))
- model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu"))
- model.add(BatchNormalization())
- model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation="relu"))
- model.add(BatchNormalization())
- model.add(MaxPooling2D(pool_size=(2,2)))
- model.add(Dropout(0.5))

- model.add(Flatten())
- model.add(Dense(128, activation="relu"))
- model.add(BatchNormalization())
- model.add(Dropout(0.5))

- model.add(Dense(out_shape, activation="softmax"))



# Conclusion

Al entrenar todos los modelos encontramos overfitting, por lo tanto, el dataset necesita mas imagenes para cada emocion.
La otra alternativa, como sugiere -Hernan Contigiani-, seria utilizar algun modelo que se enfoque en los detalles, como por ej. OSNet.



 
