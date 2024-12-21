import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2


from PIL import Image
from keras.models import Model
from keras.applications import EfficientNetB0 , ResNet50 , EfficientNetB7
from keras.layers import (Layer , GlobalAveragePooling2D,Activation,MaxPooling2D,Add,Conv2D,MaxPool2D,Dense,Flatten,InputLayer,BatchNormalization,Input,
                          Embedding,Permute,Dropout,RandomFlip,RandomRotation,LayerNormalization,MultiHeadAttention,
                          RandomContrast,Rescaling,Resizing,Reshape,Cropping2D)
from keras.losses import BinaryCrossentropy,CategoricalCrossentropy,SparseCategoricalCrossentropy

from keras.metrics import Accuracy,TopKCategoricalAccuracy,CategoricalAccuracy,SparseCategoricalAccuracy
from keras.metrics import BinaryAccuracy, Accuracy,FalseNegatives,FalsePositives,TruePositives,TrueNegatives,Precision,Recall,AUC,binary_accuracy,CategoricalCrossentropy
from keras.optimizers import Adam, Adadelta
from keras.callbacks import Callback,EarlyStopping,LearningRateScheduler,ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import L1,L2

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 224,
    "LEARNING_RATE": 0.001,
    "N_EPOCHS": 10,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 4,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["Anthracite", "Bituminous", "Lignite","Peat"],
}

data_directory = "C:\\Users\\furka\\Desktop\\KomurSiniflandirma_4-7-24-main\\KomurSiniflandirma_4-7-24-main\\Dayi\\Coal Classification"

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

def dataset_loading():
    train_dataset = keras.utils.image_dataset_from_directory(
    data_directory,
    labels = "inferred",
    label_mode="categorical",
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode="rgb",
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
    validation_split=0.3,
    subset="training"
)
    validation_dataset = keras.utils.image_dataset_from_directory(
    data_directory,
    labels = "inferred",
    label_mode="categorical",
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode="rgb",
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
    validation_split=0.2,
    subset="validation"
)
    test_dataset = keras.utils.image_dataset_from_directory(
    data_directory,
    labels = "inferred",
    label_mode="categorical",
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode="rgb",
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
    validation_split=0.1,
    subset="validation"
)
    print(train_dataset)
    print(test_dataset)
    print(validation_dataset)
    return train_dataset,validation_dataset,test_dataset

#Veri Setlerinin Yüklenmesi
train_dataset,validation_dataset,test_dataset = dataset_loading()

#Eğitim Veri Kümesinden Örnek Görsellerin Görselleştirilmesi
plt.figure(figsize=(12,12))

for images,labels in train_dataset.take(2):
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(images[i]/255)
        plt.title(CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()])
        plt.axis("off")

# https://www.tensorflow.org/guide/data_performance
# Veri Performansını Optimize Etme
train_dataset = (train_dataset.prefetch(tf.data.AUTOTUNE))
validation_dataset = (validation_dataset.prefetch(tf.data.AUTOTUNE))


#Doğrulama Veri Kümesinden Örnek Görsellerin Görselleştirilmesi
plt.figure(figsize=(12,12))

for images,labels in validation_dataset.take(1):
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(images[i]/255)
        plt.title(CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()])
        plt.axis("off")
    


print(train_dataset)
print(validation_dataset)



# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
#Eğitim sırasında modelin performansını izler ve en iyi doğrulama (validation) doğruluğunu (val_accuracy) elde eden modeli kaydeder.
checkpoint = ModelCheckpoint("fine_tuned_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

#Eğitim sırasında doğrulama kaybı (val_loss) belirli bir süre boyunca iyileşmezse (patience), eğitim sürecini erken sonlandırır.
early_stopping = EarlyStopping(
    monitor = "val_loss",
    min_delta=0,
    patience =6,
    verbose=0,
    mode = "auto",
    baseline = None,
    restore_best_weights = False
)
#Modelin performansını çeşitli metriklere göre değerlendiri
metrics = [
    Accuracy(name="accuracy"),
    TruePositives(name="tp"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    FalseNegatives(name = "fn"),
    Precision(name = "precision"),
    Recall(name= "recall"),
    AUC(name = "auc"),

]


# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

def build_efficentB0(num_classes):
    #Modelin Girdileri ve EfficientNetB0 Modelinin Yüklenmesi
    inputs = Input(shape=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"],3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
    #Modelin Dondurulması
    model.trainable = False
    # Kendi model yapımı

    x = GlobalAveragePooling2D(name = "avg_pool")(model.output)
    x = BatchNormalization()(x)

    #%20 oranında nöronları rastgele devre dışı bırakır.
    #Aşırı öğrenmeyi (overfitting) önlemek için kullanılır.
    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate,name = "top_dropout")(x)

    outputs = Dense(num_classes,activation="softmax",name = "pred")(x)

    # Modelin oluşturulması
    model = Model(inputs,outputs,name = "EfficentNet")

    #Modelin Derlenmesi
    optimizer = Adam(learning_rate=0.00001)
    model.compile(
        optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# https://www.tensorflow.org/guide/keras/transfer_learning

def unfreeze_model(model, start_layer, end_layer):
    for layer in model.layers[start_layer:end_layer]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

     #Modelin Yeniden Derlenmesi:
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])


#Modelin Oluşturulması
model = build_efficentB0(CONFIGURATION["NUM_CLASSES"])

#Modelin son 10 katmanı serbest bırakılır
unfreeze_model(model,-10,None)
#Modelin Eğitimi
history = model.fit(train_dataset,validation_data=validation_dataset,epochs=10,callbacks = [checkpoint,early_stopping])

#Modelin Kaydedilmesi
model.save("efficent_net_efficent224.h5")
model.save("efficent_net_efficent224.keras")
#Modelin Değerlendirilmesi
model.evaluate(test_dataset)