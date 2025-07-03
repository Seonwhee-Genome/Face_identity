import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from deepface.models.facial_recognition import Facenet

os.environ["DEEPFACE_HOME"] = "/home/work/Face/"

FACENET512_WEIGHTS = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
)



def load_and_add_finetune_layers(nc, do=0.5):
    model = Facenet.load_facenet512d_model()
    x = model.output
    x = Dropout(do)(x)
    predictions = Dense(nc, activation='softmax')(x)
    # Combine base model and new head into a new model
    model = Model(inputs=model.input, outputs=predictions)
    
    return model

    
def load_dataset(data_dir, BATCH_SIZE=32, IMG_SIZE=(160,160), mode="categorical", shuffle=True):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode=mode,  # use "int" if you want sparse_categorical_crossentropy
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )
    # Optional: cache, prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset
    



def train(epochs=10):
    facenet512 = Facenet.FaceNet512dClient()
    
    print("load dataset")
    train_dataset = load_dataset("/home/work/Face/train/imgs_merged2")
    
    print("initialize Facenet 512")

    model = load_and_add_finetune_layers(86145)
    
    print("model compile")

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),  # using a small learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_dataset, epochs=epochs)
