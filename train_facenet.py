import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from deepface.models.facial_recognition import Facenet


def train():
    facenet512 = Facenet.FaceNet512dClient()
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 32

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "/data179/imgs_merged2",
        label_mode="categorical",  # use "int" if you want sparse_categorical_crossentropy
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Optional: cache, prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    x = model.output
    x = Dropout(0.5)(x)  # Adding dropout for regularization

    # Adding a new dense layer for classification
    num_classes = 86142  # Replace with your number of target classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine base model and new head into a new model
    model = Model(inputs=model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),  # using a small learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_dataset, epochs=10)
