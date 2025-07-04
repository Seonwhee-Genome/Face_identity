import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from deepface.models.facial_recognition import Facenet
import itertools, random, time, datetime, pathlib
from functools import partial
from facenet_datasets import make_filelists, triplet_dataset



os.environ["DEEPFACE_HOME"] = "/home/work/Face/"

FACENET512_WEIGHTS = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
)

EMB_SIZE            = 512
ALPHA               = 0.2           # margin
MAX_EPOCHS          = 500
LR_BASE             = 0.1
LR_DECAY_EPOCHS     = 100
LR_DECAY_FACTOR     = 1.0
WEIGHT_DECAY        = 0.0
KEEP_PROB           = 1.0
LOG_DIR             = pathlib.Path("~/Face/logs/facenet_tf2").expanduser()
MODEL_DIR           = pathlib.Path("~/Face/models/facenet_tf2").expanduser()
GPU_MEM_FRACTION    = 1.0
SEED                = 666
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)


def triplet_loss(y_true, y_pred):
    # y_true is dummy (Keras requires it) — batch is multiple of 3
    anchor, positive, negative = tf.unstack(
        tf.reshape(y_pred, [-1, 3, EMB_SIZE]), axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + ALPHA
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


def load_and_add_finetune_layers():
    model = Facenet.load_facenet512d_model()
    model = Model(inputs=model.input, outputs=model.output)
    
    return model
    


def train(epochs=10, nc=86145, batch_size=8, checkpoint_path="/home/work/Face/arcface-tf2/checkpoints5/cp-{epoch:04d}.ckpt"):
    facenet512 = Facenet.FaceNet512dClient()
    
        
    print("initialize Facenet 512")

    model = load_and_add_finetune_layers()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LR_BASE,   decay_steps=LR_DECAY_EPOCHS, decay_rate=LR_DECAY_FACTOR,
        staircase=True)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    
    
    
    print("model compile")

    # Compile the model
#     loss_obj   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,  # using a small learning rate
                  loss=triplet_loss)
    
    # Logging / checkpoint directory setup
    timestamp   = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path    = LOG_DIR / timestamp
    model_path  = MODEL_DIR / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir=str(log_path)),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path / "ckpt_{epoch:04d}.weights.h5"),
            save_weights_only=True, save_best_only=False, verbose=1)
    ]
    DATA_ROOT = "/home/work/Face/train/imgs_merged2"
    filelists = make_filelists(DATA_ROOT)

    ds = triplet_dataset(filelists, model)
    
    # Because dataset is infinite, we set steps_per_epoch = epoch_size
    model.fit(ds,
              steps_per_epoch=1000,              # epoch_size like before
              epochs=MAX_EPOCHS,
              callbacks=cbs)

    
if __name__=="__main__":
    train(epochs=100)
