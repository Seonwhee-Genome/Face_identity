import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from deepface.models.facial_recognition import Facenet
import itertools, random, time, datetime, pathlib
from functools import partial
from facenet_datasets import make_filelists, triplet_dataset



os.environ["DEEPFACE_HOME"] = "/home/work/Face/"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

FACENET512_WEIGHTS = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
)

EMB_SIZE            = 512
ALPHA               = 0.2           # margin
MAX_EPOCHS          = 3500
LR_BASE             = 0.15
LR_DECAY_EPOCHS     = 100
LR_DECAY_FACTOR     = 1.0
WEIGHT_DECAY        = 0.001
KEEP_PROB           = 1.0
LOG_DIR             = pathlib.Path("~/Face/logs/facenet_tf2").expanduser()
MODEL_DIR           = pathlib.Path("~/Face/models/facenet_tf2").expanduser()
GPU_MEM_FRACTION    = 1.0
SEED                = 666
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)


class TripletVisualizationCallback(Callback):
    def __init__(self, triplet_ds, log_dir, num_triplets=5, every_n_epochs=1):
        super().__init__()
        self.triplet_ds = triplet_ds  # should yield (x, y)
        self.num_triplets = num_triplets
        self.every_n_epochs = every_n_epochs
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n_epochs != 0:
            return

        # Take one batch and reshape triplets
        for x_batch, _ in self.triplet_ds.take(1):
            B = tf.shape(x_batch)[0]
            B3 = (B // 3) * 3
            x_triplets = tf.reshape(x_batch[:B3], [-1, 3, *x_batch.shape[1:]])  # (T, 3, H, W, C)

            # Convert to image grid
            for i in range(min(self.num_triplets, tf.shape(x_triplets)[0])):
                a, p, n = x_triplets[i]
                merged = tf.concat([a, p, n], axis=1)  # horizontal strip: (H, W*3, C)
                merged = tf.expand_dims(merged, 0)    # add batch dimension

                with self.writer.as_default():
                    tf.summary.image(f"Triplet_{i}", merged, step=epoch)
            self.writer.flush()
            break  # only log one batch
            

def triplet_loss(y_true, y_pred):
    B = tf.shape(y_pred)[0]
    B3 = (B // 3) * 3
    y_pred = y_pred[:B3]

    reshaped = tf.reshape(y_pred, [-1, 3, EMB_SIZE])
    anchor, positive, negative = tf.unstack(reshaped, axis=1)

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + ALPHA
    loss = tf.maximum(basic_loss, 0.0)

    # Save this for logging manually if needed:
    tf.summary.scalar("triplet_loss_mean", tf.reduce_mean(loss))

    return tf.reduce_mean(loss)


def load_and_add_finetune_layers():
    model = Facenet.load_facenet512d_model()
    model = Model(inputs=model.input, outputs=model.output)
    
    return model
    


def train(epochs=10, nc=86145, batch_size=16, checkpoint_path="/home/work/Face/arcface-tf2/checkpoints7/cp-{epoch:04d}.ckpt"):
    facenet512 = Facenet.FaceNet512dClient()
    
        
    print("initialize Facenet 512")

    model = load_and_add_finetune_layers()
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LR_BASE,   decay_steps=LR_DECAY_EPOCHS, decay_rate=LR_DECAY_FACTOR,
        staircase=True)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    log_dir = "logs/triplets/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    
    print("model compile")

    # Compile the model
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
    DATA_ROOT = "/home/work/Face/train/imgs_merged3"
    filelists = make_filelists(DATA_ROOT)

    ds = triplet_dataset(filelists, model)
    
    triplet_vis_cb = TripletVisualizationCallback(
        triplet_ds=ds,  # your triplet dataset
        log_dir=log_dir,
        num_triplets=5,       # how many triplets to visualize
        every_n_epochs=1
    )
    cbs = cbs + [tensorboard_callback, triplet_vis_cb]
    
    # Because dataset is infinite, we set steps_per_epoch = epoch_size
    model.fit(ds,
              steps_per_epoch=1000,              # epoch_size like before
              epochs=MAX_EPOCHS,
              callbacks=cbs)

    
if __name__=="__main__":
    train(epochs=2500)
