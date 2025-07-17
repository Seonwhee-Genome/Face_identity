import tensorflow as tf
import numpy as np
import itertools, random, time, os, datetime, pathlib
from functools import partial


IMG_SIZE            = 160
PEOPLE_PER_BATCH    = 45
IMAGES_PER_PERSON   = 40            # (= K in “PK‑sampling” literature)
BATCH_SIZE          = 90 
ALPHA               = 0.2 

def make_filelists(root):
    """Return dict[class_id] -> list[paths]."""
    root = pathlib.Path(root).expanduser()
    classes = sorted([p for p in root.iterdir() if p.is_dir()])
    return {cid: sorted([str(f) for f in p.glob("*")])
            for cid, p in enumerate(classes)}


def sample_people(filelists, P=PEOPLE_PER_BATCH, K=IMAGES_PER_PERSON):
    """Replicates Facenet's `sample_people`."""
    chosen_cls = np.random.choice(list(filelists.keys()), size=P, replace=False)
    image_paths, per_class = [], []
    for cid in chosen_cls:
        paths = filelists[cid]
        random.shuffle(paths)
        n = min(K, len(paths))
        image_paths.extend(paths[:n])
        per_class.append(n)
    return image_paths, per_class



def decode_and_preprocess(filename, random_crop=False, random_flip=False):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [160, 160])
    if random_flip:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image



def triplet_dataset(filelists, embed_model):
    """
    Yields batches shaped (batch, 3, H, W, C) ready for model.fit.
    Mining is done on‑the‑fly in numpy (fast enough for CPU),
    then images are loaded via tf.data parallelism.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def generator():
        while True:
            # 1) PK sample
            image_paths, per_class = sample_people(filelists)
            # 2) run forward pass to compute embeddings
            imgs = tf.stack([decode_and_preprocess(p, False, False)
                             for p in image_paths])
            emb = embed_model(imgs, training=False)
            emb = tf.math.l2_normalize(emb, axis=1).numpy()
            # 3) numpy hard‑negative mining
            trips, _, _ = select_triplets_numpy(
                emb, per_class, image_paths, PEOPLE_PER_BATCH, ALPHA)
            if len(trips) == 0:
                continue
            random.shuffle(trips)
            for a, p, n in trips:
                for path in (a, p, n):
                    yield path

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )

    ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    # 👇 assign dummy label (for Keras model.fit compatibility)
    ds = ds.map(lambda x: (x, tf.zeros((tf.shape(x)[0], 1))),
                num_parallel_calls=AUTOTUNE)

    return ds


# -- numpy port of the original select_triplets() ----------------
def select_triplets_numpy(embeddings, nrof_per_class, img_paths,
                           people_per_batch, alpha):
    triplets, trip_idx, num_trips = [], 0, 0
    e_start = 0
    for i in range(people_per_batch):
        n_imgs = int(nrof_per_class[i])
        for j in range(1, n_imgs):
            a_idx = e_start + j - 1
            neg_dists = np.sum((embeddings[a_idx] - embeddings)**2, axis=1)
            for pair in range(j, n_imgs):
                p_idx     = e_start + pair
                pos_dist  = np.sum((embeddings[a_idx]-embeddings[p_idx])**2)
                neg_dists[e_start:e_start+n_imgs] = np.NaN
                all_neg   = np.where(neg_dists - pos_dist < alpha)[0]
                if len(all_neg):
                    n_idx = np.random.choice(all_neg)
                    triplets.append((img_paths[a_idx],
                                     img_paths[p_idx],
                                     img_paths[n_idx]))
                    trip_idx += 1
                num_trips += 1
        e_start += n_imgs
    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)