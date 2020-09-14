import tensorflow as tf
import tensorflow_datasets as tfds


# NUM_CLASSES = 120

# One-hot / categorical encoding
def input_preprocess(image, label):
    image = tf.image.resize(image, (224,224))
    label = tf.one_hot(label, 120)
    return image, label


def load_data(IMG_SIZE=224, batch_size=16, dataset_name="stanford_dogs", preprocess=input_preprocess):    
    ds_train, ds_test = tfds.load(
        dataset_name, split=["train", "test"], as_supervised=True)
        
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)
    
    return ds_train,ds_test

