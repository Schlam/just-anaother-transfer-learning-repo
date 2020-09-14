import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing


import matplotlib.pyplot as plt

# Plot to visualize training progress
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




from load_data import load_data
ds_train, ds_test = load_data()


from build_model import build_model
model = build_model()



epochs = 8
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=2,factor=.1)


if __name__ == "__main__":
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, 
                    verbose=2,callbacks=[reduce_lr])

    plot_hist(hist)
