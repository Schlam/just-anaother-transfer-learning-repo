

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:-10]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)



epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2,
                 callbacks=[reduce_lr])
plot_hist(hist)