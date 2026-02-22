import tensorflow as tf
from tensorflow.keras import layers, models, datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Resizing(224, 224, input_shape=(32, 32, 3)),
    
    layers.Conv2D(96, (11, 11), strides=4, padding='valid', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )

history = model.fit(
    train_images, train_labels, 
    epochs=10, 
    batch_size=64,
    validation_data=(test_images, test_labels),
    verbose=1
)

sample_batch = test_images[:5]
predictions = model.predict(sample_batch)

print("Predictions for first 5 images:", predictions.argmax(axis=1))
print("Actual Labels:", test_labels[:5].flatten())