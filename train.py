import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATA_DIR = 'Data'
MODEL_DIR = 'Model'
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15

os.makedirs(MODEL_DIR, exist_ok=True)

# Cargar imágenes desde Data/A, Data/B, ... (solo carpetas con imágenes)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='training', seed=42,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode='categorical')
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='validation', seed=42,
    image_size=IMG_SIZE, batch_size=BATCH, label_mode='categorical')

class_names = train_ds.class_names
print('Clases:', class_names)

# Guardar labels.txt (mismo formato de Teachable Machine: "0 A")
with open(os.path.join(MODEL_DIR, 'labels.txt'), 'w') as f:
    for i, name in enumerate(class_names):
        f.write(f'{i} {name}\n')

# Aumento de datos
augment = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(0.2),
])

# Normalizar a [-1, 1] fuera del modelo, porque cvzone Classifier ya hace
# esa normalización al predecir (evita normalizar dos veces).
norm = lambda x, y: (tf.cast(x, tf.float32) / 127.5 - 1.0, y)
train_ds = train_ds.map(lambda x, y: norm(augment(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(norm).prefetch(tf.data.AUTOTUNE)

# Modelo: MobileNetV2 preentrenado + capa de clasificación
base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Ajuste fino de las últimas capas del modelo base
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save(os.path.join(MODEL_DIR, 'keras_model.h5'))
print('Modelo guardado en Model/keras_model.h5')