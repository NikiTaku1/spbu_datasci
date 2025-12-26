import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os, json, yaml

# Список классов бабочек
CLASSES = [
    "Apoda_limacodes",
    "Hyphantria_cunea",
    "Orosanga_japonicus",
    "Heterogenea_asella",
    "Aglais_io",
    "Vanessa_atalanta",
    "Papilio_machaon"
]

# Ограничение на максимальное количество изображений в классе для ускорения обучения
MAX_IMAGES_PER_CLASS = 500

# Загружаем параметры обучения из params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

IMG_SIZE = params['train']['img_size']            # размер изображений
BATCH_SIZE = params['train']['batch_size']        # размер батча
EPOCHS = params['train']['epochs']                # количество эпох
LR = params['train']['lr']                        # learning rate
DROPOUT_RATE = params['train'].get('dropout_rate', 0.3)  # dropout (по умолчанию 0.3)
DATASET_PATH = 'dataset'                          # путь к датасету

# Восстанавливаем файлы, которые были переименованы с .skip_ с предыдущих запусков
# Это делается для того, чтобы снова использовать их при новом обучении
for cls in CLASSES:
    class_dir = os.path.join(DATASET_PATH, cls)
    for fname in os.listdir(class_dir):
        if fname.startswith(".skip_"):
            os.rename(os.path.join(class_dir, fname), os.path.join(class_dir, fname.replace(".skip_", "")))

# Ограничиваем количество изображений в классах
# Если в классе больше MAX_IMAGES_PER_CLASS, случайным образом оставляем только MAX_IMAGES_PER_CLASS изображений
for cls in CLASSES:
    class_dir = os.path.join(DATASET_PATH, cls)
    images = [f for f in os.listdir(class_dir) if not f.startswith(".skip_")]
    if len(images) > MAX_IMAGES_PER_CLASS:
        keep_images = np.random.choice(images, MAX_IMAGES_PER_CLASS, replace=False)
        # Остальные изображения переименовываются с .skip_ чтобы их временно исключить
        for img_name in set(images) - set(keep_images):
            os.rename(os.path.join(class_dir, img_name), os.path.join(class_dir, f".skip_{img_name}"))

# Data augmentation и разделение на train/validation
train_datagen = ImageDataGenerator(
    rescale=1./255,               # нормализация пикселей
    validation_split=0.2,         # 20% данных оставляем на валидацию
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Генератор для валидации
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Вычисляем веса классов для компенсации дисбаланса
classes = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Создание модели на основе MobileNetV2 (предобученной на ImageNet)
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

# Замораживаем базовую сеть, чтобы не обучать её веса сразу
base_model.trainable = False

# Добавляем глобальный слой пулинга и dropout для регуляризации
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)

# Финальный слой с количеством нейронов = числу классов
predictions = Dense(len(CLASSES), activation='softmax')(x)

# Собираем модель
model = Model(inputs=base_model.input, outputs=predictions)

# Компилируем модель с оптимизатором Adam и categorical crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучаем модель
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights_dict  # учитываем веса классов
)

# Сохраняем обученную модель
os.makedirs('models', exist_ok=True)
model.save('models/butterfly_model.h5')

# Сохраняем список классов
with open('models/classes.json', 'w') as f:
    json.dump(CLASSES, f)

# Сохраняем метрики для DVC
metrics = {
    "train_accuracy": float(history.history['accuracy'][-1]),
    "val_accuracy": float(history.history['val_accuracy'][-1])
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Training finished. Metrics saved to metrics.json")
