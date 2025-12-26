import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Путь к модели
MODEL_PATH = "models/butterfly_model.h5"
IMG_SIZE = 224

# Список классов в том же порядке, что class_indices при обучении
CLASSES = [
    'Aglais_io', 
    'Apoda_limacodes', 
    'Heterogenea_asella', 
    'Hyphantria_cunea', 
    'Orosanga_japonicus', 
    'Papilio_machaon', 
    'Vanessa_atalanta'
]

# Загружаем модель
model = tf.keras.models.load_model(MODEL_PATH)

def predict_class(image):
    """
    image: PIL.Image
    Возвращает предсказанный класс бабочки.
    """
    # Преобразуем изображение
    img = image.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)  # для batch

    # Предсказание
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = CLASSES[class_idx]
    confidence = float(preds[0][class_idx])

    return f"{class_name} ({confidence*100:.2f}%)"

# Интерфейс Gradio
iface = gr.Interface(
    fn=predict_class,
    inputs=gr.Image(type="pil", label="Upload butterfly image"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Butterfly Classifier",
    description="Upload an image of a butterfly and the model will predict its species."
)

if __name__ == "__main__":
    iface.launch()
