import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from model import get_model
from utils import predict_orientation_aggregate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Загрузка модели
model = get_model(num_classes=4, pretrained=True).to(DEVICE)
model.load_state_dict(torch.load('orientation_mobilenet.pth', map_location=DEVICE))
model.eval()

class OrientationApp:
    def __init__(self, master):
        self.master = master
        master.title('Document Orientation Correction')

        # Метка для угла
        self.angle_label = tk.Label(master, text='Predicted rotation: N/A')
        self.angle_label.pack()

        # Кнопки
        self.load_button = tk.Button(master, text='Load Image', command=self.load_image)
        self.load_button.pack()

        self.correct_button = tk.Button(master, text='Correct Orientation', command=self.correct_orientation)
        self.correct_button.pack()

        # Канвасы для изображений
        self.before_canvas = tk.Canvas(master, width=400, height=400)
        self.before_canvas.pack(side='left')
        self.after_canvas = tk.Canvas(master, width=400, height=400)
        self.after_canvas.pack(side='right')

        self.image = None
        self.tk_before = None
        self.tk_after = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert('RGB')
            self.display_image(self.image, self.before_canvas, 'tk_before')

    def display_image(self, image, canvas, attr_name):
        # Масштабируем для канваса
        img = image.copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(200, 200, image=tk_img)
        setattr(self, attr_name, tk_img)  # сохранить ссылку, чтобы не удалялось сборщиком мусора

    def correct_orientation(self):
        if self.image is None:
            return
        try:
            angle = predict_orientation_aggregate(self.image, model, device=DEVICE)
            corrected = self.image.rotate(-angle, expand=True)
            self.display_image(corrected, self.after_canvas, 'tk_after')
            self.angle_label.config(text=f'Predicted rotation: {angle}°')
        except Exception as e:
            self.angle_label.config(text=f'Error: {e}')

root = tk.Tk()
app = OrientationApp(root)
root.mainloop()