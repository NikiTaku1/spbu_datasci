"""
Generates images and annotations CSV for the orientation benchmark.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
import csv
import textwrap

FONT_CYR = None
CANDIDATE_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf"
]
for p in CANDIDATE_FONTS:
    if os.path.exists(p):
        FONT_CYR = p
        break
if FONT_CYR is None:
    print("Warning: Cyrillic-capable font not found. Install DejaVuSans or adjust CANDIDATE_FONTS.")

PAGE_SIZES = [(1240, 1754), (1000, 1400), (1240, 1600)]  # разные размеры страниц


def make_realistic_doc(text: str) -> Image.Image:
    size = random.choice(PAGE_SIZES)
    img = Image.new("RGB", size, "#fafafa")
    draw = ImageDraw.Draw(img)

    f_header = ImageFont.truetype(FONT_CYR if FONT_CYR else CANDIDATE_FONTS[0], random.randint(30, 50))
    f_body = ImageFont.truetype(FONT_CYR if FONT_CYR else CANDIDATE_FONTS[0], random.randint(18, 24))
    f_mono = ImageFont.truetype(FONT_CYR if FONT_CYR else CANDIDATE_FONTS[0], random.randint(14, 22))

    header_x, header_y = random.randint(40, 80), random.randint(20, 60)
    body_x, body_y = random.randint(40, 80), header_y + random.randint(100, 160)

    draw.text((header_x, header_y), "Организация: ООО Пример", font=f_header, fill="#111")
    draw.text((header_x, header_y + random.randint(40, 80)), "Адрес: ул. Примерная, 12, Москва", font=f_body, fill="#222")

    wrapped = textwrap.fill(text, width=random.randint(60, 90))
    draw.multiline_text((body_x, body_y), wrapped, font=f_body, fill="#111", spacing=6)

    table_x, table_y = random.randint(40, 80), body_y + random.randint(200, 400)
    cols = [0, random.randint(200, 260), random.randint(500, 600), size[0]-50]
    row_h = random.randint(30, 50)
    for r in range(6):
        y = table_y + r * row_h
        draw.line((table_x, y, table_x + cols[-1], y), fill="#333")
    for c in cols:
        draw.line((table_x + c, table_y, table_x + c, table_y + 6 * row_h), fill="#333")
    draw.text((table_x + 10, table_y + 10), "№", font=f_mono, fill="#111")
    draw.text((table_x + 260, table_y + 10), "Наименование", font=f_mono, fill="#111")

    stamp_center = (size[0]-random.randint(120, 180), size[1]-random.randint(120, 180))

    draw.ellipse((stamp_center[0]-80, stamp_center[1]-80, stamp_center[0]+80, stamp_center[1]+80), outline="#ff0000", width=4)
    draw.text((stamp_center[0]-40, stamp_center[1]-12), "Печать", font=f_mono, fill="#cc0000")

    noise = Image.effect_noise(size, 30).convert("L")
    noise = noise.point(lambda p: p * 0.15)
    img.paste(Image.composite(img, img.filter(ImageFilter.GaussianBlur(0.6)), noise), (0, 0))

    return img


# small perspective warp helper
from PIL import ImageTransform
import numpy as np

def perspective_warp(img: Image.Image) -> Image.Image:
    w, h = img.size
    shift = lambda v: v + random.randint(-40, 40)
    coeffs = find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [(shift(0), shift(0)), (shift(w), shift(0)), (shift(w), shift(h)), (shift(0), shift(h))]
    )
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def find_perspective_coeffs(pairs_src, pairs_dst):
    matrix = []
    for p1, p2 in zip(pairs_src, pairs_dst):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix)
    B = np.array([c for p in pairs_dst for c in p])
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()


def generate_dataset(out_dir='synthetic_orb_realistic', n=400):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    ann_path = os.path.join(out_dir, "annotations.csv")

    sample_texts = [
        "Счёт на оплату: Пожалуйста, оплатите до 01.12.2025. Благодарим за сотрудничество.",
        "Товар отгружен: 3 единицы товара X. Претензии принимаются в течение 14 дней.",
        "Протокол собрания: Обсуждение бюджета, назначение ответственных, утверждение сроков.",
        "Доверенность: Действительна до 2026 года. Подпись и печать обязательны.",
        "Уведомление: Требуется обновление данных клиента. Свяжитесь с отделом поддержки."
    ]

    rotations = [0, 90, 180, 270]
    rows = []

    for i in range(n):
        txt = random.choice(sample_texts)
        img = make_realistic_doc(txt)

        if random.random() < 0.3:
            img = perspective_warp(img)
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.8)))

        for r in rotations:
            out = img.rotate(r, expand=True, fillcolor="#fafafa")
            fname = f"doc_{i:04d}_{r}.jpg"
            out.save(os.path.join(img_dir, fname), quality=90)
            rows.append([fname, r])

    with open(ann_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle"])
        writer.writerows(rows)

    print(f"Generated dataset at: {out_dir} (images/, annotations.csv)")
    return out_dir