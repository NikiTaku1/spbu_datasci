import os
import pandas as pd
import requests
from tqdm import tqdm

input_csv = "filtered.csv"       
output_dir = "dataset"       

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    species = row["scientific_name"].replace(" ", "_")
    img_url = row["image_url"]
    img_id = row["id"]

    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    img_path = os.path.join(species_dir, f"{img_id}.jpg")

    try:
        response = requests.get(img_url, timeout=10)

        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Ошибка {response.status_code} — {img_url}")

    except Exception as e:
        print(f"Ошибка загрузки {img_url}: {e}")

print("\nВсе изображения скачаны.")
