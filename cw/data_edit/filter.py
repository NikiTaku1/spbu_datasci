import pandas as pd

input_csv = "observations-660936.csv"
output_csv = "filtered.csv"

target_species = [
    "Apoda limacodes",
    "Hyphantria cunea",
    "Orosanga japonicus",
    "Heterogenea asella",
    "Aglais io",
    "Vanessa atalanta",
    "Papilio machaon"
]

df = pd.read_csv(input_csv)

filtered_df = df[df["scientific_name"].isin(target_species)]

filtered_df.to_csv(output_csv, index=False)

print("Готово! Осталось строк:", len(filtered_df))
