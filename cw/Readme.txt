Классы:
"Apoda limacodes",
"Hyphantria cunea",
"Orosanga japonicus",
"Heterogenea asella",
"Aglais io",
"Vanessa atalanta",
"Papilio machaon"

Работа с dvc:
	Инициализация:
		git init
		dvc init
		git add .dvc .gitignore
		git commit -m "Init DVC"
	Добавление данных:
		dvc add dataset
		git add dataset.dvc .gitignore
		git commit -m "Add dataset to DVC"
	Создание пайплайна (одна команда):
		dvc run -n train \
    			-d src/train.py -d src/model.py -d dataset -d params.yaml \
    			-o models/butterfly_model.h5 -o models/classes.json \
    			-M metrics.json \
    			python src/train.py
	Запуск пайплайна:
		dvc repro
	Эксперименты:
		dvc exp run - запуск с текущими параметрами
		dvc exp show - посмотреть все эксперименты
		dvc exp show --sort-by val_accuracy - сортировка по метрикам



