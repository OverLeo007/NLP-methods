# Лабораторная работа №1

Все команды выполняются из корня репозитория. Использовался Python 3.12.10; создайте и активируйте
подходящее виртуальное окружение, затем установите точные версии прямых зависимостей:

```text
python -m pip install -r Laba1/requirements.txt
```

Для запуска тестов и верификации можно дополнительно установить зависимости разработки:

```text
python -m pip install -r requirements-dev.txt
```

## Корпус, обучение и проверка

```text
python -m Laba1 train --artifacts Laba1/artifacts
python -m Laba1 train --artifacts Laba1/artifacts --verbose
python -m pytest
```

Полный исходный корпус RuReviews находится в `Laba1/data/rureviews.tsv`. 
Это UTF-8 TSV с точными столбцами `review` и `sentiment`; допустимы
метки `negative`, `neutral`, `positive` и исходное написание `neautral`. Команда `train`
использует только этот корпус и не требует сети. Обучается одна конфигурация:
стемминг с удалением стоп-слов, без сведения синонимов. Флаг `--verbose` показывает в
stderr чтение и очистку, размеры частей, обучение, test-метрики и пути результатов;
stdout при этом по-прежнему содержит один JSON.

После обучения создаются четыре результата:

- `Laba1/artifacts/model.joblib` — пакет для последующих запусков `analyze`;
- `Laba1/artifacts/metrics.json` — test-метрики и матрица ошибок;
- `Laba1/artifacts/predictions.json` — метки всех примеров test-части;
- `Laba1/images/confusion_matrix.png` — визуализация матрицы ошибок для отчёта.

## Наблюдаемая нормализация

```text
python -m Laba1 vocabulary --text "Очень хорошее платье" --method lemma --stopwords --synonyms
```

Команда выводит JSON с исходными и нормализованными токенами, их частотами и активными
этапами. Воспроизведённый результат:

```json
{"active_steps": ["unicode_nfc", "lowercase", "yo_to_e", "method:lemma", "stopwords", "synonyms"], "frequencies": {"очень": 1, "платье": 1, "хороший": 1}, "normalized_tokens": ["очень", "хороший", "платье"], "original_tokens": ["Очень", "хорошее", "платье"]}
```

## Анализ текста и файла

Анализатор принимает ровно один источник: строку или UTF-8-файл. Команда создания
демонстрационного файла использует тот же интерпретатор и работает на Windows, macOS и Linux.

```text
python -m Laba1 analyze --model Laba1/artifacts/model.joblib --text "Платье отличное"
python -c "from pathlib import Path; Path('Laba1/data/review.txt').write_text('Платье отличное', encoding='utf-8')"
python -m Laba1 analyze --model Laba1/artifacts/model.joblib --file Laba1/data/review.txt
```

Оба варианта возвращают один JSON. Для модели, полученной указанной командой обучения,
воспроизведён следующий результат:

```json
{"configuration": {"method": "stem", "stopwords": true, "synonyms": false}, "label": "positive", "probabilities": {"negative": 0.024413560169950495, "neutral": 0.060743840414319114, "positive": 0.914842599415731}}
```

Вероятности классов `negative`, `neutral`, `positive` суммируются к 1 с допуском `1e-6`;
метка соответствует максимальной вероятности. Пустой ввод, неверная кодировка файла,
недоступный путь и повреждённая модель завершаются с ненулевым кодом и сообщением в stderr.
