# Лабораторная работа №2: семантический анализатор LSA

## Установка

Нужен Python 3.12. Команды выполняются из корня репозитория и используют `python` активированного окружения. Установите общие зависимости обеих лабораторных работ, а для тестов — зависимости разработки:

```text
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Анализатор строит TF-IDF и LSA (`TruncatedSVD`), а для полностью размеченного корпуса — supervised LDA. LSA имеет не более 50 компонент; детерминированные операции используют seed 42. LDA-координаты не являются вероятностями.

## Команды

Обычный порядок работы: `prepare` приводит источник к каноническому CSV, `train` сохраняет модель, `analyze` ищет документы, `topics` объясняет компоненты LSA, `evaluate` сравнивает качество TF-IDF и LSA.

| Команда | Обязательные параметры | Дополнительные параметры и результат |
| --- | --- | --- |
| `prepare` | `--output CSV`; для своего источника — `--input FILE` | `--config TOML` выбирает конфигурацию; `--profile rureviews\|lenta` выбирает встроенный профиль. `--retained-source CSV` задаёт путь полного CSV Lenta.ru. Результат — корпус и сводка подготовки. |
| `train` | `--corpus CSV --model JOBLIB` | Сохраняет словарь, преобразования, документы, настройки и версии в пакет модели. |
| `analyze` | `--model JOBLIB` и ровно один из `--text "запрос"`, `--file UTF8_FILE` | `--top-k N` ограничивает выдачу; по умолчанию 5 документов. Загрузка модели не запускает обучение. |
| `topics` | `--model JOBLIB` | `--top-terms N` задаёт число терминов каждого знака; по умолчанию 5. Показывает положительные и отрицательные термы каждой LSA-компоненты с нагрузками. |
| `evaluate` | `--corpus CSV --output JSON` | Обучает отдельную модель только на `train` и сохраняет протокол оценки на `test`. Требует размеченный корпус с частями. |

Каждая команда принимает `--format text|json|pretty-json`; по умолчанию — `text`. `json` выводит один компактный объект, `pretty-json` — тот же объект с отступами. Справка доступна через `python -m Laba2 --help` и `python -m Laba2 <команда> --help`.

`analyze` возвращает метаданные, нормализованные токены, все ненулевые пары TF-IDF, полный LSA-вектор, LDA-вектор либо `null` и найденные документы. Для документа указаны `id`, метка при наличии, фрагмент, место, `tfidf_cosine` и `lsa_cosine`.

В `evaluate` каждый test-документ ищется только среди train-документов, а его метка применяется только к метрикам. Протокол содержит параметры, версии, seed, размеры частей, число признаков и компонент, пропуски, примеры, `Precision@5`, `Hit@5` для TF-IDF и LSA, а также silhouette LSA/LDA, когда он определён.

## Примеры встроенных профилей

### RuReviews

Профиль [`profiles/rureviews.toml`](profiles/rureviews.toml) использует `Laba1/data/rureviews.tsv`, если `--input` не указан. Он преобразует `neautral` в `neutral`, разрешает конфликты меток большинством и после разбиения оставляет по каждому классу 200 train и 50 test: всего 600/150 документов.

```text
python -m Laba2 prepare --profile rureviews --output Laba2/artifacts/rureviews.csv
python -m Laba2 train --corpus Laba2/artifacts/rureviews.csv --model Laba2/artifacts/rureviews.joblib
python -m Laba2 analyze --model Laba2/artifacts/rureviews.joblib --text "Хорошее качество товара" --top-k 5
python -m Laba2 topics --model Laba2/artifacts/rureviews.joblib --top-terms 5
python -m Laba2 evaluate --corpus Laba2/artifacts/rureviews.csv --output Laba2/artifacts/rureviews-evaluation.json --format pretty-json
```

В воспроизведённом протоколе: 600 train и 150 test-документов, 2 145 признаков, 50 LSA- и 2 LDA-компоненты; обработано 150 запросов без пропусков. TF-IDF: `Precision@5` 0.556000, `Hit@5` 0.913333; LSA — 0.528000 и 0.880000. Silhouette: LSA −0.007440, LDA 0.113338.

### Lenta.ru

Профиль [`profiles/lenta.toml`](profiles/lenta.toml) принимает локальный CSV либо `.bz2` из [Lenta.Ru-News-Dataset v1.1](https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/tag/v1.1) **(необходимо загрузить вручную)**. Адаптер сохраняет полный CSV, отбирает по 250 уникальных статей тем «Россия», «Мир», «Спорт», «Экономика» и создаёт стратифицированные 800/200 документов. `--retained-source` задаёт путь полного CSV. Архив `.bz2` удаляется после успешной записи поднабора.

```text
python -m Laba2 prepare --profile lenta --input lenta-ru-news.csv.bz2 --retained-source Laba2/data/lenta-ru-news.csv --output Laba2/artifacts/lenta-subset.csv
python -m Laba2 train --corpus Laba2/artifacts/lenta-subset.csv --model Laba2/artifacts/lenta.joblib
python -m Laba2 analyze --model Laba2/artifacts/lenta.joblib --text "Футбольная команда выиграла матч" --top-k 5
python -m Laba2 topics --model Laba2/artifacts/lenta.joblib --top-terms 5
python -m Laba2 evaluate --corpus Laba2/artifacts/lenta-subset.csv --output Laba2/artifacts/lenta-evaluation.json --format pretty-json
```

В воспроизведённом протоколе: 800 train и 200 test-документов, 14 562 признака, 50 LSA- и 3 LDA-компоненты; обработано 200 запросов без пропусков. TF-IDF: `Precision@5` 0.791000, `Hit@5` 0.985000; LSA: 0.827000 и 0.990000. Silhouette: LSA 0.049496, LDA 0.396176.

## Пример с генерацией своего корпуса

Первые две команды создают небольшой CSV и одноимённый TOML в `Laba2/artifacts/`. Следующие пять команд выполняют весь путь от подготовки до оценки; все созданные файлы остаются в этой папке.

```text
python -c "from pathlib import Path; p = Path('Laba2/artifacts'); p.mkdir(parents=True, exist_ok=True); (p / 'news.csv').write_text('title,body,kind\nЯблоко,растет в саду,fruit\nГруша,сладкая в саду,fruit\nПерсик,спелый фрукт,fruit\nФутбол,команда забила гол,sport\nМатч,игрок выиграл матч,sport\nХоккей,команда забила шайбу,sport\n', encoding='utf-8')"
python -c 'from pathlib import Path; Path("Laba2/artifacts/news.toml").write_text("[dataset]\nformat = \"csv\"\n[columns]\ntext = [\"title\", \"body\"]\nlabel = \"kind\"\n[split]\nenabled = true\ntrain = 67\nvalidation = 0\ntest = 33\nrandom_state = 42\n[expect]\nsplits = { train = 67, test = 33 }\n", encoding="utf-8")'
python -m Laba2 prepare --input Laba2/artifacts/news.csv --output Laba2/artifacts/corpus.csv --format pretty-json
python -m Laba2 train --corpus Laba2/artifacts/corpus.csv --model Laba2/artifacts/model.joblib --format json
python -m Laba2 analyze --model Laba2/artifacts/model.joblib --text "Команда забила гол" --top-k 2 --format pretty-json
python -m Laba2 topics --model Laba2/artifacts/model.joblib --top-terms 2 --format text
python -m Laba2 evaluate --corpus Laba2/artifacts/corpus.csv --output Laba2/artifacts/evaluation.json --format json
```

`prepare` сначала выбирает явный `--config FILE`, иначе ищет TOML с тем же именем рядом с входным файлом: для `news.csv` — `news.toml`. `--profile` и `--config` взаимоисключающие.

Канонический CSV сохраняется атомарно в UTF-8 с заголовком `id,text`, `id,text,label` либо `id,text,label,split`. `id` и `text` непусты и уникальны. Если исходный столбец ID не указан, идентификатор имеет вид `sha256:<digest текста>`. Метка, если есть, заполнена у всех строк; допустимые части — `train`, `validation`, `test`.

Неразмеченный CSV использует только `id,text`: его можно обучать и анализировать, но `lda` и `lda_components` равны `null`; оценка требует размеченного корпуса с частями.

## Конфигурация TOML

Схема строгая: неизвестные таблицы и ключи отклоняются. Обязательны только `[dataset]` с форматом и `[columns]` с текстовыми столбцами. Ниже перечислены все настройки; необязательные блоки можно удалить. Размеры в `[sampling]` и `expect.rows` задаются в строках, а `[split]` и `expect.splits` — в процентах от 0 до 100.

```toml
[dataset]
format = "csv"                  # Формат исходной таблицы: csv или tsv; обязателен.
encoding = "utf-8"              # Кодировка источника; по умолчанию utf-8.
delimiter = ","                 # Разделитель из одного символа; по умолчанию , или \t для tsv.

[columns]
id = "id"                       # Столбец уникального ID; без настройки ID строится из текста.
text = ["title", "body"]         # Обязательные текстовые поля в порядке объединения; можно строку.
label = "label"                 # Столбец меток; без него корпус считается неразмеченным.
split = "split"                 # Готовые train/validation/test; убрать при split.enabled = true.
joiner = " "                    # Строка между текстовыми полями; по умолчанию пробел.

[cleaning]
trim = true                     # Удалять пробелы по краям каждого поля; по умолчанию true.
empty = "error"                 # error: ошибка при пустом обязательном поле; drop: удалить строку.
labels = { old = "new" }         # Заменить метку old на new; прочие метки сохраняются.
duplicates = "error"            # Одинаковый текст и метка: error, drop всех копий или keep одной.
conflicts = "error"             # Разные метки одного текста: error или majority (большинство).
tie = "error"                   # Ничья при majority: error или drop всей группы.

[split]
enabled = false                 # true: создать стратифицированные части; требуется columns.label.
train = 80                      # Процент train; по умолчанию 80.
validation = 0                  # Процент validation; по умолчанию 0.
test = 20                       # Процент test; по умолчанию 20. Сумма трёх процентов равна 100.
random_state = 42               # Неотрицательный seed разбиения; по умолчанию 42.

[sampling]
per_class = 100                 # Лимит строк каждого класса в каждой части; требуется label.
per_split = { train = 80, test = 20 } # Лимиты на класс внутри частей; пропущенные части удаляются.

[expect]
rows = 100                      # Ожидаемое общее число строк после всех преобразований.
classes = 2                     # Ожидаемое число разных меток итогового корпуса.
splits = { train = 80, test = 20 } # Ожидаемые проценты частей; сумма 100, пропущенные = 0.

[adapter]
path = "package.module:prepare"  # Функция нестандартного чтения вместо табличных политик.
parameters = { option = "value" } # Именованные параметры, передаваемые этой функции.
```

В `[cleaning]` политики `empty`, `duplicates`, `conflicts`, `tie` по умолчанию равны `error`, а отображение `labels` пустое. `duplicates = "keep"` оставляет одну копию с минимальным ID. `majority` также оставляет одну строку с победившей меткой и минимальным ID. Пустое настроенное поле ID считается ошибкой или удаляется по политике `empty`.

`[split]` создаёт части по каждому классу отдельно: строки детерминированно перемешиваются, размеры округляются вниз, оставшиеся строки получают части с наибольшим дробным остатком. Проценты могут быть дробными, например `train = 67.5`; бесконечности, отрицательные значения и сумма, отличная от 100, отклоняются. Входной столбец `split` нельзя сочетать с `enabled = true`.

`[sampling]` ограничивает уже созданные части: `per_split` может быть таблицей лимитов либо одним целым числом не меньше 1 для всех частей. При одновременном `per_class` применяется меньший лимит. Без разбиения `per_class` ограничивает весь класс. Выборка детерминирована; если строк меньше лимита, сохраняются доступные. Без этих настроек выборка не выполняется.

`[expect]` только проверяет результат. Для `splits` ожидаемые размеры рассчитываются из общего итогового числа строк методом наибольших остатков и сравниваются с фактическими размерами всех трёх частей. Например, `{ train = 67, test = 33 }` для 6 строк означает 4 train и 2 test. Проверка не требует `rows`; при отсутствии `splits` размеры частей не проверяются. В полной схеме блок `[adapter]` — альтернативный способ подготовки: для обычного CSV/TSV удалите его.

Порядок табличной подготовки: очистка и объединение полей, проверка обязательных значений и генерация ID, разрешение дубликатов и конфликтов, стратифицированное разбиение, выборка, валидация канонического корпуса, проверка `[expect]`, атомарная запись.

## Внешний адаптер

Для нестандартного источника задайте `[adapter].path` в форме `module:object`. Пример конфигурации для [`external_adapter_template.py`](adapters/external_adapter_template.py):

```toml
[dataset]
format = "csv"  # Обязательное поле схемы; способ чтения источника определяет адаптер.
[columns]
text = "text"   # Обязательное поле схемы; адаптер сам формирует CorpusRow.
[adapter]
path = "Laba2.adapters.external_adapter_template:prepare" # Модуль и вызываемая функция.
parameters = {} # Параметры примера пусты; свой адаптер может принимать имена и значения.
```

Модуль должен быть доступен для импорта из корня репозитория. `prepare(source, parameters)` получает `Path` источника и неизменяемое отображение параметров. Функция читает нестандартный формат и возвращает `AdapterResult(rows, metadata)`:

| Элемент | Назначение |
| --- | --- |
| `source` | Путь из `--input`; адаптер сам открывает и проверяет источник. |
| `parameters` | Значения `[adapter].parameters`; изменять отображение внутри адаптера нельзя. |
| `CorpusRow(identifier, text, label, split)` | Уникальный непустой ID и текст; метка и часть необязательны. Если часть задана, нужна метка. |
| `AdapterResult.rows` | Последовательность канонических строк, одинаково размеченных и оформленных. |
| `AdapterResult.metadata` | Дополнительные сведения для сводки подготовки; можно `{}`. |

Шаблон не читает источник: он возвращает одну демонстрационную строку `CorpusRow("example-1", "Пример текста", "example", "train")`. Для своего формата замените тело функции и укажите её импортируемый путь в TOML. Ошибки входных данных можно сообщать через `PreparationError` из `LabsCommon.dataset_adapter`.

При подключении адаптера табличные настройки очистки, разбиения и выборки не выполняются: эти действия реализует сам адаптер. Возвращённые строки проходят общий валидатор, проверки `[expect]` и атомарную запись CSV.

## Локальные файлы и ошибки

Полные входные корпуса хранятся в `Laba2/data/`, встроенные конфигурации — в `Laba2/profiles/`, файлы демонстрационных примеров, подготовленные поднаборы, модели `.joblib` и JSON-протоколы — в `Laba2/artifacts/`.

Пустой/OOV-запрос, неверный CSV/TOML, недоступный путь или повреждённая модель завершаются сообщением в stderr. Ошибка подготовки или записи протокола не оставляет частичный CSV либо JSON.
