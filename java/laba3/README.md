# Лабораторная работа №3 — поиск документов с LSA и Word2Vec

LSA и Word2Vec обучаются на общем корпусе и выполняют поиск документов
по текстовому запросу. Оба метода входят в приложение `semantic-analyzer`.

[Корневой README](../../readme.md) содержит список работ.
Сборка, развёртывание, параметры команд и форматы пользовательских данных описаны
в [инструкции приложения](../semantic-analyzer/README.md).
Методы, условия экспериментов и результаты приведены в [отчёте](Laba3.md).

## Сборка и запуск

Подготовьте JDK 25 по [общей инструкции Java](../README.md).
Выполните команды из `java/` в Windows CMD:

```bat
gradlew.bat :semantic-analyzer:build :semantic-analyzer:javadoc
run-lab.cmd semantic-analyzer --help
```

В PowerShell используйте `./gradlew.bat` и `./run-lab.cmd`, в Linux/macOS —
`sh ./gradlew` и `sh ./run-lab.sh`. Остальные аргументы сохраните.
Для обучения укажите новый каталог `--output`.

## Примеры работы

### Word2Vec

```bat
run-lab.cmd semantic-analyzer train --algorithm word2vec --input semantic-analyzer/examples/documents.csv --output semantic-analyzer/artifacts/word2vec-demo --limit 3 --dimensions 8 --epochs 2 --min-count 1 --window 3 --workers 1
run-lab.cmd semantic-analyzer search --algorithm word2vec --model semantic-analyzer/artifacts/word2vec-demo --query "Космос ракеты спутники космос ракеты спутники" --top 3
```

В проверенном запуске получены 3 документа, 18 токенов и словарь из 9 слов.
Покрытие запроса — 6/6. Порядок выдачи: `space` (cosine 1,000000),
`music` (0,672102), `sport` (0,394221).
Запрос повторяет текст документа и проверяет сохранение и поиск после загрузки.

### LSA

```bat
run-lab.cmd semantic-analyzer train --algorithm lsa --input semantic-analyzer/examples/related-documents.csv --output semantic-analyzer/artifacts/lsa-demo --limit 6 --dimensions 3 --workers 1
run-lab.cmd semantic-analyzer search --algorithm lsa --model semantic-analyzer/artifacts/lsa-demo --query "Ракеты спутники" --top 3
```

В проверенном запуске получены 6 документов, 24 токена, 15 терминов и 3 компоненты.
Документов с нулевым вектором нет. Покрытие запроса — 2/2.
Первые результаты — `space-a` и `space-b`, cosine обоих равен 1,000000.
Сходство третьего результата имеет порядок численной погрешности.

Для собственного корпуса используйте CSV со столбцами `id,text` или папку TXT.
Замените `--input`, укажите новый `--output` и передайте свой текст в `--query`.
Полные требования приведены в [описании входных данных](../semantic-analyzer/README.md#входные-данные).

## Сравнение методов

LSA использует TF-IDF и усечённое SVD. Word2Vec использует обученные векторы слов
и их взвешенное усреднение для представления документа. Оба метода ищут документы
по косинусному сходству.

На общей коллекции из 10 000 статей РИА и 500 отложенных запросов получены:

| Показатель | LSA | Word2Vec |
| --- | ---: | ---: |
| Recall@5 | 0,304000 | 0,778000 |
| MRR@10 | 0,231521 | 0,649069 |
| Покрытие токенов запросов | 60,08% | 97,67% |
| Документов с нулевым вектором | 3 | 0 |

При выбранных параметрах Word2Vec чаще находил релевантную статью. LSA быстрее
обучился в проведённом эксперименте. Результаты зависят от корпуса, параметров
и ограничения словаря LSA; их нельзя переносить на другие данные без проверки.

Повторите [подготовку коллекции и обучение](../semantic-analyzer/README.md#подготовка-и-обучение-на-риа),
затем выполните [единую оценку](../semantic-analyzer/README.md#единая-оценка).
Параметры эксперимента и границы измерений приведены в [отчёте](Laba3.md#34-условия-и-границы-измерений).
