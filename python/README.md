# Практические работы на Python

Этот каталог является корнем Python-проекта. Из корня репозитория перейдите в него:

```text
cd python
```

Команды из инструкций лабораторных работ выполняются из этого каталога. Нужен Python
3.12 и активированное окружение с зависимостями из `requirements.txt`:

```text
python -m pip install -r requirements.txt
```

Для тестов, проверки стиля и форматирования:

```text
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Работы

- №1: [инструкция](Laba1/README.md), [отчёт](Laba1/Laba1.md).
- №2: [инструкция](Laba2/README.md), [отчёт](Laba2/Laba2.md).

Пакет `LabsCommon` содержит общую обработку текстов и корпусов. Имя пакета и команды
`python -m Laba1` и `python -m Laba2` сохранены. Относительные пути к данным и результатам
в инструкциях указаны от корня Python-проекта, то есть от `python/`.
