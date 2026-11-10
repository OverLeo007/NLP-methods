# Практические работы на Java

Этот каталог — корень Java-проекта. Все команды из инструкций работ выполняются
из него (из корня репозитория: `cd java`). Требуется **JDK 25**;
каталог `bin` JDK должен быть первым в `PATH`. Проверьте `java --version`.
После изменения `PATH` откройте новый терминал.

Gradle устанавливать отдельно не нужно: Wrapper входит в проект. При первой сборке
нужен доступ к интернету для загрузки Gradle и зависимостей.

## Сборка и проверки

Windows CMD:

```bat
gradlew.bat build spotlessCheck
```

В PowerShell используйте `./gradlew.bat`, в Linux/macOS — `sh ./gradlew`.
Зависимости и версии задаются общим каталогом `gradle/libs.versions.toml`.

## Запуск

Каждая работа — отдельное CLI-приложение. Сначала соберите нужный модуль,
затем передайте раннеру его имя и аргументы приложения:

```bat
run-lab.cmd laba3 --help
```

В PowerShell используйте `./run-lab.cmd`, в Linux/macOS — `sh ./run-lab.sh`.

JAR ищется относительно скрипта в `<модуль>/build/libs/`; там должен быть ровно
один исполняемый JAR. При нескольких версиях выполните `clean` и пересоберите модуль.
Пути данных и результатов отсчитываются от текущей папки терминала.

Для прямого запуска вместо раннера (в Windows консоль должна использовать UTF-8):

```sh
java "-Dfile.encoding=UTF-8" "-Dstdout.encoding=UTF-8" "-Dstderr.encoding=UTF-8" --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -jar laba3/build/libs/laba3-0.1.0.jar --help
```

Имя JAR и аргументы заменяются на нужную работу. Параметры JVM, например
`-Xmx4g`, указываются перед `-jar`.

## Работы

- №3: [инструкция](laba3/README.md), [отчёт](laba3/Laba3.md).
