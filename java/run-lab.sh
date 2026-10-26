#!/bin/sh
# Запуск: sh ./run-lab.sh laba3 <аргументы лабораторной>
if [ "$#" -eq 0 ]; then
    echo 'Использование: sh ./run-lab.sh <модуль> [аргументы]. Требуется Java 25 первой в PATH.' >&2
    exit 2
fi
lab=$1
shift
case "$lab" in
    ''|*[!A-Za-z0-9_-]*) echo 'Некорректное имя модуля.' >&2; exit 2 ;;
esac
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
java_version=$(command java --version) || {
    echo 'Для запуска требуется Java 25 первой в PATH.' >&2
    exit 2
}
java_major=${java_version#* }
java_major=${java_major%%[!0-9]*}
if [ "$java_major" != '25' ]; then
    echo 'Для запуска требуется Java 25 первой в PATH.' >&2
    exit 2
fi
jar_file=
for candidate in "$script_dir/$lab/build/libs/$lab-"*.jar; do
    [ -f "$candidate" ] || continue
    case "$candidate" in *-plain.jar) continue ;; esac
    if [ -n "$jar_file" ]; then
        echo 'Найдено несколько JAR. Удалите устаревшие файлы через clean и соберите модуль.' >&2
        exit 2
    fi
    jar_file=$candidate
done
if [ -z "$jar_file" ]; then
    echo "Не найден JAR для $lab. Сначала выполните сборку модуля." >&2
    exit 2
fi
exec java -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -jar "$jar_file" "$@"
