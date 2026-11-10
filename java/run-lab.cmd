@echo off
setlocal DisableDelayedExpansion
"%SystemRoot%\System32\chcp.com" 65001 >nul 2>nul
if "%~1"=="" goto usage
set "LAB_MODULE=%~1"
set LAB_ARGS=%*
if [%2]==[] (
    set "LAB_ARGS="
) else (
    goto arguments
)
goto findJava

:arguments
set LAB_ARGS=%LAB_ARGS:* =%

:findJava
for %%J in (java.exe) do set "LAB_JAVA=%%~$PATH:J"
if not defined LAB_JAVA (
    >&2 echo Error: java.exe was not found in PATH. Put JDK 25 bin first in PATH.
    exit /b 2
)
set "LAB_VERSION="
for /f "tokens=2" %%V in ('""%LAB_JAVA%" --version"') do if not defined LAB_VERSION set "LAB_VERSION=%%V"
set "LAB_MAJOR="
for /f "tokens=1 delims=.-+" %%V in ("%LAB_VERSION%") do set "LAB_MAJOR=%%V"
if not "%LAB_MAJOR%"=="25" (
    >&2 echo Error: Java 25 must be first in PATH. Found: %LAB_VERSION%
    >&2 echo Java executable: "%LAB_JAVA%"
    exit /b 2
)
set "LAB_JAR="
for %%J in ("%~dp0%LAB_MODULE%\build\libs\%LAB_MODULE%-*.jar") do if exist "%%~fJ" (
    if defined LAB_JAR goto ambiguousJar
    set "LAB_JAR=%%~fJ"
)
if not defined LAB_JAR (
    >&2 echo Error: no JAR found for "%LAB_MODULE%". Build the module first.
    exit /b 2
)
"%LAB_JAVA%" -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dstderr.encoding=UTF-8 --enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow -jar "%LAB_JAR%" %LAB_ARGS%
exit /b %errorlevel%

:ambiguousJar
>&2 echo Error: multiple JAR files found for "%LAB_MODULE%". Clean and rebuild the module.
exit /b 2

:usage
>&2 echo Usage: run-lab.cmd MODULE [arguments]
>&2 echo Example: run-lab.cmd laba3 --help
exit /b 2
