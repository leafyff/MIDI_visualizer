@echo off
REM ====================================================================
REM  MIDI Visualizer - double-click this file to start the program.
REM
REM  It finds a Python to run, installs the required libraries the first
REM  time, checks that ffmpeg is available, and then opens the window.
REM
REM  Any arguments are passed straight through, so you can also run:
REM      run.bat song.mid                 open the app with a file loaded
REM      run.bat song.mid -o out.mp4      render without the interface
REM ====================================================================

setlocal

REM Work from the folder this script lives in, no matter where it was
REM started from. "%~dp0" is that folder; /d allows changing drive too.
cd /d "%~dp0"


REM --- 1. Find a Python to use ----------------------------------------
REM The project's own virtual environment is preferred, because that is
REM where the libraries get installed.

set "PYTHON="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :found_python
)

REM The Windows Python launcher, installed with Python itself.
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
    goto :found_python
)

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :found_python
)

echo.
echo   Python was not found on this computer.
echo.
echo   Install Python 3.10 or newer from https://www.python.org/downloads/
echo   and tick "Add Python to PATH" during the installation.
echo.
pause
exit /b 1

:found_python


REM --- 2. Install the libraries, the first time only -------------------
REM Importing them is the quickest way to ask "are they already there?".

%PYTHON% -c "import PyQt6, mido, numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Installing the required libraries. This happens only once
    echo   and may take a minute...
    echo.
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
    echo.
)


REM --- 3. Check for ffmpeg ---------------------------------------------
REM Only needed to save a video, so this is a warning rather than an error.

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo   NOTE: ffmpeg was not found.
    echo   The app will still open, but saving a video needs it:
    echo       winget install Gyan.FFmpeg
    echo.
)


REM --- 4. Start the program --------------------------------------------
REM %* passes on anything typed after run.bat.

%PYTHON% main.py %*
if errorlevel 1 goto :failed

exit /b 0


:failed
echo.
echo   Something went wrong. The message above should explain what.
echo.
pause
exit /b 1
