@ECHO off

set /P installer=Enter name of package manager:

if "%installer%" == "conda" (
conda install --file py_packages.txt
) else (
pip install -r py_packages.txt
)

ECHO "Finished!"
PAUSE 