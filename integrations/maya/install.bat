@echo off
rem Install GEM-X Live for Maya from this extracted folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
