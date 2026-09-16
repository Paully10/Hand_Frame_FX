@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python hand_frame_fx.py
