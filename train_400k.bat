@echo off
echo ============================================
echo   HUMANOID WALKING - SAC 400K TIMESTEPS
echo ============================================
echo.
echo Starting training with RTX 4070...
echo This will take approx 2-3 hours
echo.
cd /d C:\Users\omarb\Desktop\comp_media\AC\src
python train_sac.py
echo.
echo Training done! Check runs\humanoid_walk.mp4
pause