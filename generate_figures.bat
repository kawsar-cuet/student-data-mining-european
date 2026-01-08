@echo off
echo ================================================================================
echo Generating Journal Paper Figures with Subfigure Labels
echo ================================================================================
echo.

cd /d "d:\MS program\Final Thesis\Final Thesis project"

echo [1/3] Generating class distribution figure (a, b)...
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/01_dataset_analysis.py"
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate class distribution figure
    pause
    exit /b 1
)
echo.

echo [2/3] Generating confusion matrices, ROC curves, and cross-validation figures...
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/regenerate_figures_with_ahfs_ta.py"
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate comparison figures
    pause
    exit /b 1
)
echo.

echo [3/3] Generating comprehensive metrics comparison figure (a, b, c, d)...
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/generate_comprehensive_metrics_comparison.py"
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate comprehensive metrics figure
    pause
    exit /b 1
)
echo.

echo ================================================================================
echo SUCCESS! All figures generated with subfigure labels
echo ================================================================================
echo.
echo Figures saved to:
echo   - supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/
echo   - outputs/figures/
echo   - Journal Paper Writing/figures/
echo.
pause
