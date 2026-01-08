"""
Run all figure generation scripts
"""
import subprocess
import sys
from pathlib import Path

# Change to project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

scripts = [
    "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/01_dataset_analysis.py",
    "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/regenerate_figures_with_ahfs_ta.py",
    "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/generate_comprehensive_metrics_comparison.py"
]

for script in scripts:
    print(f"\n{'='*80}")
    print(f"Running: {Path(script).name}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(
        [sys.executable, script],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ Script failed with return code {result.returncode}")
    else:
        print(f"\n✅ Script completed successfully")

print(f"\n{'='*80}")
print("All scripts completed!")
print(f"{'='*80}")
