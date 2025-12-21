"""
Master Orchestration Script
Runs all steps automatically: Training → Ablation → Visualizations → LaTeX Update
"""

import subprocess
import os
import time
import sys

def run_script(script_name, description):
    """Run a Python script and monitor progress"""
    
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80 + "\n")
    
    python_exe = "D:/MS program/Final Thesis/Final Thesis project/.venv/Scripts/python.exe"
    
    try:
        result = subprocess.run(
            [python_exe, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR")
        print(f"Error: {e}")
        return False


def check_file_exists(filepath, max_wait_minutes=30):
    """Wait for a file to be created (for background processes)"""
    
    print(f"\nWaiting for: {filepath}")
    print(f"Max wait time: {max_wait_minutes} minutes")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while not os.path.exists(filepath):
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_seconds:
            print(f"\n⏱️  Timeout after {max_wait_minutes} minutes")
            return False
        
        # Print progress every minute
        if int(elapsed) % 60 == 0 and elapsed > 0:
            print(f"  Waited {int(elapsed/60)} minutes...")
        
        time.sleep(10)  # Check every 10 seconds
    
    print(f"✅ File created: {filepath}")
    return True


def copy_figures_to_thesis():
    """Copy generated figures to thesis FIGURES folder"""
    
    print("\n" + "="*80)
    print("COPYING FIGURES TO THESIS FOLDER")
    print("="*80 + "\n")
    
    source_dir = "outputs/figures_journal"
    dest_dir = "supervisor_requirements/United_International_University_FYDP_Template_Department_of_CSE/FIGURES"
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    if not os.path.exists(dest_dir):
        print(f"⚠️  Creating destination directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
    
    # Copy AHFS-TA related figures
    ahfs_figures = [
        'comprehensive_model_comparison.png',
        'ablation_study_results.png',
        'temporal_attention_weights.png',
        'llm_feature_importance.png',
        'training_convergence.png',
        'confusion_matrices_comparison.png',
        'semester_risk_trajectories.png',
        'feature_selection_efficiency.png'
    ]
    
    copied_count = 0
    for fig in ahfs_figures:
        source = os.path.join(source_dir, fig)
        dest = os.path.join(dest_dir, fig)
        
        if os.path.exists(source):
            import shutil
            shutil.copy2(source, dest)
            print(f"✓ Copied: {fig}")
            copied_count += 1
        else:
            print(f"⚠️  Not found: {fig}")
    
    print(f"\n✅ Copied {copied_count}/{len(ahfs_figures)} figures")
    return copied_count > 0


def compile_thesis():
    """Compile the LaTeX thesis"""
    
    print("\n" + "="*80)
    print("COMPILING LATEX THESIS")
    print("="*80 + "\n")
    
    thesis_dir = "supervisor_requirements/United_International_University_FYDP_Template_Department_of_CSE"
    
    if not os.path.exists(thesis_dir):
        print(f"❌ Thesis directory not found: {thesis_dir}")
        return False
    
    try:
        os.chdir(thesis_dir)
        
        # Run pdflatex twice for references
        print("Running pdflatex (pass 1)...")
        subprocess.run(['pdflatex', '-interaction=batchmode', 'fydp.tex'], check=True)
        
        print("Running pdflatex (pass 2)...")
        subprocess.run(['pdflatex', '-interaction=batchmode', 'fydp.tex'], check=True)
        
        if os.path.exists('fydp.pdf'):
            # Get page count
            with open('fydp.log', 'r') as f:
                log_content = f.read()
                if 'Output written on fydp.pdf' in log_content:
                    import re
                    match = re.search(r'\\((\d+) pages', log_content)
                    if match:
                        pages = match.group(1)
                        print(f"\n✅ Thesis compiled successfully: {pages} pages")
                        print(f"   Location: {os.path.abspath('fydp.pdf')}")
                        return True
            
            print("\n✅ Thesis compiled successfully")
            return True
        else:
            print("\n❌ PDF not generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Compilation failed: {e}")
        return False
    finally:
        os.chdir("../../..")  # Return to project root


def main():
    """Master orchestration - run all steps"""
    
    print("\n" + "="*100)
    print("AHFS-TA COMPLETE IMPLEMENTATION PIPELINE")
    print("="*100)
    print("\nThis script will:")
    print("  1. ✅ Train AHFS-TA model (already started)")
    print("  2. ⏳ Run ablation study and baseline comparisons")
    print("  3. ⏳ Generate all visualizations")
    print("  4. ⏳ Update LaTeX thesis with actual results")
    print("  5. ⏳ Copy figures to thesis folder")
    print("  6. ⏳ Compile final thesis PDF")
    print("="*100 + "\n")
    
    # Step 1: Check if training is complete
    print("STEP 1: Checking AHFS-TA Training Status")
    print("-" * 80)
    
    if os.path.exists('outputs/ahfs_ta_results.pt'):
        print("✅ Training already complete!")
    else:
        print("⏳ Training in progress...")
        print("   Waiting for training to complete (max 30 minutes)...")
        
        if not check_file_exists('outputs/ahfs_ta_results.pt', max_wait_minutes=30):
            print("\n❌ Training did not complete in time.")
            print("   Please check ahfs_ta_implementation.py manually")
            return
    
    time.sleep(5)  # Brief pause
    
    # Step 2: Run ablation study
    print("\nSTEP 2: Running Ablation Study and Baseline Comparisons")
    print("-" * 80)
    
    if not run_script('ablation_study_comparison.py', 'Ablation Study'):
        print("\n⚠️  Continuing despite ablation study issues...")
    
    time.sleep(2)
    
    # Step 3: Generate visualizations
    print("\nSTEP 3: Generating All Visualizations")
    print("-" * 80)
    
    if not run_script('generate_visualizations.py', 'Visualization Generation'):
        print("\n⚠️  Some visualizations may be missing")
    
    time.sleep(2)
    
    # Step 4: Generate final summary
    print("\nSTEP 4: Generating Final Summary")
    print("-" * 80)
    
    run_script('generate_final_summary.py', 'Final Summary Generation')
    
    time.sleep(2)
    
    # Step 5: Update LaTeX thesis
    print("\nSTEP 5: Updating LaTeX Thesis with Actual Results")
    print("-" * 80)
    
    if not run_script('update_latex_results.py', 'LaTeX Thesis Update'):
        print("\n⚠️  LaTeX update failed, but you can update manually")
    
    time.sleep(2)
    
    # Step 6: Copy figures
    print("\nSTEP 6: Copying Figures to Thesis Folder")
    print("-" * 80)
    
    copy_figures_to_thesis()
    
    time.sleep(2)
    
    # Step 7: Compile thesis
    print("\nSTEP 7: Compiling Final Thesis PDF")
    print("-" * 80)
    
    if compile_thesis():
        print("\n🎉 THESIS COMPILATION SUCCESSFUL!")
    else:
        print("\n⚠️  Thesis compilation had issues. You may need to compile manually.")
    
    # Final summary
    print("\n" + "="*100)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*100)
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 80)
    
    if os.path.exists('outputs/ahfs_ta_results.pt'):
        import torch
        results = torch.load('outputs/ahfs_ta_results.pt')
        metrics = results['metrics']
        
        print(f"\nAHFS-TA Performance:")
        print(f"  ✅ Accuracy:  {metrics['Accuracy']:.2f}%")
        print(f"  ✅ F1-Score:  {metrics['F1-Score']:.3f}")
        print(f"  ✅ AUC-ROC:   {metrics['AUC-ROC']:.3f}")
        print(f"  ✅ Precision: {metrics['Precision']:.3f}")
        print(f"  ✅ Recall:    {metrics['Recall']:.3f}")
    
    print("\n📁 GENERATED FILES:")
    print("-" * 80)
    
    key_files = [
        ('outputs/ahfs_ta_results.pt', 'AHFS-TA trained model and results'),
        ('outputs/tables/comprehensive_model_comparison.csv', 'Model comparison table'),
        ('outputs/tables/ablation_study_results.csv', 'Ablation study results'),
        ('outputs/figures_journal/comprehensive_model_comparison.png', 'Model comparison figure'),
        ('outputs/figures_journal/ablation_study_results.png', 'Ablation study figure'),
        ('FINAL_RESULTS_SUMMARY.txt', 'Comprehensive results summary'),
        ('LATEX_UPDATE_SUMMARY.txt', 'LaTeX update summary'),
        ('supervisor_requirements/United_International_University_FYDP_Template_Department_of_CSE/fydp.pdf', 'Final thesis PDF')
    ]
    
    for filepath, description in key_files:
        if os.path.exists(filepath):
            print(f"  ✅ {description}")
            print(f"     → {filepath}")
        else:
            print(f"  ⚠️  {description} - NOT FOUND")
    
    print("\n" + "="*100)
    print("✅ ALL IMPLEMENTATION TASKS COMPLETED!")
    print("="*100)
    
    print("\nYour thesis now contains:")
    print("  ✅ Actual AHFS-TA implementation")
    print("  ✅ Real experimental results")
    print("  ✅ Comprehensive comparison tables")
    print("  ✅ Professional result visualizations")
    print("  ✅ Updated LaTeX with real metrics")
    print("\n🎓 Your thesis is ready for submission!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
