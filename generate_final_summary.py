"""
Final Results Summary Generator
Compiles all results into thesis-ready format
"""

import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime

def generate_latex_tables():
    """Generate LaTeX tables for thesis"""
    
    print("\n" + "="*80)
    print("GENERATING LATEX TABLES FOR THESIS")
    print("="*80 + "\n")
    
    os.makedirs('outputs/latex_tables', exist_ok=True)
    
    # Table 1: Comprehensive Model Comparison
    if os.path.exists('outputs/tables/comprehensive_model_comparison.csv'):
        df = pd.read_csv('outputs/tables/comprehensive_model_comparison.csv')
        
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\caption{Comprehensive Model Performance Comparison}\n"
        latex_table += "\\label{tab:comprehensive_comparison}\n"
        latex_table += "\\begin{tabular}{lccccc}\n"
        latex_table += "\\toprule\n"
        latex_table += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{AUC-ROC} & \\textbf{Features} & \\textbf{Temporal} \\\\\n"
        latex_table += "\\midrule\n"
        
        for _, row in df.iterrows():
            model = row['Model']
            if 'AHFS-TA' in model:
                latex_table += f"\\textbf{{{model}}} & \\textbf{{{row['Accuracy']:.2f}\\%}} & \\textbf{{{row['F1-Score']:.3f}}} & \\textbf{{{row['AUC-ROC']:.3f}}} & \\textbf{{{int(row['Features'])}}} & \\textbf{{{row['Temporal']}}} \\\\\n"
            else:
                latex_table += f"{model} & {row['Accuracy']:.2f}\\% & {row['F1-Score']:.3f} & {row['AUC-ROC']:.3f} & {int(row['Features'])} & {row['Temporal']} \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        with open('outputs/latex_tables/comprehensive_comparison.tex', 'w') as f:
            f.write(latex_table)
        print("✓ LaTeX table saved: comprehensive_comparison.tex")
    
    # Table 2: Ablation Study
    if os.path.exists('outputs/tables/ablation_study_results.csv'):
        df = pd.read_csv('outputs/tables/ablation_study_results.csv')
        
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\caption{AHFS-TA Ablation Study Results}\n"
        latex_table += "\\label{tab:ablation_study}\n"
        latex_table += "\\begin{tabular}{lcccc}\n"
        latex_table += "\\toprule\n"
        latex_table += "\\textbf{Configuration} & \\textbf{Accuracy} & \\textbf{AUC-ROC} & \\textbf{$\\Delta$ Accuracy} & \\textbf{Features} \\\\\n"
        latex_table += "\\midrule\n"
        
        for _, row in df.iterrows():
            config = row['Configuration']
            if config == 'Total Improvement':
                latex_table += "\\midrule\n"
                latex_table += f"\\textbf{{{config}}} & \\textbf{{+{row['Accuracy']:.2f}\\%}} & \\textbf{{+{row['AUC-ROC']:.3f}}} & -- & -- \\\\\n"
            else:
                delta = row['Δ Accuracy']
                delta_str = f"+{delta:.2f}\\%" if delta > 0 else "--"
                features = str(int(row['Features'])) if pd.notna(row['Features']) and row['Features'] != '--' else '--'
                latex_table += f"{config} & {row['Accuracy']:.2f}\\% & {row['AUC-ROC']:.3f} & {delta_str} & {features} \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        with open('outputs/latex_tables/ablation_study.tex', 'w') as f:
            f.write(latex_table)
        print("✓ LaTeX table saved: ablation_study.tex")


def generate_results_summary():
    """Generate comprehensive results summary"""
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'implementation_status': {},
        'key_results': {},
        'files_generated': []
    }
    
    # Check implementation status
    if os.path.exists('outputs/ahfs_ta_results.pt'):
        try:
            results = torch.load('outputs/ahfs_ta_results.pt')
            metrics = results['metrics']
            
            summary['implementation_status'] = {
                'Python Implementation': '✅ Complete',
                'Model Training': '✅ Complete',
                'Experimental Results': '✅ Complete'
            }
            
            summary['key_results'] = {
                'Accuracy': f"{metrics['Accuracy']:.2f}%",
                'F1-Score': f"{metrics['F1-Score']:.3f}",
                'AUC-ROC': f"{metrics['AUC-ROC']:.3f}",
                'Precision': f"{metrics['Precision']:.3f}",
                'Recall': f"{metrics['Recall']:.3f}",
                'MCC': f"{metrics['MCC']:.3f}"
            }
            
            print("🎯 AHFS-TA Performance Metrics:")
            print("-" * 40)
            for metric, value in summary['key_results'].items():
                print(f"  {metric:.<25} {value}")
            
        except Exception as e:
            print(f"⚠️  Could not load results: {e}")
            summary['implementation_status'] = {
                'Python Implementation': '✅ Complete',
                'Model Training': '⏳ In Progress',
                'Experimental Results': '⏳ Pending'
            }
    else:
        print("⏳ Model training still in progress...")
        summary['implementation_status'] = {
            'Python Implementation': '✅ Complete',
            'Model Training': '⏳ In Progress',
            'Experimental Results': '⏳ Pending'
        }
    
    # Check generated files
    print("\n" + "-" * 80)
    print("📁 Generated Files:")
    print("-" * 80)
    
    file_categories = {
        'Implementation': ['ahfs_ta_implementation.py', 'ablation_study_comparison.py', 'generate_visualizations.py'],
        'Results': ['outputs/ahfs_ta_results.pt'],
        'Tables': [],
        'Figures': [],
        'LaTeX Tables': []
    }
    
    if os.path.exists('outputs/tables'):
        file_categories['Tables'] = [f"outputs/tables/{f}" for f in os.listdir('outputs/tables') if f.endswith('.csv')]
    
    if os.path.exists('outputs/figures_journal'):
        file_categories['Figures'] = [f"outputs/figures_journal/{f}" for f in os.listdir('outputs/figures_journal') 
                                       if f.endswith('.png') and 'ahfs' in f.lower() or 'ablation' in f.lower() or 'temporal' in f.lower() or 'llm' in f.lower()]
    
    if os.path.exists('outputs/latex_tables'):
        file_categories['LaTeX Tables'] = [f"outputs/latex_tables/{f}" for f in os.listdir('outputs/latex_tables')]
    
    for category, files in file_categories.items():
        print(f"\n{category}:")
        existing_files = [f for f in files if os.path.exists(f)]
        if existing_files:
            for file in existing_files:
                print(f"  ✅ {file}")
                summary['files_generated'].append(file)
        else:
            print(f"  ⏳ No files yet")
    
    # Save summary
    summary_text = f"""
===============================================================================
AHFS-TA IMPLEMENTATION FINAL SUMMARY
===============================================================================
Generated: {summary['timestamp']}

IMPLEMENTATION STATUS:
"""
    for item, status in summary['implementation_status'].items():
        summary_text += f"  {status} {item}\n"
    
    if summary['key_results']:
        summary_text += "\nKEY PERFORMANCE METRICS:\n"
        for metric, value in summary['key_results'].items():
            summary_text += f"  {metric:.<30} {value}\n"
    
    summary_text += f"\nFILES GENERATED: {len(summary['files_generated'])}\n"
    for file in summary['files_generated']:
        summary_text += f"  ✓ {file}\n"
    
    summary_text += """
===============================================================================
NEXT STEPS FOR THESIS:
===============================================================================
1. Run ablation_study_comparison.py to generate comparison tables
2. Run generate_visualizations.py to create all figures
3. Copy LaTeX tables from outputs/latex_tables/ to thesis
4. Copy figures from outputs/figures_journal/ to thesis FIGURES/ folder
5. Update Chapter 5 Results section with actual performance metrics
6. Compile thesis and verify all tables/figures appear correctly

===============================================================================
"""
    
    with open('FINAL_RESULTS_SUMMARY.txt', 'w') as f:
        f.write(summary_text)
    
    print("\n" + "="*80)
    print("✅ Summary saved to FINAL_RESULTS_SUMMARY.txt")
    print("="*80 + "\n")
    
    print(summary_text)
    
    return summary


if __name__ == "__main__":
    generate_latex_tables()
    summary = generate_results_summary()
