"""
Quick Results Summary - Check Implementation Progress
"""

import os
import torch

print("\n" + "="*80)
print("AHFS-TA IMPLEMENTATION PROGRESS CHECK")
print("="*80 + "\n")

# Check if training log exists
if os.path.exists('ahfs_ta_training.log'):
    print("✓ Training log found")
    with open('ahfs_ta_training.log', 'r') as f:
        lines = f.readlines()
        print(f"  Log lines: {len(lines)}")
        if len(lines) > 0:
            print(f"  Last few lines:")
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")
else:
    print("⏳ Training log not yet created")

print("\n" + "-"*80)

# Check if results file exists
if os.path.exists('outputs/ahfs_ta_results.pt'):
    print("✓ Results file found: outputs/ahfs_ta_results.pt")
    try:
        results = torch.load('outputs/ahfs_ta_results.pt')
        print("\n  Saved Components:")
        for key in results.keys():
            print(f"    - {key}")
        
        if 'metrics' in results:
            print("\n  Performance Metrics:")
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")
                    
    except Exception as e:
        print(f"  Error loading results: {e}")
else:
    print("⏳ Results file not yet created")

print("\n" + "-"*80)

# Check for comparison tables
tables_dir = 'outputs/tables'
if os.path.exists(tables_dir):
    print(f"✓ Tables directory exists: {tables_dir}")
    tables = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
    if tables:
        print(f"\n  Found {len(tables)} CSV files:")
        for table in tables:
            print(f"    - {table}")
    else:
        print("  No CSV files yet")
else:
    print("⏳ Tables directory not yet created")

print("\n" + "-"*80)

# Check for figures
figures_dir = 'outputs/figures_journal'
if os.path.exists(figures_dir):
    print(f"✓ Figures directory exists: {figures_dir}")
    figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    if figures:
        print(f"\n  Found {len(figures)} PNG files:")
        for fig in figures:
            print(f"    - {fig}")
    else:
        print("  No PNG files yet")
else:
    print("⏳ Figures directory not yet created")

print("\n" + "="*80)
print("IMPLEMENTATION STATUS SUMMARY")
print("="*80)

status = {
    '✅ Python Implementation': True,
    '⏳ Model Training': os.path.exists('ahfs_ta_training.log'),
    '⏳ Experimental Results': os.path.exists('outputs/ahfs_ta_results.pt'),
    '⏳ Comparison Tables': os.path.exists('outputs/tables') and len([f for f in os.listdir('outputs/tables') if f.endswith('.csv')]) > 0 if os.path.exists('outputs/tables') else False,
    '⏳ Result Figures': os.path.exists('outputs/figures_journal') and len([f for f in os.listdir('outputs/figures_journal') if f.endswith('.png')]) > 0 if os.path.exists('outputs/figures_journal') else False
}

for item, done in status.items():
    symbol = '✅' if done else '⏳'
    print(f"{symbol} {item.split(' ', 1)[1]}")

print("\n" + "="*80 + "\n")
