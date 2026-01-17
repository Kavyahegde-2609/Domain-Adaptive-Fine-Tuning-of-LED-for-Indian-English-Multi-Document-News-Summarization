"""
Collect All 10 Model Scores - Complete Working Version
Date: January 11, 2025
"""

import json
import pandas as pd
from pathlib import Path

def collect_all_scores():
    """Collect scores from all 10 models"""
    
    base_dir = Path('results_backup_successful/8_working_models')
    
    model_folders = [
        'bart_base',
        'bart_large_cnn',
        'distilbart',
        'led_arxiv',
        'longt5_base',
        'mt5_base',
        'pegasus_cnn',
        'pegasus_xsum',
        't5_base',
        't5_large'
    ]
    
    all_scores = []
    
    print("\n" + "="*80)
    print("COLLECTING ALL 10 MODEL SCORES")
    print("="*80 + "\n")
    
    for folder in model_folders:
        score_file = base_dir / folder / 'aggregate_scores.json'
        
        if score_file.exists():
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_scores.append({
                'Model': data['model_name'],
                'ROUGE-1': data['rouge1_mean'],
                'ROUGE-2': data['rouge2_mean'],
                'ROUGE-L': data['rougeL_mean'],
                'ROUGE-1 Std': data['rouge1_std'],
                'ROUGE-2 Std': data['rouge2_std'],
                'ROUGE-L Std': data['rougeL_std'],
                'Total Samples': data['total_samples'],
                'Successful': data['successful_samples'],
                'Failed': data['failed_samples'],
                'Success Rate': data['success_rate'],
                'Time Hours': round(data['evaluation_time_seconds'] / 3600, 2),
            })
            
            print(f" {data['model_name']:<20} R-2: {data['rouge2_mean']:.2f}%")
        else:
            print(f" {folder:<20} NOT FOUND")
    
    if not all_scores:
        print("\n No scores found!")
        return None
    
    df = pd.DataFrame(all_scores)
    df = df.sort_values('ROUGE-2', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    print("\n" + "="*95)
    print("ALL 10 MODELS - RANKED BY ROUGE-2")
    print("="*95)
    
    for _, row in df.iterrows():
        star = ' (SELECTED)' if 'LED' in row['Model'] else ''
        
        if row['Rank'] == 1:
            medal = 'GOLD   '
        elif row['Rank'] == 2:
            medal = 'SILVER '
        elif row['Rank'] == 3:
            medal = 'BRONZE '
        else:
            medal = '       '
        
        print(f"{medal}{row['Rank']:>2}. {row['Model']:<22} "
              f"R-1: {row['ROUGE-1']:>5.2f}%  R-2: {row['ROUGE-2']:>5.2f}%  "
              f"R-L: {row['ROUGE-L']:>5.2f}%  Success: {row['Success Rate']:>6.1f}%  "
              f"Time: {row['Time Hours']:>5.2f}h{star}")
    
    print("="*95)
    
    best = df.iloc[0]
    led_df = df[df['Model'].str.contains('LED', na=False)]
    led = led_df.iloc[0] if len(led_df) > 0 else None
    
    print(f"\nBEST OVERALL (Highest ROUGE-2):")
    print(f"  {best['Model']}: {best['ROUGE-2']:.2f}% +/- {best['ROUGE-2 Std']:.2f}%")
    print(f"  Success: {best['Success Rate']:.1f}% | Time: {best['Time Hours']:.2f}h")
    
    if led is not None:
        print(f"\nLED-ArXiv (SELECTED FOR FACTUALITY MODULE):")
        print(f"  Rank: #{led['Rank']}")
        print(f"  ROUGE-2: {led['ROUGE-2']:.2f}% +/- {led['ROUGE-2 Std']:.2f}%")
        print(f"  Success: {led['Success Rate']:.1f}%")
        print(f"  Time: {led['Time Hours']:.2f} hours")
        print(f"  Reason: 16K context, 100% success, long document handling")
    
    print(f"\nSTATISTICS:")
    print(f"  Total Models: {len(df)}")
    print(f"  100% Success: {len(df[df['Success Rate'] == 100.0])}")
    print(f"  Total Samples: {df['Successful'].sum():,}")
    print(f"  Total Failed: {df['Failed'].sum()}")
    print(f"  Avg ROUGE-2: {df['ROUGE-2'].mean():.2f}%")
    print(f"  Total Time: {df['Time Hours'].sum():.2f} hours")
    
    output_dir = Path('results_backup_successful')
    
    csv_file = output_dir / 'ALL_10_MODELS_COMPLETE.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV: {csv_file}")
    
    json_file = output_dir / 'ALL_10_MODELS_COMPLETE.json'
    df.to_json(json_file, orient='records', indent=2)
    print(f"Saved JSON: {json_file}")
    
    summary_cols = ['Rank', 'Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Success Rate', 'Time Hours']
    summary_df = df[summary_cols]
    summary_file = output_dir / 'SUMMARY_TABLE.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved Summary: {summary_file}")
    
    print("\n" + "="*95)
    print("COLLECTION COMPLETE!")
    print("="*95 + "\n")
    
    return df


if __name__ == "__main__":
    df = collect_all_scores()
    
    if df is not None:
        print("SUCCESS! All scores collected.\n")
        print("Next steps:")
        print("  1. Commit to GitHub")
        print("  2. Show to mentor")
        print("  3. Apply factuality module to LED-ArXiv\n")