"""
Comprehensive Evaluation of FGR System
=======================================
Compare FGR with all 10 baseline models
Date: January 2025
"""

from factuality_guided_reranking import FactualityGuidedReranker
from factuality_module import FactualityChecker
from rouge_score import rouge_scorer
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
import time


def evaluate_fgr_on_newssumm(num_samples=200):
    """
    Evaluate FGR on NewsSumm test set
    """
    print("\n" + "="*80)
    print("EVALUATING FGR ON NEWSSUMM")
    print("="*80)
    
    # Load test data
    print("\n Loading test data...")
    test_df = pd.read_csv('data/processed/test_full.csv').head(num_samples)
    print(f"  Samples: {len(test_df)}")
    
    # Initialize FGR
    print("\n Initializing FGR system...")
    fgr = FactualityGuidedReranker(device='cpu')
    
    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    fact_checker = FactualityChecker()
    
    # Results storage
    results = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    entity_scores, temporal_scores, semantic_scores, factuality_scores = [], [], [], []
    failed = []
    
    print("\n Generating FGR summaries...")
    start_time = time.time()
    
    for idx in tqdm(range(len(test_df)), desc="FGR Evaluation"):
        try:
            row = test_df.iloc[idx]
            article = str(row['article'])
            reference = str(row['summary'])
            
            # Generate FGR summary
            fgr_summary, metadata = fgr.generate_fgr_summary(article, reference, verbose=False)
            
            # ROUGE scores
            rouge_scores = rouge_scorer_obj.score(reference, fgr_summary)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
            
            # Factuality scores
            fact_scores = fact_checker.compute_factuality_score(fgr_summary, article)
            entity_scores.append(fact_scores['entity'])
            temporal_scores.append(fact_scores['temporal'])
            semantic_scores.append(fact_scores['semantic'])
            factuality_scores.append(fact_scores['overall'])
            
            # Store detailed result
            results.append({
                'sample_id': int(idx),
                'article_length': len(article.split()),
                'reference_length': len(reference.split()),
                'fgr_summary': fgr_summary,
                'fgr_length': len(fgr_summary.split()),
                'rouge1': round(rouge_scores['rouge1'].fmeasure, 4),
                'rouge2': round(rouge_scores['rouge2'].fmeasure, 4),
                'rougeL': round(rouge_scores['rougeL'].fmeasure, 4),
                'entity': round(fact_scores['entity'], 3),
                'temporal': round(fact_scores['temporal'], 3),
                'semantic': round(fact_scores['semantic'], 3),
                'factuality': round(fact_scores['overall'], 3),
                'num_models_used': metadata['num_candidates'],
                'num_sentences_selected': metadata['num_selected']
            })
            
        except Exception as e:
            failed.append(idx)
            print(f"\n Error on sample {idx}: {str(e)[:100]}")
            continue
    
    elapsed = time.time() - start_time
    
    # Aggregate results
    aggregate = {
        'model_name': 'FGR (Proposed)',
        'model_type': 'Factuality-Guided Re-ranking Ensemble',
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(test_df),
        'successful_samples': len(results),
        'failed_samples': len(failed),
        'success_rate': round(len(results) / len(test_df) * 100, 2),
        
        # ROUGE scores
        'rouge1_mean': round(np.mean(rouge1_scores) * 100, 2),
        'rouge2_mean': round(np.mean(rouge2_scores) * 100, 2),
        'rougeL_mean': round(np.mean(rougeL_scores) * 100, 2),
        'rouge1_std': round(np.std(rouge1_scores) * 100, 2),
        'rouge2_std': round(np.std(rouge2_scores) * 100, 2),
        'rougeL_std': round(np.std(rougeL_scores) * 100, 2),
        
        # Factuality scores
        'entity_mean': round(np.mean(entity_scores), 3),
        'temporal_mean': round(np.mean(temporal_scores), 3),
        'semantic_mean': round(np.mean(semantic_scores), 3),
        'factuality_mean': round(np.mean(factuality_scores), 3),
        'entity_std': round(np.std(entity_scores), 3),
        'temporal_std': round(np.std(temporal_scores), 3),
        'semantic_std': round(np.std(semantic_scores), 3),
        'factuality_std': round(np.std(factuality_scores), 3),
        
        # Efficiency
        'evaluation_time_seconds': round(elapsed, 2),
        'time_per_sample_seconds': round(elapsed / len(results), 2),
        'time_hours': round(elapsed / 3600, 2),
    }
    
    # Print results
    print("\n" + "="*80)
    print("FGR EVALUATION RESULTS")
    print("="*80)
    print(f"Success: {aggregate['successful_samples']}/{aggregate['total_samples']} ({aggregate['success_rate']:.1f}%)")
    
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {aggregate['rouge1_mean']:.2f} ± {aggregate['rouge1_std']:.2f}")
    print(f"  ROUGE-2: {aggregate['rouge2_mean']:.2f} ± {aggregate['rouge2_std']:.2f}")
    print(f"  ROUGE-L: {aggregate['rougeL_mean']:.2f} ± {aggregate['rougeL_std']:.2f}")
    
    print(f"\nFactuality Scores:")
    print(f"  Entity: {aggregate['entity_mean']:.3f} ± {aggregate['entity_std']:.3f}")
    print(f"  Temporal: {aggregate['temporal_mean']:.3f} ± {aggregate['temporal_std']:.3f}")
    print(f"  Semantic: {aggregate['semantic_mean']:.3f} ± {aggregate['semantic_std']:.3f}")
    print(f"  Overall: {aggregate['factuality_mean']:.3f} ± {aggregate['factuality_std']:.3f}")
    
    print(f"\nEfficiency:")
    print(f"  Total time: {aggregate['time_hours']:.2f} hours")
    print(f"  Time per sample: {aggregate['time_per_sample_seconds']:.2f} seconds")
    
    print("="*80 + "\n")
    
    # Save results
    output_dir = Path('results/proposed_model/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'fgr_aggregate_scores.json', 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    with open(output_dir / 'fgr_detailed_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f" Results saved to: {output_dir}\n")
    
    return aggregate, results


def compare_with_baselines():
    """
    Load all baseline results and compare with FGR
    """
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)
    
    # Load baseline results
    baselines_dir = Path('results/baselines/')
    baseline_models = [
        'bart_base',
        'bart_large_cnn',
        'distilbart',
        'led_arxiv',
        't5_large',
        'pegasus_cnn',
        't5_base',
        'longt5_base',
        'pegasus_xsum',
        'mt5_base'
    ]
    
    all_results = []
    
    print("\n Loading baseline results...")
    for model_dir in baseline_models:
        model_path = baselines_dir / model_dir / 'aggregate_scores.json'
        if model_path.exists():
            with open(model_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)
            print(f"   {result['model_name']}")
    
    # Load FGR results
    fgr_path = Path('results/proposed_model/fgr_aggregate_scores.json')
    if fgr_path.exists():
        with open(fgr_path, 'r') as f:
            fgr_result = json.load(f)
            all_results.append(fgr_result)
        print(f"  FGR (Proposed)")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    
    # Sort by ROUGE-2
    all_results.sort(key=lambda x: x.get('rouge2_mean', 0), reverse=True)
    
    print(f"\n{'Rank':<6}{'Model':<25}{'R-1':<8}{'R-2':<8}{'R-L':<8}{'Fact':<8}{'Success':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(all_results, 1):
        model = result.get('model_name', 'Unknown')[:24]
        r1 = result.get('rouge1_mean', 0)
        r2 = result.get('rouge2_mean', 0)
        rl = result.get('rougeL_mean', 0)
        fact = result.get('factuality_mean', 0) if 'factuality_mean' in result else 0
        success = result.get('success_rate', 0)
        
        # Highlight FGR
        print(f"{rank:<6}{model:<30}{r2:<10.2f}{fact:<8.3f}")

    
    print("="*80 + "\n")
    
    # Analysis
    print("KEY FINDINGS:")
    
    fgr_r2 = fgr_result.get('rouge2_mean', 0)
    best_baseline_r2 = max([r.get('rouge2_mean', 0) for r in all_results if 'FGR' not in r.get('model_name', '')])
    improvement = ((fgr_r2 - best_baseline_r2) / best_baseline_r2) * 100
    
    print(f"1. FGR achieves {fgr_r2:.2f}% ROUGE-2")
    print(f"2. Best baseline: {best_baseline_r2:.2f}% ROUGE-2")
    print(f"3. Improvement: {improvement:+.2f}%")
    
    fgr_fact = fgr_result.get('factuality_mean', 0)
    print(f"4. FGR factuality: {fgr_fact:.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Save comparison
    output_path = Path('results/comparison_table.json')
    with open(output_path, 'w') as f:
        json.dump({
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': all_results
        }, f, indent=2)
    
    print(f" Comparison saved to: {output_path}\n")
    
    return all_results


def main():
    """
    Complete evaluation pipeline
    """
    print("\n" + "="*80)
    print("FGR COMPREHENSIVE EVALUATION")
    print("="*80)
    print("\nThis will:")
    print("1. Evaluate FGR on 200 NewsSumm samples")
    print("2. Compare with 10 baseline models")
    print("3. Generate comparison tables")
    print("\nEstimated time: 4-6 hours (200 samples)")
    print("="*80)
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Aborted.")
        return
    
    # Step 1: Evaluate FGR
    print("\n" + "="*80)
    print("STEP 1: EVALUATING FGR")
    print("="*80)
    fgr_aggregate, fgr_detailed = evaluate_fgr_on_newssumm(num_samples=200)
    
    # Step 2: Compare with baselines
    print("\n" + "="*80)
    print("STEP 2: COMPARING WITH BASELINES")
    print("="*80)
    all_results = compare_with_baselines()
    
    # Step 3: Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nResults saved in:")
    print("  - results/proposed_model/fgr_aggregate_scores.json")
    print("  - results/proposed_model/fgr_detailed_predictions.json")
    print("  - results/comparison_table.json")
    print("\nNext steps:")
    print("  1. Analyze results")
    print("  2. Create visualizations")
    print("  3. Write paper")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()