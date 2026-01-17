"""
Re-run 3 Failed Models with Fixed Code
=======================================
- mT5-Base: Verify low scores (should be ~4-5%)
- PEGASUS-XSum: Network issue fixed (need stable connection)
- LED-ArXiv: Code fixed (removed attention window optimization)
"""

from baseline_evaluator import ResumableEvaluator
import pandas as pd
from config import DATA_PATHS, EVAL_CONFIG
import time

def main():
    print("\n" + "="*80)
    print("RE-RUNNING 3 FAILED MODELS")
    print("="*80)
    print()
    print("Models to evaluate:")
    print("  1. mT5-Base (verify low scores)")
    print("  2. PEGASUS-XSum (network issue - need stable connection)")
    print("  3. LED-ArXiv (code fixed)")
    print()
    print("Expected total time: 4-6 hours")
    print("Make sure you have:")
    print("  - Stable internet connection")
    print("  - Laptop plugged in (won't sleep)")
    print("  - No other heavy apps running")
    print("="*80 + "\n")
    
    # Ask for confirmation
    proceed = input("Ready to start? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Aborted. Run again when ready.")
        return
    
    # Initialize evaluator
    evaluator = ResumableEvaluator()
    
    # Load test data
    test_df = pd.read_csv(DATA_PATHS['test_full']).head(EVAL_CONFIG['test_samples'])
    
    # Models to re-run
    models_to_rerun = {
        'mT5-Base': 'google/mt5-base',
        'PEGASUS-XSum': 'google/pegasus-xsum',
        'LED-ArXiv': 'allenai/led-large-16384-arxiv'
    }
    
    all_results = []
    start_time = time.time()
    
    for i, (model_name, model_id) in enumerate(models_to_rerun.items(), 1):
        print("\n" + "="*80)
        print(f"PROGRESS: {i}/3 - {model_name}")
        print("="*80)
        
        model_start = time.time()
        
        try:
            result = evaluator.evaluate_single_model(
                model_name,
                model_id,
                test_df,
                resume=False  # Start fresh for clean results
            )
            
            if result:
                all_results.append(result)
                model_time = time.time() - model_start
                
                print(f"\n✓ {model_name} COMPLETED!")
                print(f"  ROUGE-1: {result['rouge1_mean']:.2f}%")
                print(f"  ROUGE-2: {result['rouge2_mean']:.2f}%")
                print(f"  ROUGE-L: {result['rougeL_mean']:.2f}%")
                print(f"  Success Rate: {result['success_rate']:.1f}%")
                print(f"  Time: {model_time/3600:.2f} hours\n")
            else:
                print(f"\n✗ {model_name} FAILED!\n")
                
        except Exception as e:
            print(f"\n✗ {model_name} FAILED with error:")
            print(f"  {str(e)[:200]}\n")
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("RE-EVALUATION COMPLETE")
    print("="*80)
    print(f"Models completed: {len(all_results)}/3")
    print(f"Total time: {total_time/3600:.2f} hours")
    print()
    
    if len(all_results) > 0:
        print("Results:")
        for result in all_results:
            print(f"  {result['model_name']:20s}: ROUGE-2 = {result['rouge2_mean']:5.2f}%  |  Success = {result['success_rate']:5.1f}%")
    
    print("="*80 + "\n")
    
    # Show what to do next
    if len(all_results) == 3:
        print("🎉 SUCCESS! All 3 models completed!")
        print()
        print("NEXT STEPS:")
        print("1. Add to backup folder:")
        print("   xcopy results\\baselines\\mt5_base results_backup_successful\\8_working_models\\mt5_base /E /I /Y")
        print("   xcopy results\\baselines\\pegasus_xsum results_backup_successful\\8_working_models\\pegasus_xsum /E /I /Y")
        print("   xcopy results\\baselines\\led_arxiv results_backup_successful\\8_working_models\\led_arxiv /E /I /Y")
        print()
        print("2. Update SUCCESS_REPORT.txt to include all 10 models")
        print()
        print("3. Commit to GitHub:")
        print("   git add results_backup_successful/")
        print("   git commit -m 'Complete: All 10 baseline models evaluated'")
        print("   git push origin main")
        print()
        print("4. Apply factuality module to best model")
    else:
        print(f"⚠️ Only {len(all_results)}/3 models completed.")
        print("Check errors above and re-run if needed.")
    
    return all_results


if __name__ == "__main__":
    main()