import json
from pathlib import Path

print("=" * 80)
print("CHECKING LED FILES")
print("=" * 80)

# Check LED aggregate file directly
led_file = Path('results/baselines/led_arxiv/aggregate_scores.json')
with open(led_file, 'r') as f:
    led_data = json.load(f)

print("\n📊 LED-ARXIV (from aggregate_scores.json):")
print(f"ROUGE-1: {led_data['rouge1_mean']}")
print(f"ROUGE-2: {led_data['rouge2_mean']}")
print(f"ROUGE-L: {led_data['rougeL_mean']}")
print(f"Success: {led_data['success_rate']}%")
print(f"Samples: {led_data['successful_samples']}/{led_data['total_samples']}")

# Check factuality file structure
fact_file = Path('results/analysis/led_factuality_analysis_n200.json')
with open(fact_file, 'r') as f:
    fact_data = json.load(f)

print("\n📊 FACTUALITY FILE KEYS:")
print(list(fact_data.keys()))

# Try to extract factuality
if 'overall_stats' in fact_data:
    stats = fact_data['overall_stats']
    print("\n📊 LED FACTUALITY:")
    print(f"Entity: {stats.get('entity_mean', 'N/A')}")
    print(f"Temporal: {stats.get('temporal_mean', 'N/A')}")
    print(f"Semantic: {stats.get('semantic_mean', 'N/A')}")
    print(f"Overall: {stats.get('factuality_mean', 'N/A')}")
elif 'entity_mean' in fact_data:
    print("\n📊 LED FACTUALITY:")
    print(f"Entity: {fact_data['entity_mean']}")
    print(f"Temporal: {fact_data['temporal_mean']}")
    print(f"Overall: {fact_data['factuality_mean']}")
else:
    print("\n📊 FACTUALITY DATA STRUCTURE:")
    print(json.dumps(fact_data, indent=2)[:500])