from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

print("="*70)
print("Quick Baseline: Pretrained BART")
print("="*70)

# Load pretrained model
print("\nLoading pretrained BART-CNN model...")
print("(This will download ~1.6GB on first run)")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
print("✅ Model loaded!")

# Load test data (just 50 samples for speed)
print("\nLoading test data...")
test_df = pd.read_csv('data/raw/newsumm/test.csv', nrows=50)
print(f"✅ Loaded {len(test_df)} test samples")

# Generate summaries
results = []
errors = 0

print("\n🚀 Generating summaries...")
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    try:
        article = str(row['article'])[:1024]
        summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
        
        results.append({
            'article_id': int(idx),
            'reference': str(row['summary']),
            'generated': summary[0]['summary_text']
        })
    except Exception as e:
        print(f"\nError on article {idx}: {e}")
        errors += 1
        continue

print(f"\n✅ Generated {len(results)} summaries")
print(f"   Errors: {errors}")

# Create results directory
Path('results/baselines').mkdir(parents=True, exist_ok=True)

# Save results
with open('results/baselines/bart_quick_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Results saved!")
print("="*70)