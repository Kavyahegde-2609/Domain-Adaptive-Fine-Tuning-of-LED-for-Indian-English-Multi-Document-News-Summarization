
import pandas as pd
from pathlib import Path

print("="*70)
print("Splitting NewsSumm Dataset")
print("="*70)

# Load the big CSV
input_file = Path('data/raw/newsumm/NewsSumm Dataset.csv')

print(f"\nLoading {input_file.name}...")
df = pd.read_csv(input_file)

print(f"✅ Loaded {len(df):,} articles")
print(f"   Columns: {df.columns.tolist()}")

# Rename columns to standard names
df = df.rename(columns={
    'article_text': 'article',
    'human_summary': 'summary',
    'news_category': 'category',
    'published_date\n': 'date'  # Note the \n
})

# Clean column names (remove any extra whitespace)
df.columns = df.columns.str.strip()

print(f"\n✅ Renamed columns: {df.columns.tolist()}")

# Remove rows with missing data
print("\nCleaning data...")
original_len = len(df)
df = df.dropna(subset=['article', 'summary'])
print(f"   Removed {original_len - len(df):,} rows with missing article/summary")

# Split: 90% train, 5% val, 5% test
train_size = int(0.90 * len(df))
val_size = int(0.05 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print("\n" + "="*70)
print("Split sizes:")
print(f"   Train: {len(train_df):,} articles")
print(f"   Val:   {len(val_df):,} articles")
print(f"   Test:  {len(test_df):,} articles")
print("="*70)

# Save splits
output_path = Path('data/raw/newsumm')

train_df.to_csv(output_path / 'train.csv', index=False)
print(f"\n✅ Saved train.csv")

val_df.to_csv(output_path / 'val.csv', index=False)
print(f"✅ Saved val.csv")

test_df.to_csv(output_path / 'test.csv', index=False)
print(f"✅ Saved test.csv")

print("\n" + "="*70)
print("✅ DATASET SPLIT COMPLETE!")
print("="*70)