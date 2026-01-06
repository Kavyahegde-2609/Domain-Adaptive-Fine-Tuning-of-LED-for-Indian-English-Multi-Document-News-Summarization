import pandas as pd

df = pd.read_csv('data/raw/newsumm/NewsSumm Dataset.csv')
df = df.drop_duplicates().dropna(subset=['article_text', 'human_summary'])

df['article_len'] = df['article_text'].str.len()
df['summary_len'] = df['human_summary'].str.len()

print('Article length stats:')
print(df['article_len'].describe())
print(f'\nVery short articles (<200 chars): {(df["article_len"] < 200).sum():,}')
print(f'Very long articles (>20000 chars): {(df["article_len"] > 20000).sum():,}')
print(f'\nVery short summaries (<20 chars): {(df["summary_len"] < 20).sum():,}')
print(f'\nTotal after basic cleaning: {len(df):,}')
print(f'Difference from paper (317,498): {len(df) - 317498:,}')