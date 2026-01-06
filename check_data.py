import pandas as pd
from pathlib import Path

print('='*70)
print('Checking NewsSumm Dataset')
print('='*70)

data_path = Path('data/raw/newsumm')

if not data_path.exists():
    print('\n❌ Dataset folder not found!')
    print('   Please create: data/raw/newsumm/')
    print('   And place train.csv, val.csv, test.csv there')
else:
    files = ['train.csv', 'val.csv', 'test.csv']
    for file in files:
        filepath = data_path / file
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, nrows=5)
                print(f'\n✅ {file}: Found!')
                print(f'   Columns: {df.columns.tolist()}')
                print(f'   Shape (first 5 rows): {df.shape}')
                
                # Show first few characters of first row
                if 'article' in df.columns:
                    print(f'   First article preview:')
                    print(f'      {df["article"].iloc[0][:100]}...')
                if 'summary' in df.columns:
                    print(f'   First summary preview:')
                    print(f'      {df["summary"].iloc[0][:100]}...')
                    
            except Exception as e:
                print(f'\n❌ {file}: Error reading - {e}')
        else:
            print(f'\n❌ {file}: Not found in {data_path}')

print('\n' + '='*70)