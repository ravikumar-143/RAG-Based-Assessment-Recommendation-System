import pandas as pd
from pathlib import Path

xl_path = Path('evulation dataset/Gen_AI Dataset.xlsx')
wb = pd.ExcelFile(xl_path)
print('Sheets:', wb.sheet_names)
for sheet in wb.sheet_names:
    df = wb.parse(sheet)
    print(f"\nSheet {sheet} shape {df.shape}")
    print(df.head())
