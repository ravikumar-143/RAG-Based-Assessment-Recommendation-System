import pandas as pd
from pathlib import Path

xl_path = Path('evulation dataset/Gen_AI Dataset.xlsx')
test_df = pd.read_excel(xl_path, sheet_name='Test-Set')
queries = test_df['Query'].dropna().reset_index(drop=True)
out_path = Path('data/test_dataset.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
queries.to_csv(out_path, index=False)
print(f"Wrote {len(queries)} queries to {out_path}")
