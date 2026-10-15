
#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Input XLS file
input_path = Path(
    "/Users/taka/Downloads/41589_2011_BFnchembio530_MOESM137_ESM.xls"
)

# Output directory (same as input)
out_dir = input_path.parent

# Read all sheets
sheets = pd.read_excel(
    input_path,
    sheet_name=None,
    engine="xlrd"
)

for sheet_name, df in sheets.items():
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in sheet_name
    )

    output = out_dir / f"{input_path.stem}_{safe_name}.csv"

    df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSheet: {sheet_name}")
    print(f"Shape: {df.shape}")
    print(f"Saved: {output}")
    print("Columns:", df.columns.tolist()[:20])
    print(df.iloc[:3, :8].to_string(index=False))

print("\nConversion complete.")
