import rdata
import pandas as pd

data = rdata.read_rda("/Users/taka/Downloads/metz_data.Rda")

for name, obj in data.items():
    print(f"\n=== {name} ===")
    print("Type:", type(obj))
    print("Shape:", getattr(obj, "shape", None))
    print("Preview:")
    print(str(obj)[:2000])

    if isinstance(obj, pd.DataFrame):
        obj.to_csv(f"metz_{name}.csv", index=False)
    elif isinstance(obj, pd.Series):
        obj.to_csv(f"metz_{name}.csv")
    elif hasattr(obj, "shape") and len(obj.shape) == 2:
        pd.DataFrame(obj).to_csv(f"metz_{name}.csv", index=False)