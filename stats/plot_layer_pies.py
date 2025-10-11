# All comments are in English.
import io
import pandas as pd
import matplotlib.pyplot as plt

csv_text = """layer_id,step_cycles_total,preload_input_cycles,weight_load_cycle,output_drain_cycles,output_store_cycles
0,392005,122332,37748736,41742,13529,
1,97145,30309,9437184,12513,4028,
2,242711,75800,9437184,13731,4409,
3,41871,13074,4718592,2491,805,
4,87423,27306,4718592,2690,870,
5,85167,26602,4718592,4175,1334,
6,17638,5509,2359296,802,257,
7,43838,13697,2359296,2771,872,
8,114272,35708,2359296,20796,6506,
9,10208,3188,589824,235,75,
10,13276,4148,589824,27,10,
11,17912,5596,589824,208,67,
12,734,229,36864,4,2,
"""

# ---- Load CSV ----
df = pd.read_csv(io.StringIO(csv_text))
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

cols = [
    "step_cycles_total",
    "preload_input_cycles",
    "weight_load_cycle",
    "output_drain_cycles",
    "output_store_cycles"
]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# ---- Sum each column ----
agg = df[cols].sum()

labels = [
    "Step total",
    "Preload input",
    "Weight load",
    "Output drain",
    "Output store"
]
sizes = agg.values

# ---- Create table summary ----
total_all = sizes.sum()
percentages = sizes / total_all * 100

summary_df = pd.DataFrame({
    "Latency component": labels,
    "Total cycles": sizes.astype(int),
    "Share (%)": percentages
}).sort_values("Total cycles", ascending=False).reset_index(drop=True)

print(summary_df)

# ---- Plot pie chart ----
plt.figure(figsize=(7,7))
# Use 'explode' to slightly pop out the weight_load slice for emphasis
explode = [0, 0, 0.05, 0, 0]
plt.pie(
    sizes,
    labels=labels,
    autopct=lambda p: '{:.1f}%\n({:.2e})'.format(p, p*total_all/100),
    startangle=90,
    explode=explode
)
plt.title("Latency distribution including step_cycles_total")
plt.axis("equal")
plt.tight_layout()

out_path = "./latency_distribution_pie_with_total.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"Saved pie chart with total to: {out_path}")
