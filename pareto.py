import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np

# --- Prepare the data ---
data = {
    "model": ["9s", "9m", "9c", "rtdetr-l", "rtdetr-x", "10s", "10m", "10b"],
    "map50_day": [0.3960, 0.555, 0.5465, 0.3632, 0.4924, 0.4056, 0.5165, 0.4935],
    "inference_day": [199.39, 442.15, 595.72, 1122.73, 1968.66, 176.96, 423.65, 593.39],
    "map50_night": [0.4365, 0.5566, 0.5629, 0.3513, 0.5205, 0.4336, 0.5473, 0.4666],
    "inference_night": [168.98, 392.85, 613.88, 1008.14, 2123.71, 148.62, 377.96, 559.03]
}
df = pd.DataFrame(data)

# --- Pareto Frontier Function ---
def compute_pareto(df, score_col, cost_col):
    is_dominated = np.zeros(df.shape[0], dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (df.loc[j, score_col] >= df.loc[i, score_col] and
                df.loc[j, cost_col] <= df.loc[i, cost_col] and
                (df.loc[j, score_col] > df.loc[i, score_col] or df.loc[j, cost_col] < df.loc[i, cost_col])):
                is_dominated[i] = True
                break
    return df[~is_dominated]

# --- Compute Pareto frontiers ---
pareto_day = compute_pareto(df, "map50_day", "inference_day")
pareto_night = compute_pareto(df, "map50_night", "inference_night")

# --- Color palette for families and levels ---
palette = {
    "9": {
        "s": "#e0ffff",
        "m": "#00ced1",
        "c": "#008b8b"
    },
    "rtdetr": {
        "l": "#ffffe0",
        "x": "#b8860b"
    },
    "10": {
        "s": "#ffe0ff",
        "m": "#ff00ff",
        "b": "#8b008b"
    }
}

def parse_model(model):
    if model.startswith("9"):
        return "9", model[1]
    elif model.startswith("10"):
        return "10", model[2]
    elif model.startswith("rtdetr"):
        return "rtdetr", model.split("-")[1]
    else:
        return None, None

# --- Plotting ---
plt.figure(figsize=(10, 6))

edge_day = "red"
edge_night = "blue"

# Plot each point with appropriate color and annotated label
for idx, row in df.iterrows():
    family, level = parse_model(row['model'])
    if family is None:
        continue
    facecolor = palette[family][level]

    # Day point
    plt.scatter(row["inference_day"], row["map50_day"],
                facecolor=facecolor, edgecolor=edge_day, s=120, linewidth=1.5)

    # Adjust label placement
    y_offset_day = row["map50_day"] * (0.975 if row["model"] in ['10s', '10m'] else 1.01)

    t_day = plt.annotate(row['model'],
                         (row["inference_day"] * 1.01, y_offset_day),
                         color=edge_day, fontsize=9)
    t_day.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                            path_effects.Normal()])

    # Night point
    plt.scatter(row["inference_night"], row["map50_night"],
                facecolor=facecolor, edgecolor=edge_night, s=120, linewidth=1.5)

    # Adjust label placement
    y_offset_night = row["map50_night"] * (0.98 if row["model"] in ['10s', '10m'] else 1.01)

    t_night = plt.annotate(row['model'],
                           (row["inference_night"] * 1.01, y_offset_night),
                           color=edge_night, fontsize=9)
    t_night.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                              path_effects.Normal()])

# Draw Pareto lines
pareto_day_sorted = pareto_day.sort_values("inference_day")
plt.plot(pareto_day_sorted["inference_day"], pareto_day_sorted["map50_day"],
         color=edge_day, linestyle='-', linewidth=2, label='Pareto Day Frontier')

pareto_night_sorted = pareto_night.sort_values("inference_night")
plt.plot(pareto_night_sorted["inference_night"], pareto_night_sorted["map50_night"],
         color=edge_night, linestyle='-', linewidth=2, label='Pareto Night Frontier')

# --- Labels & Aesthetics ---
plt.xlabel("Inference Time (ms)", fontsize=12)
plt.ylabel("mAP50", fontsize=12)
plt.title("Pareto Frontier Visualization", fontsize=14)

# Custom hollow circle legends
import matplotlib.lines as mlines
day_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                           markersize=10, markerfacecolor='none', label='Day')
night_marker = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                             markersize=10, markerfacecolor='none', label='Night')

# Legend with both pareto and custom markers
plt.legend(handles=[day_marker, night_marker,
                    plt.Line2D([], [], color=edge_day, label='Pareto Day Frontier'),
                    plt.Line2D([], [], color=edge_night, label='Pareto Night Frontier')],
           loc='best')

plt.grid(True)
plt.tight_layout()
plt.show()

