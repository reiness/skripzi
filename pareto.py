import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np

# --- Prepare the data ---
data = {
    "model": ["9s", "9m", "9c", "rtdetr-l", "rtdetr-x", "10s", "10m", "10b"],
    "map50_day": [0.3972, 0.5739, 0.5324, 0.3632, 0.4924, 0.3960, 0.5047, 0.4900],
    "inference_day": [256.12, 606.62, 860.54, 1380.11, 2583.54, 221.08, 515.80, 770.25],
    "map50_night": [0.4439, 0.5707, 0.5704, 0.3513, 0.5205, 0.4114, 0.5391, 0.4652],
    "inference_night": [236.07, 588.45, 829.73, 1289.16, 2444.03, 246.00, 574.25, 765.94]
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

# --- Define marker shapes per family ---
marker_map = {
    "9": "^",        # triangle
    "10": "*",       # star
    "rtdetr": "s"    # square
}

# --- Plotting ---
plt.figure(figsize=(10, 6))

edge_day = "red"
edge_night = "blue"

# Plot each point with appropriate color, marker, and annotated label
for idx, row in df.iterrows():
    family, level = parse_model(row['model'])
    if family is None:
        continue
    facecolor = palette[family][level]
    marker = marker_map[family]

    # Day point
    plt.scatter(
        row["inference_day"], row["map50_day"],
        marker=marker,
        facecolor=facecolor,
        edgecolor=edge_day,
        s=120, linewidth=1.5
    )
    # Annotate Day
    y_offset_day = row["map50_day"] * (0.975 if row["model"] in ['10s', '10m'] else 1.01)
    t_day = plt.annotate(
        row['model'],
        (row["inference_day"] * 1.01, y_offset_day),
        color=edge_day, fontsize=9
    )
    t_day.set_path_effects([
        path_effects.Stroke(linewidth=0.5, foreground='black'),
        path_effects.Normal()
    ])

    # Night point
    plt.scatter(
        row["inference_night"], row["map50_night"],
        marker=marker,
        facecolor=facecolor,
        edgecolor=edge_night,
        s=120, linewidth=1.5
    )
    # Annotate Night
    y_offset_night = row["map50_night"] * 1.01
    t_night = plt.annotate(
        row['model'],
        (row["inference_night"] * 1.01, y_offset_night),
        color=edge_night, fontsize=9
    )
    t_night.set_path_effects([
        path_effects.Stroke(linewidth=0.5, foreground='black'),
        path_effects.Normal()
    ])

# Draw Pareto lines
pareto_day_sorted = pareto_day.sort_values("inference_day")
plt.plot(
    pareto_day_sorted["inference_day"], pareto_day_sorted["map50_day"],
    color=edge_day, linestyle='-', linewidth=2, label='Pareto Day Frontier'
)

pareto_night_sorted = pareto_night.sort_values("inference_night")
plt.plot(
    pareto_night_sorted["inference_night"], pareto_night_sorted["map50_night"],
    color=edge_night, linestyle='-', linewidth=2, label='Pareto Night Frontier'
)

# --- Labels & Aesthetics ---
plt.xlabel("Inference Time (ms)", fontsize=12)
plt.ylabel("mAP50", fontsize=12)
plt.title("Pareto Frontier Visualization", fontsize=14)

# Custom legend entries
import matplotlib.lines as mlines
day_marker = mlines.Line2D([], [], color=edge_day, marker='o', linestyle='None',
                           markersize=10, markerfacecolor='none', label='Well-lit')
night_marker = mlines.Line2D([], [], color=edge_night, marker='o', linestyle='None',
                             markersize=10, markerfacecolor='none', label='Low-light')

# # Marker legend for families
# triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
#                          markersize=10, label='YOLOv9')
# star     = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
#                          markersize=10, label='YOLOv10')
# square   = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
#                          markersize=10, label='RTDETR')

plt.legend(handles=[
    day_marker, night_marker,
    plt.Line2D([], [], color=edge_day, label='Pareto well-lit Frontier'),
    plt.Line2D([], [], color=edge_night, label='Pareto low-light Frontier'),
    # triangle, star, square
], loc='best')

plt.grid(True)
plt.tight_layout()
plt.show()
