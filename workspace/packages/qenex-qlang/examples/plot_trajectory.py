import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load trajectory data
try:
    df = pd.read_csv("optimization_trajectory.csv")
    print("Loaded trajectory data.")
    print(df.head())
except FileNotFoundError:
    print("Error: optimization_trajectory.csv not found.")
    exit(1)

# Create ASCII plot of Energy vs Step
print("\n--- Optimization Energy Trajectory ---")
max_energy = df["energy"].max()
min_energy = df["energy"].min()
range_energy = max_energy - min_energy

if range_energy == 0: range_energy = 1.0

width = 60
height = 20

plot_grid = [[' ' for _ in range(width)] for _ in range(height)]

for i, row in df.iterrows():
    step = int(row["step"])
    energy = row["energy"]
    
    # Normalize to plot dimensions
    x = int((i / (len(df) - 1)) * (width - 1))
    
    # Invert y (higher energy is higher up, but row 0 is top)
    # y = 0 (top) corresponds to max_energy
    # y = height-1 (bottom) corresponds to min_energy
    y = int(((max_energy - energy) / range_energy) * (height - 1))
    
    if 0 <= x < width and 0 <= y < height:
        plot_grid[y][x] = '*'

# Draw plot
print(f"Energy Range: {min_energy:.4f} Eh to {max_energy:.4f} Eh")
print("-" * (width + 2))
for row in plot_grid:
    print("|" + "".join(row) + "|")
print("-" * (width + 2))
print(f"Steps: 0 to {len(df)-1}")

# Plot Variables
variables = [c for c in df.columns if c not in ["step", "energy"]]
for var in variables:
    print(f"\n--- {var} Trajectory ---")
    max_val = df[var].max()
    min_val = df[var].min()
    range_val = max_val - min_val
    if range_val == 0: range_val = 1.0
    
    plot_grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for i, row in df.iterrows():
        val = row[var]
        x = int((i / (len(df) - 1)) * (width - 1))
        y = int(((max_val - val) / range_val) * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            plot_grid[y][x] = 'o'
            
    print(f"Range: {min_val:.4f} to {max_val:.4f}")
    print("-" * (width + 2))
    for row in plot_grid:
        print("|" + "".join(row) + "|")
    print("-" * (width + 2))
