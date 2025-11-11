#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_distances.py
Extracts 'top 1 distance' values from log lines, 
plots their distribution, and saves the plots as PNG files.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os


def open_log(file_path):
    with open(file_path, "r") as f:
        logs = f.readlines()

    identified = []
    for log in logs:
        if "UNIDENTIFIED" in log:
            continue
        if "IDENTIFIED" in log:
            identified.append(log)
    return identified


def extract_distances(identified, start_idx):
    distances = []
    for line in identified[start_idx:]:
        match = re.search(r"'top 1 distance': ([0-9\.eE+-]+)", line)
        if match:
            distances.append(float(match.group(1)))
    return distances
    

def analyze(distances):
    # Convert to numpy array for convenience
    distances = np.array(distances)

    # Print summary statistics
    print("Summary Statistics:")
    print(f"Count: {len(distances)}")
    print(f"Mean: {distances.mean():.6f}")
    print(f"Median: {np.median(distances):.6f}")
    print(f"Std Dev: {distances.std():.6f}")
    print(f"Min: {distances.min():.6f}")
    print(f"Max: {distances.max():.6f}")
    

def draw_histogram(distances, output_dir):
    # === Output directory ===    
    os.makedirs(output_dir, exist_ok=True)

    # === Plot: Histogram ===
    plt.figure(figsize=(10, 4))
    plt.hist(distances, bins=10, edgecolor='black')
    plt.title("Distribution of 'top 1 distance'")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    hist_path = os.path.join(output_dir, "distance_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved in '{output_dir}/':")
    print(f"  - {hist_path}")
    

def draw_boxplot(distances, output_dir):
    # === Output directory ===    
    os.makedirs(output_dir, exist_ok=True)
    
    # === Plot: Box plot ===
    plt.figure(figsize=(6, 4))
    plt.boxplot(distances, vert=False)
    plt.title("Box Plot of 'top 1 distance'")
    plt.xlabel("Distance")
    plt.grid(True, linestyle='--', alpha=0.6)
    box_path = os.path.join(output_dir, "distance_boxplot.png")
    plt.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved in '{output_dir}/':")
    print(f"  - {box_path}")


if __name__=="__main__":
    identified = open_log("./chat.log")    
    distances = extract_distances(identified, 9450)
    analyze(distances)
    draw_histogram(distances, "./plots")
    draw_boxplot(distances, "./plots")
