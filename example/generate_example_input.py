# generate_test_data.py

import numpy as np
import pandas as pd

print("Generating sample CSV files for CLI testing...")

# --- Configuration ---
n_cells = 500
features = ["CD4", "CD8A", "GNLY", "MS4A1", "CD19", "NKG7", "FOXP3", "OtherGene1", "OtherGene2"]
groups = {
    "T-helper": 0.3,
    "T-cytotoxic": 0.3,
    "B-cells": 0.25,
    "NK": 0.15,
}

# Define which genes are markers for which groups
marker_map = {
    "T-helper": ["CD4", "FOXP3"],
    "T-cytotoxic": ["CD8A", "GNLY"],
    "B-cells": ["MS4A1", "CD19"],
    "NK": ["NKG7", "GNLY"], # GNLY is a marker for two groups
}

# --- Data Generation ---

# 1. Create cell group labels
cell_group_labels = []
cell_ids = []
group_names = list(groups.keys())
group_probs = list(groups.values())

for i in range(n_cells):
    cell_id = f"cell_{i+1:03d}"
    group = np.random.choice(group_names, p=group_probs)
    cell_group_labels.append(group)
    cell_ids.append(cell_id)

# Create the cell_groups DataFrame
cell_groups_df = pd.DataFrame({
    'group': cell_group_labels
}, index=pd.Index(cell_ids, name='cell_id'))


# 2. Create the expression matrix
# Start with a low level of background noise
expression_matrix = np.random.poisson(0.1, size=(n_cells, len(features)))
expression_df = pd.DataFrame(expression_matrix, index=cell_ids, columns=features)

# Add strong marker gene expression
for group, markers in marker_map.items():
    # Get the boolean mask for cells in this group
    is_in_group = (cell_groups_df['group'] == group)
    n_cells_in_group = is_in_group.sum()
    
    for marker in markers:
        if marker in expression_df.columns:
            # Add high expression values for this marker in the correct cells
            high_expression = np.random.poisson(5, size=n_cells_in_group)
            expression_df.loc[is_in_group, marker] += high_expression


# --- Save to CSV ---
expression_csv_path = "expression.csv"
groups_csv_path = "cell_groups.csv"

expression_df.to_csv(expression_csv_path)
cell_groups_df.to_csv(groups_csv_path)

print(f"✅ Success! Created two files:")
print(f"   1. Expression matrix: {expression_csv_path}")
print(f"   2. Cell group labels: {groups_csv_path}")
print("\nYou can now run the CLI command.")