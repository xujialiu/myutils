import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap


def plot_accuracy_vs_exclusion(
    dict_compare_lines,
    figsize=(10, 6),
    title="Model Accuracy vs Manual Analysis Threshold",
    save_path=None,
):
    """
    Plot multiple models' accuracy vs manual review percentage relationships

    Parameters:
    dict_compare_lines -- List of dictionaries, each containing:
        'name': Series name (str)
        'df': DataFrame containing the data
        'uncertainty_col': Uncertainty column name (str)
        'true_col': Correct prediction indicator column name (str, 1=correct/0=wrong)
        'points': List of special points to annotate (optional, e.g. [0,10,20,50])
        'color': Specific color (optional)

    figsize -- Figure size, default (10,6)
    title -- Chart title
    save_path -- Path to save the figure (optional)
    """
    plt.figure(figsize=figsize)
    colors = get_cmap("tab10").colors

    for idx, line in enumerate(dict_compare_lines):
        # Get parameters for current series
        df = line["df"]
        uncertainty_col = line["uncertainty_col"]
        true_col = line["true_col"]
        name = line["name"]
        points = line.get("points", [])
        color = line.get("color", colors[idx % len(colors)])

        # Prepare data
        df_sorted = df.sort_values(by=uncertainty_col).reset_index(drop=True)
        total_samples = len(df_sorted)
        exclude_percentages = np.linspace(0, 100, 101)
        accuracies = []

        # Calculate accuracy for each exclusion percentage
        for p in exclude_percentages:
            keep_count = int(total_samples * (1 - p / 100))
            if keep_count == 0:
                acc = np.nan
            else:
                acc = df_sorted.iloc[:keep_count][true_col].mean()
            accuracies.append(acc)

        # Plot main curve
        plt.plot(
            exclude_percentages,
            accuracies,
            linewidth=2.5,
            color=color,
            label=name,
        )

        # Annotate special points
        if points:
            point_x = [p for p in points if 0 <= p <= 100]
            point_y = [accuracies[int(p)] for p in point_x]

            plt.scatter(
                point_x,
                point_y,
                s=60,
                color="red",
                zorder=5,
            )

            for x, y in zip(point_x, point_y):
                plt.annotate(
                    f"{y:.3f}",
                    (x, y),
                    xytext=(-10, 10),
                    textcoords="offset points",
                    fontsize=9,
                )

    # Set chart properties
    plt.xlabel("Samples to be analysed manually (%)", fontsize=12, labelpad=10)
    plt.ylabel("Accuracy", fontsize=12, labelpad=10)
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3, linestyle="--")
    sns.despine()
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)
        
if __name__ == "__main__":
    df = pd.read_csv("test_data.csv")
    
    comparison = [
        {
            "name": "predictive_entropy",
            "df": df,
            "uncertainty_col": "predictive_entropy",
            "true_col": "T/F",
            "points": [10, 30, 70],
        },
        {
            "name": "mutual_information",
            "df": df,
            "uncertainty_col": "mutual_information",
            "true_col": "T/F",
            "points": [],
        },
        {
            "name": "pred_class_variance",
            "df": df,
            "uncertainty_col": "pred_class_variance",
            "true_col": "T/F",
            "points": [],
        },
        {
            "name": "pred_class_coef_of_variation",
            "df": df,
            "uncertainty_col": "pred_class_coef_of_variation",
            "true_col": "T/F",
            "points": [],
        },
    ]

    plot_accuracy_vs_exclusion(
        comparison,
        title="Model Comparison: Accuracy vs Manual Review",
    )