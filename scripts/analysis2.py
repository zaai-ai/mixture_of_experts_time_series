import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import partial

from utilsforecast.losses import mase
from modelradar.evaluate.radar import ModelRadar

# Enable pgf backend for LaTeX-friendly output
matplotlib.use("pgf")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

def create_matplotlib_radar_plot(df, dataset_name, ax):
    labels = df.index.tolist()
    models = df.columns.tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for model in models:
        values = df[model].tolist()
        print(values[:1])
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_title(dataset_name.upper(), size=16)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])  # Hide radial labels
    ax.tick_params(labelsize=10)

rmae_func = partial(mase, seasonality=1)

results_list = {
    'm1m': 'results,gluonts,m1_monthly.csv',
    'm1q': 'results,gluonts,m1_quarterly.csv',
    'm1y': 'results,gluonts,m1_yearly.csv',
    'tm': 'results,gluonts,tourism_monthly.csv',
    'tq': 'results,gluonts,tourism_quarterly.csv',
    'ty': 'results,gluonts,tourism_yearly.csv',
    'm3m': 'results,m3,Monthly.csv',
    'm3q': 'results,m3,Quarterly.csv',
    'm3y': 'results,m3,Yearly.csv',
    'm4m': 'results,m4,Monthly.csv',
    'm4q': 'results,m4,Quarterly.csv',
    'm4y': 'results,m4,Yearly.csv',
}

all_results = {}
mase_func = partial(mase, seasonality=1)

fig, axes = plt.subplots(4, 3, figsize=(12, 12), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, (dataset, file_path) in enumerate(results_list.items()):
    print(f"Processing {dataset}...")

    df = pd.read_csv(file_path)
    train_path = re.sub('results', 'train', file_path)
    train = pd.read_csv(train_path)

    radar = ModelRadar(
        cv_df=df,
        metrics=[mase_func],
        model_names=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive'],
        hardness_reference='SeasonalNaive',
        ratios_reference='NBEATS',
        cvar_quantile=0.75,
        hardness_quantile=0.75,
        agg_func='median',
        train_df=train,
        rope=10
    )

    err = radar.evaluate(keep_uids=True)
    radar.uid_accuracy.get_hard_uids(err, return_df=False)

    df_plot = pd.concat([
        radar.evaluate(return_plot=False),
        radar.uid_accuracy.expected_shortfall(err),
        radar.evaluate_by_horizon_bounds(),
        radar.uid_accuracy.accuracy_on_hard(err),
        radar.evaluate_by_group(group_col='anomaly_status'),
    ], axis=1)

    df_plot = df_plot.rank(ascending=False) - 1 # put to false only for the plot
    df_plot = df_plot.T

    create_matplotlib_radar_plot(df_plot, dataset, axes[idx])
    all_results[dataset] = radar.evaluate()

# Global figure settings
fig.suptitle("Aspect-based Evaluation", fontsize=24, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.98])

# Create a proxy artist for legend to get correct colors/labels
models = ['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

proxy_lines = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(models))]
fig.legend(proxy_lines, models, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=14)

# Save figure as PGF
plt.savefig("plots/all_radar_grid.pgf", bbox_inches='tight')

plt.show()

# Summary statistics
r = pd.concat(all_results, axis=1).T

print("\nSummary Statistics")
print("Average Rank:")
print(r.rank(axis=1).mean())
print("Standard Deviation of Rank:")
print(r.rank(axis=1).std())
print("Median Score:")
print(r.median())
print("Mean Score:")
print(r.mean())
