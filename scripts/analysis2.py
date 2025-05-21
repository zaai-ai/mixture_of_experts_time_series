import re
import pandas as pd
from datasetsforecast.m4 import M4
from utilsforecast.losses import smape, mape, rmae, mae, mase, rmse, rmsse
from functools import partial

rmae_func = partial(rmae, baseline='SeasonalNaive')

# from utils import load_dataset, train_test_split
from modelradar.evaluate.radar import ModelRadar
from modelradar.visuals.plotter import ModelRadarPlotter, SpiderPlot
import plotnine as p9
import matplotlib.pyplot as plt


# dataset='gluonts'
# group='m1_monthly'
# m1_yearly
# m1_quarterly
# tourism_monthly
# tourism_quarterly
# tourism_yearly
# df, horizon, n_lags, freq_str, freq_int = load_dataset(f'{dataset}_{group}', {})
# train, _ = train_test_split(df, horizon)


results_list = {
    'm1m': 'results,gluonts,m1_monthly.csv',
    'm1q': 'results,gluonts,m1_quarterly.csv',
    'm1y': 'results,gluonts,m1_yearly.csv',
    'tm': 'results,gluonts,tourism_monthly.csv',
    'tq': 'results,gluonts,tourism_quarterly.csv',
    'm3m': 'results,m3,Monthly.csv',
    'm3q': 'results,m3,Quarterly.csv',
    'm4m': 'results,m4,Monthly.csv',
    'm4q': 'results,m4,Quarterly.csv',
    'm4y': 'results,m4,Yearly.csv',
}

mase_func = partial(mase, seasonality=1)

all_results = {}
plots = {}
for dataset, file_path in results_list.items():
    print(dataset)
    df = pd.read_csv(file_path)
    train = pd.read_csv(re.sub('results','train', file_path))

    radar = ModelRadar(cv_df=df,
                       # metrics=[smape],
                       # metrics=[rmae_func],
                       metrics=[mase_func],
                       model_names=['NBeatsMoe', 'NBEATS', 'NBeatsStackMoe', 'SeasonalNaive'],
                       hardness_reference='SeasonalNaive',
                       ratios_reference='NBEATS',
                       cvar_quantile=0.75,
                       hardness_quantile=0.75,
                       agg_func='median',
                       train_df=train,
                       rope=10)

    err = radar.evaluate(keep_uids=True)
    radar.uid_accuracy.get_hard_uids(err, return_df=False)
    hard_uid_list = radar.uid_accuracy.hard_uid

    err_hard = err.loc[hard_uid_list, :]
    # err_anomalies = radar.evaluate_by_anomaly(anomaly_col='is_anomaly', mode='observations')

    es_err = radar.uid_accuracy.expected_shortfall(err)
    es_errh = radar.uid_accuracy.expected_shortfall(err_hard)
    wdl = radar.rope.get_winning_ratios(err)

    print('Overall scores')
    all_results[dataset] = radar.evaluate()
    # all_results[dataset] = es_err
    # all_results[dataset] = err_hard.mean()
    # all_results[dataset] = es_errh
    # all_results[dataset] = err_anomalies.mean()

    err_hard = radar.uid_accuracy.get_hard_uids(err)

    df_plot = pd.concat([radar.evaluate(return_plot=False),
                radar.uid_accuracy.expected_shortfall(err),
                radar.evaluate_by_horizon_bounds(),
                radar.uid_accuracy.accuracy_on_hard(err),
                radar.evaluate_by_group(group_col='anomaly_status')
                #error_on_trend,
                #error_on_seas
                    ], axis=1)

    df_plot.head()

    plot = SpiderPlot.create_plot(df=df_plot, values='rank', include_title=False)
    plot = plot + p9.ggtitle(f"Spider Plot for {dataset.upper()}") + p9.theme(
            plot_margin=0.05,
            figure_size=(12, 12),
            legend_position='top',
            strip_text=p9.element_text(size=17, family='cmtt10'),
            plot_title=p9.element_text(size=20, family='cmtt10'),
            panel_grid=p9.element_blank(),
            legend_text=p9.element_text(size=17, family='cmtt10'),
            legend_key_size=20,
            legend_key_width=20,
            text=p9.element_text(size=17, family='cmtt10'),

            # These lines remove x/y axis elements
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            axis_text_x=p9.element_blank(),
            axis_text_y=p9.element_blank(),
            axis_ticks=p9.element_blank(),
            axis_line=p9.element_blank()
        )
    
    plots[dataset] = plot
    plot.save(f'plots/{dataset}.png', dpi=300, limitsize=False)


# put all plots in a grid


r = pd.concat(all_results, axis=1).T

print(r)
# r = r.drop('tq')
# print(r.drop('tq').rank(axis=1).mean())
# print(r.rank(axis=1).mean())
print('Average rank across datasets')
print(r.rank(axis=1).mean())
print(r.rank(axis=1).std())
print('Median score across datasets')
print(r.median())
print('Mean score across datasets')
print(r.mean())
# print(r.drop('tq').mean())
