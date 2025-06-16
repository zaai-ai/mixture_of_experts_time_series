from cardtale.analytics.testing.card.trend import DifferencingTests
import pandas as pd
import re

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


for idx, (dataset, file_path) in enumerate(results_list.items()):
    print(f"Processing {dataset}...")

    df = pd.read_csv(file_path)
    train_path = re.sub('results', 'train', file_path)
    train = pd.read_csv(train_path)

    features_l = []
    for uid, uid_df in train.groupby('unique_id'):
        try:
            trend = DifferencingTests.ndiffs(uid_df['y'], test='kpss', test_type='level')
        except OverflowError:
            trend = 0

        seas = DifferencingTests.nsdiffs(uid_df['y'], test='seas', period=12)

        trend_str = 'Non-stationary' if trend > 0 else 'Stationary'
        seas_str = 'Seasonal' if seas > 0 else 'Non-seasonal'

        features_l.append({
            'unique_id': uid,
            'trend_str': trend_str,
            'seas_str': seas_str,
        })


    features_df = pd.DataFrame(features_l)
    df = pd.merge(df, features_df, on="unique_id")

    df.to_csv(file_path, index=False)

    
