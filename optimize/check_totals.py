import pandas as pd, numpy as np
df = pd.read_csv(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data\dataset.csv')
AVG_D = 106.09; scale = 1.04
poss = np.sqrt(df['h_adjT'] * df['a_adjT'])
pred = (df['h_adjO']*(df['a_adjD']/AVG_D)*poss/100*scale +
        df['a_adjO']*(df['h_adjD']/AVG_D)*poss/100*scale)
print(f'Pred total mean:   {pred.mean():.1f}')
print(f'Actual total mean: {df["actual_total"].mean():.1f}')
print(f'Bias vs actual:    {(pred - df["actual_total"]).mean():+.1f}')
has = df['vegas_total'].notna()
print(f'Bias vs Vegas:     {(pred[has] - df.loc[has,"vegas_total"]).mean():+.1f}  ({has.sum()} games)')
