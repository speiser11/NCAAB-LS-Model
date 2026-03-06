import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data\dataset.csv')
AVG_D = 106.09; scale = 1.04
poss = np.sqrt(df['h_adjT'] * df['a_adjT'])
raw = (df['h_adjO']*(df['a_adjD']/AVG_D)*poss/100*scale +
       df['a_adjO']*(df['h_adjD']/AVG_D)*poss/100*scale)

# Only use games with Vegas total lines
has = df['vegas_total'].notna()
raw_v = raw[has].values.reshape(-1,1)
veg_v = df.loc[has,'vegas_total'].values

reg = LinearRegression().fit(raw_v, veg_v)
print(f'Linear fit: vegas_total = {reg.coef_[0]:.4f} * rawTotal + {reg.intercept_:.2f}')
print(f'  → totalScale  = {reg.coef_[0]:.4f}')
print(f'  → totalOffset = {reg.intercept_:.2f}')
print(f'  R²: {reg.score(raw_v, veg_v):.4f}')

# Show bias by bucket
print('\nBias by raw total bucket:')
buckets = [(130,145),(145,155),(155,165),(165,180)]
for lo,hi in buckets:
    mask = (raw[has]>=lo)&(raw[has]<hi)
    if mask.sum() < 10: continue
    bias = (raw[has][mask] - df.loc[has,'vegas_total'][mask]).mean()
    print(f'  raw {lo}-{hi}: n={mask.sum():4d}  avg bias={bias:+.1f}')

# Preview correction for Drake/Belmont-style game
print('\nDrake/Belmont example:')
raw_db = 168.8
adj = reg.coef_[0] * raw_db + reg.intercept_
print(f'  raw={raw_db}  → adjusted={adj:.1f}  (Vegas 152)')
