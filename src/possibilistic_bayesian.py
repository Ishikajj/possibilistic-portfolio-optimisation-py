"""
this file is doing something unconventional known as the possibilistic analog
takes the data frame from data input


for each asset, creates it into a separate dataframe with the following properties:
mean (from t = 0 to ti), variance(from t = 0 to ti), and delta, k, scale, degrees of freedom,
inferred optimal time window using a normal inverse wishart distribution of the possibilistic form

we have a burn in period of 100 days where we form a very weak (the average of averages priors with possibility one)

now, for each point of time, we determine a series of "models": each defined by the mean, variance, delta
"""

import data_input


df = data_input.load_industry_portfolios()
df_from_1963 = data_input.slice_timeframe(df)
