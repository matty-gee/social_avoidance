library(pwr)

sig_level = 0.05  # Significance level
power = 0.80
n = NULL

#----------------------------------------------------------------------------
# power analysis on the correlation of social distance to avoidance
#----------------------------------------------------------------------------

r = .1695
result = pwr.r.test(power = power, 
                   r = r,
                   n = n,
                   sig.level = sig_level, 
                   alternative = 'greater')
print(result)

#----------------------------------------------------------------------------
# f2 power analysis
# dont think this makes as much sense...
#----------------------------------------------------------------------------

# n_predictors = 22
# partial_r2 = 0.042 # partial r-squared from multiple ols
# f2 = partial_r2 / (1 - partial_r2) # Convert r2 to f2
# result = pwr.f2.test(u = n_predictors, 
#                      v = NULL, 
#                      f2 = f2, 
#                      sig.level = sig_level, 
#                      power = power)
# print(result)
