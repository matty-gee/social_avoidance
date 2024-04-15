library(pwr)
library(effsize)

# two sample cohens d = .58
# assume the effect is a little inflated (maybe really is d = .5)
# but also will try to enhance the effect w/ more trials...

pwr.r.test(power=.95, r=.166, sig.level=0.05, alternative='greater')