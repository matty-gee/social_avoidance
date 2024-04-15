# ---------------------------
# DIFF QUESTIONNAIRES & ORDERS
# iq: binary
# pid5: ordinal
# pdi: ordinal - mnight need some organization
# ucls: ordinal 
# dt:
# sd
# talc
# pq16:
# sh:
# pss: 
# maq:
# amisos
# mqusf
# eat: ordinal 0:5
# oci: ordinal0:4
# audit: ordinal items 1-8 0:4, 9-10 0,2,4
# aq: ordinal
# sds: ordinal 1:4
# stai: ordinal 1:4
# aes: ordinal 1:4
# sss: binary... how to treat? i think just assume the same...
# https://it.unt.edu/sites/default/files/binaryfa_l_jds_sep2014_0.pdf
# zbpd: ordinal ?
# lsas: ordinal0-3
# apdis:ordinal 0-4
# bapq: ordinal 1-6

#-------------------------------------------------------------------------
# LOAD PACKAGES
# polycor has hetcor, function that produces heterogeneous correlation matrix
# seems it wants everything to be ordinal or numeric...
# https://www.rdocumentation.org/packages/polycor/versions/0.8-0/topics/hetcor

## First specify the packages of interest
# packages = c("polycor")
# ## Now load or install&load all
# package.check <- lapply(
#   packages,
#   FUN = function(x) {
#     if (!require(x, character.only = TRUE)) {
#       install.packages(x, dependencies = TRUE)
#       library(x, character.only = TRUE)
#     }
#   }
# )

library('polycor')
# dir <- "/Volumes/synapse/projects/SocialSpace/Projects/Prolific/hetcor"
dir <- "/mnt/synapse/projects/SocialSpace/Projects/Prolific/hetcor"

# ---------------------------
# load the items
items <- data.frame(read.csv(paste(dir,"/questionnaire_items_n681.csv", sep=""), 
                   header=TRUE, stringsAsFactors=FALSE))
items <- subset (items, select = -sub_id)                  

# ---------------------------
# data typing

items <- sapply(items, as.factor) # ordinal variables... discrete value, with some order

# ---------------------------
# compute a heterogeneous correlation matrix

hetcor_obj <- hetcor(items,
                     use="pairwise.complete.obs", # use as much data as possible
                     ML=TRUE) # MLE

# ---------------------------
# save object & correlation matrix

save(hetcor_obj, file=paste(dir, "/n681_all-items_hetcor_MLE.RData", sep="")) # hetcor returns a het cor object
write.csv(hetcor_obj$correlations, paste(dir, "/n681_all-items_hetcor_MLE.csv", sep=""), row.names=TRUE)
