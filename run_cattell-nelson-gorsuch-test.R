library(nFactors)
eigvals <- read.csv("/Users/matty_gee/Desktop/Behavior/Online/Prolific/Data/Summary/pca_eigenvalues_217items_011322.csv", 
                    header=TRUE, stringsAsFactors=FALSE)
nCng(eigvals$X0, details=TRUE)