#!/usr/bin/env Rscript

# Updated set of samples---i.e. model configurations---based on executed model configurations and their resulting output
# data, which is post-processed to be represented by a single value.
# Author: Gijs G. Hendrickx

# load and/or install tgp-package
# install.packages("tgp")
library(tgp)

# read input arguments
args <- commandArgs(TRUE)
# > file w/ post-processed data
sol_file <- args[1]
# > file w/ optional samples
opt_file <- args[2]
# > file with plots
plot_file <- args[3]
# > output variable to optimise for
output_var = args[4]

# load data
sol <- read.csv(sol_file)
opt <- read.csv(opt_file)

# normalise data
# mins <- apply(opt, 2, min)
# maxs <- apply(opt, 2, max)

# for(i in 1:ncol(opt)) {
#     sol[, i] <- (sol[, i] - mins[i]) / (maxs[i] - mins[i])
#     opt[, i] <- (opt[, i] - mins[i]) / (maxs[i] - mins[i])
# }

# separate input and output values
X <- sol[, 2:ncol(opt)] # changed 1 to 2
Z <- sol[output_var]
XX <- opt[, -1] #remove X column

# fit TGP-LLM: treed Gaussian Process, Linearly Limited Model
fit <- btgpllm(X = X, Z = Z, XX=XX, pred.n = TRUE, R = 1, verb = 0) # remove XX if using .design function

# save model
# save(fit, file = save_file)
pdf(plot_file, width = 10, height = 5)  # Save plot as PDF file
plot(fit, main='treed GP,', layout = 'both')
dev.off()       # Close the device

# create D-optimal candidates
# cand <- tgp.design(howmany = strtoi(howmany), Xcand = XX, out = fit) # changed opt to XX

# # invert normalisation: de-normalise
# # for(i in 1:ncol(opt)) {
# #     cand[, i] <- cand[, i] * (maxs[i] - mins[i]) + mins[i]
# # }

# set data to data-frame: include variable names
# cand <- setNames(data.frame(cand), colnames(XX)) # changed colnames from opt to XX
# export data: overwrite previous samples-file
# write.table(
#     cand, out_file, sep = ",", row.names = FALSE, col.names = !append_samples, append = append_samples
# )
