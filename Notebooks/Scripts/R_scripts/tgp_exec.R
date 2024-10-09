# Author: Lucas Terlinden-Ruhl, based on script from Gijs G. Hendrickx

# install.packages("tgp")
library(tgp)

# read input arguments
args <- commandArgs(TRUE)
# > file w/ post-processed data
sol_file <- args[1]
# > file w/ optional samples
opt_file <- args[2]
# > file w/ acquisition
acqui_file <- args[3]
# > file with predictive means at non sampled locations
X_file <- args[4]
# > file with predictive means at sampled locations
XX_file <- args[5]
# > file with plots
plot_file <- args[6]
# > output variable to optimise for
output_var = args[7]

X5_file <- args[8]
X95_file <- args[9]

XX5_file <- args[10]
XX95_file <- args[11]

prior = args[12]

# load data
sol <- read.csv(sol_file)
opt <- read.csv(opt_file)

# separate input and output values
X <- sol[, 2:ncol(opt)] # changed 1 to 2
Z <- sol[output_var]
XX <- opt[, -1] #remove X column

# set.seed(5) # use for consistent results
# fit TGP-LLM: treed Gaussian Process, Linearly Limited Model
fit <- btgpllm(X = X, Z = Z, XX=XX, bprior = prior, pred.n = TRUE, verb = 0) # better estimates: BTE = c(2000, 10000, 2), R = 5
write.csv(fit$ZZ.q, file = acqui_file, row.names = FALSE) # pred.n: ZZ.q, Ds2x: Ds2x, improv: improv
write.csv(fit$ZZ.mean, file = XX_file, row.names = FALSE)
write.csv(fit$Zp.mean, file = X_file, row.names = FALSE)
write.csv(fit$Zp.q1, file = X5_file, row.names = FALSE)
write.csv(fit$Zp.q2, file = X95_file, row.names = FALSE)
write.csv(fit$ZZ.q1, file = XX5_file, row.names = FALSE)
write.csv(fit$ZZ.q2, file = XX95_file, row.names = FALSE)

pdf(plot_file, width = 10, height = 5)  # Save plot as PDF file
plot(fit, main='treed GP,', layout = 'both')
dev.off()       # Close the device

tree_plot <- gsub("\\.pdf", "_2.pdf", plot_file)
pdf(tree_plot, width = 10, height = 5)
tgp.trees(fit)
dev.off()
