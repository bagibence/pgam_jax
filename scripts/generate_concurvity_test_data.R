# Golden-file generator for test_matches_mgcv_golden. Called per scenario by
# scripts/generate_concurvity_test_data.py:
#
#     Rscript scripts/generate_concurvity_test_data.R <scenario> <data.csv> <out.json>
#
# Fits the scenario's mgcv model on the CSV data, runs mgcv::concurvity with
# full=TRUE and full=FALSE, and writes everything the Python test needs
# (model matrix, coefficients, 0-indexed term blocks, both concurvity
# outputs) as JSON. `digits = I(17)` means 17 significant digits, which
# round-trips doubles through text bit-for-bit (`digits = NA`, despite the
# docs saying max precision, gives only 15 and does not).

library(mgcv)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: <scenario> <data.csv> <out.json>")
scenario <- args[1]
csv_path <- args[2]
out_path <- args[3]

specs <- list(
  scenario1 = y ~ s(x1, k = 12) + s(x2, k = 12),
  scenario2 = y ~ s(t, k = 15) + s(x, k = 15),
  scenario3 = y ~ s(x1, k = 10) + s(x2, k = 10) + s(x3, k = 10)
)
if (!scenario %in% names(specs)) stop("unknown scenario: ", scenario)
formula <- specs[[scenario]]

dat <- read.csv(csv_path)
fit <- gam(formula, family = poisson, data = dat, method = "REML")

# The test hard-codes a single-column parametric block ("para" = intercept),
# matching mgcv's own handling. Fail loud if the formula ever grows
# parametric terms.
first_para <- vapply(fit$smooth, function(s) s$first.para, numeric(1))
last_para <- vapply(fit$smooth, function(s) s$last.para, numeric(1))
if (min(first_para) != 2) stop("expected intercept-only parametric block")

labels <- c("para", vapply(fit$smooth, function(s) s$label, character(1)))

conc_full <- concurvity(fit, full = TRUE) # 3 x m: worst, observed, estimate
conc_pair <- concurvity(fit, full = FALSE) # list of m x m matrices
stopifnot(identical(colnames(conc_full), labels))

# 0-indexed inclusive column ranges, para block first.
blocks <- c(
  list(list(label = "para", start = 0L, stop = 0L)),
  lapply(seq_along(fit$smooth), function(i) {
    list(
      label = fit$smooth[[i]]$label,
      start = as.integer(first_para[i] - 1),
      stop = as.integer(last_para[i] - 1)
    )
  })
)

full_measures <- lapply(rownames(conc_full), function(m) unname(conc_full[m, ]))
names(full_measures) <- rownames(conc_full)

out <- list(
  scenario = scenario,
  formula = paste(deparse(formula), collapse = " "),
  family = "poisson",
  mgcv_version = as.character(packageVersion("mgcv")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  labels = labels,
  blocks = blocks,
  X = unname(model.matrix(fit)),
  beta = unname(coef(fit)),
  full = full_measures,
  pairwise = lapply(conc_pair, unname)
)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE), out_path)
