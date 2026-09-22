## Load the pixtract R functions: source("R/pixtract.R") from the repo root,
## or source(file.path(<repo>, "R", "pixtract.R")) from anywhere.
## Needs gdalraster and xml2 (xml2 is already a gdalraster dependency).

local({
  here <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  if (is.null(here) || !file.exists(file.path(here, "plan.R"))) here <- "R"
  for (f in c("plan.R", "cells.R", "reduce.R", "execute.R", "extract.R"))
    sys.source(file.path(here, f), envir = globalenv())
})
