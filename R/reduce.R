## Reducers: what to do with the values of each block's cells.
##
## A reducer is a list of three functions: start(plan), add(cells) and
## result(). `cells` is a list of equal-length vectors for one block:
## id, w, run, row, col, value (NA for nodata).

## one value per id (points): out[id] <- value; ids with no cell are NA
pix_place <- function() {
  out <- NULL
  list(start = function(plan) out <<- rep(NA_real_, attr(plan, "n_id")),
       add = function(cells) out[cells$id] <<- cells$value,
       result = function() out)
}

## every cell with its value, in plan order
pix_cells <- function() {
  parts <- list()
  list(start = function(plan) parts <<- list(),
       add = function(cells) parts[[length(parts) + 1L]] <<- cells,
       result = function() {
         keys <- c("row", "col", "id", "w", "run", "value")
         as.data.frame(lapply(stats::setNames(keys, keys),
                              function(k) unlist(lapply(parts, `[[`, k), use.names = FALSE)))
       })
}

## grouped weighted statistics by id, NA values excluded
## count: valid cells; weight: sum of w; sum: sum of w * value;
## mean: sum / weight; min, max: unweighted
pix_stats <- function() {
  n <- 0L
  count <- weight <- total <- mn <- mx <- NULL
  list(
    start = function(plan) {
      n <<- attr(plan, "n_id")
      count <<- integer(n); weight <<- total <<- numeric(n)
      mn <<- rep(Inf, n); mx <<- rep(-Inf, n)
    },
    add = function(cells) {
      ok <- !is.na(cells$value)
      if (!any(ok)) return(invisible())
      i <- cells$id[ok]; w <- cells$w[ok]; v <- cells$value[ok]
      count <<- count + tabulate(i, n)
      s <- rowsum(cbind(w, w * v), i, reorder = FALSE)
      u <- as.integer(rownames(s))
      weight[u] <<- weight[u] + s[, 1L]
      total[u] <<- total[u] + s[, 2L]
      o <- order(i, v)
      i <- i[o]; v <- v[o]
      lo <- !duplicated(i); hi <- !duplicated(i, fromLast = TRUE)
      mn[i[lo]] <<- pmin(mn[i[lo]], v[lo])
      mx[i[hi]] <<- pmax(mx[i[hi]], v[hi])
    },
    result = function() {
      empty <- count == 0L
      mean <- total / weight
      mean[empty] <- NA; mn[empty] <- NA; mx[empty] <- NA
      data.frame(id = seq_len(n), count = count, weight = weight, sum = total,
                 mean = mean, min = mn, max = mx)
    })
}
