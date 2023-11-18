library(tidyverse)
theme_set(theme_classic())

#################### Function definitions ####################
print_variables = function(message, variables, items_per_line = 1) {
  insert_newlines <- function(x, n) {
    groups = split(x, ceiling(seq_along(x) / n))
    
    sapply(groups, function(g) paste(g, collapse = ", ")) %>%
      paste(collapse = "\n")
  }
  
  formatted_variables = insert_newlines(variables, items_per_line)
  
  sprintf("%s:\n%s", message, formatted_variables) %>%
    cat
}

get_pairwise_correlations = function(df) {
  # Selects numeric inputs, gets the aboslute correlation matrix, subsets to unique but
  # distinct pairs and outputs as a long tibble
  df %>%
    select(where(is.numeric)) %>%
    cor %>%
    {
      matrix = .
      diag(matrix) = NA
      matrix[upper.tri(matrix)] = NA
      abs(matrix)
    } %>%
    {
      data = as_tibble(.)
      data$row = rownames(.)
      data
    } %>%
    pivot_longer(
      cols = where(is.numeric),
      names_to = "column",
      values_to = "abs_cor"
    ) %>%
    filter(!is.na(abs_cor)) %>%
    arrange(desc(abs_cor))
}