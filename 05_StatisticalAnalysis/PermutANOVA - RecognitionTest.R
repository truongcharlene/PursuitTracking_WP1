# =========================================================
# ANALYSIS OF THE RECOGNITION TEST
# Permutation ANOVA: Grp (between) x Temps (within)
# Post-hoc Wilcoxon / Mann-Whitney tests with
# rank-biserial correlations and 95% bootstrap CIs
# =========================================================
#This script is an example using the visual recognition test data of the main experiment.
#Replace the file path in (2)
#for the visuo-motor recognition test, change the variables "V.." by "VM.." in (3) and (4)
#for the control experiment, change the sheet with the control experiment data ' Recognition Control' in (2)

# =========================================================
# 1. Packages
# =========================================================

# Run ONCE if needed:
# install.packages(c("readxl", "tidyr", "dplyr",
#                    "permuco", "rstatix", "openxlsx"))

library(readxl)
library(tidyr)
library(dplyr)
library(permuco)
library(rstatix)
library(openxlsx)


# =========================================================
# 2. Data import
# =========================================================
# Replace with your own file path
data <- read_excel(
  "C:\Users\...\02_Recognition_data.xlsx",
  sheet = "Recognition main" #change with the Recognition Control sheet for the control experiment
  )


# =========================================================
# 3. Reshape to long format
# =========================================================

data_long <- pivot_longer(
  data,
  cols = c(`V_sequence`, `V_mirror`, `V_random`), #change with 'VM...' for the visuo-motor condition
  names_to = "Temps",
  values_to = "Score"
)


# =========================================================
# 4. Variable conversion
# =========================================================

data_long$Grp <- factor(data_long$Grp)

data_long$Temps <- factor(
  data_long$Temps,
  levels = c("V_sequence", "V_mirror", "V_random") #change with 'VM' for the visuo-motor condition
)

data_long$ID <- factor(data_long$ID)


# =========================================================
# 5. PERMUTATION ANOVA
# =========================================================

# Mixed design: Grp (between-subject) x Temps (within-subject)
result <- aovperm(
  Score ~ Grp * Temps + Error(ID / Temps),
  data = data_long,
  np = 10000
)

summary(result)


# =========================================================
# 6. PARTIAL ETA SQUARED
# =========================================================

tab <- result$table

# Partial eta squared computed from F and degrees of freedom
tab$eta_p2 <- with(
  tab,
  (F * dfn) / (F * dfn + dfd)
)

anova_export <- tab %>%
  select(F, dfn, dfd, eta_p2)

print(anova_export)


# =========================================================
# 7. FUNCTION: RANK-BISERIAL CORRELATION (r_rb)
#    + 95% BOOTSTRAP CI
# =========================================================

rank_biserial_bootstrap <- function(
    x,
    y,
    paired = FALSE,
    nboot = 5000,
    conf = 0.95
) {
  
  # -------------------------------------------------------
  # Keep complete cases only
  # -------------------------------------------------------
  
  if (paired) {
    
    complete <- complete.cases(x, y)
    x <- x[complete]
    y <- y[complete]
    
    n <- length(x)
    
    if (n < 2) {
      return(
        data.frame(
          effsize = NA,
          CI_low = NA,
          CI_high = NA
        )
      )
    }
    
  } else {
    
    x <- x[!is.na(x)]
    y <- y[!is.na(y)]
    
    nx <- length(x)
    ny <- length(y)
    
    if (nx < 2 || ny < 2) {
      return(
        data.frame(
          effsize = NA,
          CI_low = NA,
          CI_high = NA
        )
      )
    }
  }
  
  
  # -------------------------------------------------------
  # r_rb computation
  # -------------------------------------------------------
  
  calc_r_rb <- function(x, y, paired) {
    
    if (paired) {
      
      d <- x - y
      
      # Wilcoxon signed-rank statistic (sum of positive ranks, zeros dropped)
      W <- sum(rank(abs(d[d != 0]))[d[d != 0] > 0])
      
      n_nonzero <- sum(d != 0)
      
      if (n_nonzero == 0) {
        return(0)
      }
      
      # Maximum possible rank sum
      max_W <- n_nonzero * (n_nonzero + 1) / 2
      
      # Rank-biserial correlation
      r_rb <- (2 * W - max_W) / max_W
      
      return(r_rb)
      
    } else {
      
      # Mann-Whitney U statistic
      test <- wilcox.test(
        x,
        y,
        exact = FALSE
      )
      
      U <- as.numeric(test$statistic)
      
      nx <- length(x)
      ny <- length(y)
      
      # Rank-biserial correlation
      r_rb <- (2 * U) / (nx * ny) - 1
      
      return(r_rb)
    }
  }
  
  
  # -------------------------------------------------------
  # Observed r_rb
  # -------------------------------------------------------
  
  observed <- calc_r_rb(x, y, paired)
  
  
  # -------------------------------------------------------
  # Bootstrap
  # -------------------------------------------------------
  
  boot_values <- numeric(nboot)
  
  for (i in seq_len(nboot)) {
    
    if (paired) {
      
      # Resample pairs (keeps within-subject pairing)
      index <- sample(
        seq_along(x),
        size = length(x),
        replace = TRUE
      )
      
      boot_x <- x[index]
      boot_y <- y[index]
      
    } else {
      
      # Resample each group independently
      boot_x <- sample(
        x,
        size = length(x),
        replace = TRUE
      )
      
      boot_y <- sample(
        y,
        size = length(y),
        replace = TRUE
      )
    }
    
    boot_values[i] <- calc_r_rb(
      boot_x,
      boot_y,
      paired
    )
  }
  
  
  # -------------------------------------------------------
  # Percentile CI
  # -------------------------------------------------------
  
  alpha <- 1 - conf
  
  CI <- quantile(
    boot_values,
    probs = c(alpha / 2, 1 - alpha / 2),
    na.rm = TRUE
  )
  
  
  return(
    data.frame(
      effsize = observed,
      CI_low = as.numeric(CI[1]),
      CI_high = as.numeric(CI[2])
    )
  )
}


# =========================================================
# 8. POST-HOC: GROUPS WITHIN EACH TIME POINT
# =========================================================

# Mann-Whitney tests (unpaired), Bonferroni-adjusted
group_posthoc <-
  data_long %>%
  group_by(Temps) %>%
  pairwise_wilcox_test(
    Score ~ Grp,
    paired = FALSE,
    p.adjust.method = "bonferroni"
  )

print(group_posthoc)


# =========================================================
# 9. r_rb + 95% CI: GROUPS WITHIN EACH TIME POINT
# =========================================================

# List all pairwise group comparisons for each time point
group_comparisons <- data_long %>%
  group_by(Temps) %>%
  summarise(
    comparisons = list(
      combn(
        levels(droplevels(Grp)),
        2,
        simplify = FALSE
      )
    ),
    .groups = "drop"
  )


group_effectsize <- data.frame()

for (i in seq_len(nrow(group_comparisons))) {
  
  temps_i <- group_comparisons$Temps[i]
  comparisons_i <- group_comparisons$comparisons[[i]]
  
  data_i <- data_long %>%
    filter(Temps == temps_i)
  
  for (comp in comparisons_i) {
    
    g1 <- comp[1]
    g2 <- comp[2]
    
    x <- data_i %>%
      filter(Grp == g1) %>%
      pull(Score)
    
    y <- data_i %>%
      filter(Grp == g2) %>%
      pull(Score)
    
    effect <- rank_biserial_bootstrap(
      x,
      y,
      paired = FALSE,
      nboot = 5000
    )
    
    group_effectsize <- rbind(
      group_effectsize,
      data.frame(
        Temps = as.character(temps_i),
        group1 = g1,
        group2 = g2,
        effsize = effect$effsize,
        CI_low = effect$CI_low,
        CI_high = effect$CI_high
      )
    )
  }
}


# =========================================================
# 10. POST-HOC: TIME POINTS WITHIN EACH GROUP
# =========================================================

# Wilcoxon signed-rank tests (paired), Bonferroni-adjusted
time_posthoc <-
  data_long %>%
  group_by(Grp) %>%
  pairwise_wilcox_test(
    Score ~ Temps,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

print(time_posthoc)


# =========================================================
# 11. r_rb + 95% CI: TIME POINTS WITHIN EACH GROUP
# =========================================================

# List all pairwise time comparisons for each group
time_comparisons <- data_long %>%
  group_by(Grp) %>%
  summarise(
    comparisons = list(
      combn(
        levels(droplevels(Temps)),
        2,
        simplify = FALSE
      )
    ),
    .groups = "drop"
  )


time_effectsize <- data.frame()

for (i in seq_len(nrow(time_comparisons))) {
  
  grp_i <- time_comparisons$Grp[i]
  comparisons_i <- time_comparisons$comparisons[[i]]
  
  data_i <- data_long %>%
    filter(Grp == grp_i)
  
  for (comp in comparisons_i) {
    
    t1 <- comp[1]
    t2 <- comp[2]
    
    # Wide format to pair observations by participant
    temp_data <- data_i %>%
      select(ID, Temps, Score) %>%
      filter(Temps %in% c(t1, t2)) %>%
      pivot_wider(
        names_from = Temps,
        values_from = Score
      ) %>%
      filter(
        !is.na(.data[[t1]]),
        !is.na(.data[[t2]])
      )
    
    x <- temp_data[[t1]]
    y <- temp_data[[t2]]
    
    effect <- rank_biserial_bootstrap(
      x,
      y,
      paired = TRUE,
      nboot = 5000
    )
    
    time_effectsize <- rbind(
      time_effectsize,
      data.frame(
        Grp = as.character(grp_i),
        group1 = t1,
        group2 = t2,
        effsize = effect$effsize,
        CI_low = effect$CI_low,
        CI_high = effect$CI_high
      )
    )
  }
}


# =========================================================
# 12. TIME POINTS, ALL GROUPS POOLED
# =========================================================

group_ctr <-
  data_long %>%
  pairwise_wilcox_test(
    Score ~ Temps,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

print(group_ctr)


# =========================================================
# 13. r_rb + 95% CI: TIME POINTS, ALL GROUPS POOLED
# =========================================================

temp_levels <- levels(data_long$Temps)

ctr_comparisons <- combn(
  temp_levels,
  2,
  simplify = FALSE
)

group_ctr_effectsize <- data.frame()

for (comp in ctr_comparisons) {
  
  t1 <- comp[1]
  t2 <- comp[2]
  
  # Wide format to pair observations by participant
  temp_data <- data_long %>%
    select(ID, Temps, Score) %>%
    filter(Temps %in% c(t1, t2)) %>%
    pivot_wider(
      names_from = Temps,
      values_from = Score
    ) %>%
    filter(
      !is.na(.data[[t1]]),
      !is.na(.data[[t2]])
    )
  
  x <- temp_data[[t1]]
  y <- temp_data[[t2]]
  
  effect <- rank_biserial_bootstrap(
    x,
    y,
    paired = TRUE,
    nboot = 5000
  )
  
  group_ctr_effectsize <- rbind(
    group_ctr_effectsize,
    data.frame(
      group1 = t1,
      group2 = t2,
      effsize = effect$effsize,
      CI_low = effect$CI_low,
      CI_high = effect$CI_high
    )
  )
}


# =========================================================
# 14. MERGE RESULTS (tests + effect sizes)
# =========================================================

group_results <- group_posthoc %>%
  left_join(
    group_effectsize,
    by = c("Temps", "group1", "group2")
  )


time_results <- time_posthoc %>%
  left_join(
    time_effectsize,
    by = c("Grp", "group1", "group2")
  )


group_ctr_results <- group_ctr %>%
  left_join(
    group_ctr_effectsize,
    by = c("group1", "group2")
  )


# =========================================================
# 15. ROUNDING
# =========================================================

group_results <- group_results %>%
  mutate(
    effsize = round(effsize, 3),
    CI_low = round(CI_low, 3),
    CI_high = round(CI_high, 3)
  )


time_results <- time_results %>%
  mutate(
    effsize = round(effsize, 3),
    CI_low = round(CI_low, 3),
    CI_high = round(CI_high, 3)
  )


group_ctr_results <- group_ctr_results %>%
  mutate(
    effsize = round(effsize, 3),
    CI_low = round(CI_low, 3),
    CI_high = round(CI_high, 3)
  )


anova_export <- anova_export %>%
  mutate(
    eta_p2 = round(eta_p2, 3)
  )


# =========================================================
# 16. DISPLAY
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("ANOVA + PARTIAL ETA SQUARED\n")
cat("=========================================================\n")

print(anova_export)


cat("\n")
cat("=========================================================\n")
cat("GROUPS WITHIN EACH TIME POINT\n")
cat("=========================================================\n")

print(
  group_results %>%
    select(
      Temps,
      group1,
      group2,
      statistic,
      p,
      p.adj,
      p.adj.signif,
      effsize,
      CI_low,
      CI_high
    )
)


cat("\n")
cat("=========================================================\n")
cat("TIME POINTS WITHIN EACH GROUP\n")
cat("=========================================================\n")

print(
  time_results %>%
    select(
      Grp,
      group1,
      group2,
      statistic,
      p,
      p.adj,
      p.adj.signif,
      effsize,
      CI_low,
      CI_high
    )
)


cat("\n")
cat("=========================================================\n")
cat("TIME POINTS, ALL GROUPS POOLED\n")
cat("=========================================================\n")

print(
  group_ctr_results %>%
    select(
      group1,
      group2,
      statistic,
      p,
      p.adj,
      p.adj.signif,
      effsize,
      CI_low,
      CI_high
    )
)


# =========================================================
# 17. EXCEL EXPORT
# =========================================================

wb <- createWorkbook()


# ---------------------------------------------------------
# Sheet 1: ANOVA + partial eta squared
# ---------------------------------------------------------

addWorksheet(wb, "ANOVA_eta_p2")

writeData(
  wb,
  "ANOVA_eta_p2",
  anova_export
)


# ---------------------------------------------------------
# Sheet 2: Post-hoc, groups
# ---------------------------------------------------------

addWorksheet(wb, "Posthoc_Groupes")

writeData(
  wb,
  "Posthoc_Groupes",
  group_results
)


# ---------------------------------------------------------
# Sheet 3: Post-hoc, time points
# ---------------------------------------------------------

addWorksheet(wb, "Posthoc_Temps")

writeData(
  wb,
  "Posthoc_Temps",
  time_results
)


# ---------------------------------------------------------
# Sheet 4: Post-hoc, time points (all groups pooled)
# ---------------------------------------------------------

addWorksheet(wb, "Posthoc_Tous_Temps")

writeData(
  wb,
  "Posthoc_Tous_Temps",
  group_ctr_results
)


# ---------------------------------------------------------
# Sheet 5: Effect sizes, groups
# ---------------------------------------------------------

addWorksheet(wb, "EffectSize_Groupes")

writeData(
  wb,
  "EffectSize_Groupes",
  group_effectsize
)


# ---------------------------------------------------------
# Sheet 6: Effect sizes, time points
# ---------------------------------------------------------

addWorksheet(wb, "EffectSize_Temps")

writeData(
  wb,
  "EffectSize_Temps",
  time_effectsize
)


# ---------------------------------------------------------
# Sheet 7: Effect sizes, time points (all groups pooled)
# ---------------------------------------------------------

addWorksheet(wb, "EffectSize_Tous_Temps")

writeData(
  wb,
  "EffectSize_Tous_Temps",
  group_ctr_effectsize
)


# =========================================================
# 18. EXCEL FORMATTING
# =========================================================

header_style <- createStyle(
  textDecoration = "bold",
  halign = "center",
  border = "Bottom"
)

for (sheet in names(wb)) {
  
  # Header style
  addStyle(
    wb,
    sheet = sheet,
    style = header_style,
    rows = 1,
    cols = 1:20,
    gridExpand = TRUE
  )
  
  # Automatic column width
  setColWidths(
    wb,
    sheet = sheet,
    cols = 1:20,
    widths = "auto"
  )
  
  # Freeze the first row
  freezePane(
    wb,
    sheet = sheet,
    firstRow = TRUE
  )
}


# =========================================================
# 19. SAVE
# =========================================================

output_file <-
  "C:/Users/.../Resultats_Permut.xlsx"

saveWorkbook(
  wb,
  file = output_file,
  overwrite = TRUE
)


cat("\n")
cat("=========================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=========================================================\n")
cat("Excel file:\n")
cat(output_file)
cat("\n")
cat("=========================================================\n")