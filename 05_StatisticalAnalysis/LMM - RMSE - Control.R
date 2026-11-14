# ==========================================================
# LINEAR MIXED MODEL ANALYSIS: RMSE (version without EXP / IMP)
# Interaction: Group * Hand * Sequence * Session
# Post-hoc tests with estimates, CIs, Cohen's d and CIs of d
# ==========================================================

#Replace the file path in (2)



# 0. Clear the workspace ----------------------------------
rm(list = ls())

# 1. Libraries --------------------------------------------
# install.packages(c("openxlsx", "performance"))
library(readxl)      # Read Excel files
library(lme4)        # Linear mixed models
library(lmerTest)    # p-values for the ANOVA table
library(emmeans)     # Post-hoc tests
library(ggplot2)     # Plots
library(tidyverse)   # Data manipulation
library(performance) # check_distribution
library(openxlsx)    # Excel export

# Raise the observation limits for large datasets
emm_options(pbkrtest.limit = 10000, lmerTest.limit = 20000)

# 2. Load and prepare data --------------------------------
setwd("C:/Users/datafolder")

data <- read_excel("01_RMSE_alldata.xlsx")

# Convert variables to factors
data$ID       <- as.factor(data$ID)
data$Group    <- as.factor(data$Group)
data$Hand     <- as.factor(data$Hand)
data$Sequence <- as.factor(data$Sequence)
data$Session  <- factor(data$Session)

# Outlier exclusion (±2.5 SD within each ID x Session x Hand x Sequence cell)
data <- data %>%
  group_by(ID, Session, Hand, Sequence) %>%
  mutate(
    m  = mean(RMSE, na.rm = TRUE),
    s  = sd(RMSE, na.rm = TRUE),
    is_outlier = RMSE > (m + 2.5 * s) | RMSE < (m - 2.5 * s)
  ) %>%
  filter(is_outlier == FALSE | is.na(is_outlier)) %>%
  ungroup()

# 3. RMSE distribution ------------------------------------
check_distribution(data$RMSE)
qqnorm(data$RMSE)

ggplot(data, aes(x = RMSE)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "RMSE distribution (raw)", x = "RMSE", y = "Frequency")

# Log transformation (non-normality / heteroscedasticity)
data$log_RMSE <- log(data$RMSE)

ggplot(data, aes(x = log_RMSE)) +
  geom_histogram(fill = "seagreen", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "RMSE distribution (log-transformed)", x = "log(RMSE)", y = "Frequency")

check_distribution(data$log_RMSE)
qqnorm(data$log_RMSE)

# Save the cleaned dataset (outliers removed)
write.xlsx(data, "RMSE_withoutoutlier.xlsx")

# 4. Group exclusion --------------------------------------
data <- data %>%
  filter(!(Group %in% c("EXP", "IMP")))
# filter(!(Group %in% c("CTR")))
data <- droplevels(data)

# 5. Linear mixed model -----------------------------------
mod_complet <- lmer(log_RMSE ~ Group * Hand * Sequence * Session + (1 | ID),
                    data = data, REML = TRUE,
                    control = lmerControl(optimizer = "bobyqa"))

# Convergence check
if (is.null(mod_complet@optinfo$conv$lme4$messages)) {
  print("Success: the model converged.")
} else {
  print("Warning: convergence issues.")
}

# 6. ANOVA table ------------------------------------------
print("--- ANOVA TABLE ---")
anova_results <- anova(mod_complet)
print(anova_results)

# 7. Post-hoc function ------------------------------------
# Returns: contrast, estimate (log scale), SE, df, adjusted CI, adjusted p,
#          RMSE ratio (exp) + CI, and Cohen's d + CI (unadjusted)
sig <- sigma(mod_complet)        # Residual SD (excluding (1|ID) variance)
edf <- df.residual(mod_complet)  # Approximate df for the CI of d

posthoc_complet <- function(emm, by_vars, adjust = "bonferroni") {
  
  emm_by <- update(emm, by = by_vars)
  
  # a) Raw differences (log scale) with adjusted CIs and p-values
  contr    <- pairs(emm_by, adjust = adjust)
  tab_diff <- as.data.frame(summary(contr, infer = c(TRUE, TRUE)))
  
  # b) RMSE ratio (back-transformed to the original scale)
  tab_diff$ratio          <- exp(tab_diff$estimate)
  tab_diff$ratio_lower.CL <- exp(tab_diff$lower.CL)
  tab_diff$ratio_upper.CL <- exp(tab_diff$upper.CL)
  
  # c) Cohen's d with CI (computed from the estimated marginal means)
  es     <- eff_size(emm_by, sigma = sig, edf = edf, method = "pairwise")
  tab_es <- as.data.frame(summary(es, infer = c(TRUE, TRUE)))
  
  tab_diff$d          <- tab_es$effect.size
  tab_diff$d_lower.CL <- tab_es$lower.CL
  tab_diff$d_upper.CL <- tab_es$upper.CL
  
  tab_diff
}

# 8. Post-hocs: Group x Session x Sequence ----------------
emm_total <- emmeans(mod_complet, ~ Group * Session * Sequence)

cat("\n--- POST-HOC: GROUP COMPARISONS ---\n")
ph_groups <- posthoc_complet(emm_total, c("Session", "Sequence"))
print(ph_groups)

cat("\n--- POST-HOC: SESSION COMPARISONS ---\n")
ph_sessions <- posthoc_complet(emm_total, c("Group", "Sequence"))
print(ph_sessions)

cat("\n--- POST-HOC: SEQUENCE COMPARISONS ---\n")
ph_seq <- posthoc_complet(emm_total, c("Group", "Session"))
print(ph_seq)

# 9. Excel export -----------------------------------------
wb <- createWorkbook()

# Helper: add a worksheet and write a data frame into it
add_sheet <- function(name, df) {
  addWorksheet(wb, name)
  writeData(wb, name, df)
}

add_sheet("ANOVA",             as.data.frame(anova_results))
add_sheet("Emm GxSxS",         as.data.frame(summary(emm_total, infer = c(TRUE, TRUE))))
add_sheet("PostHoc_Groupes",   ph_groups)
add_sheet("PostHoc_Sessions",  ph_sessions)
add_sheet("PostHoc_Sequences", ph_seq)

saveWorkbook(wb, file = "Results_ControlAnalyses.xlsx", overwrite = TRUE)

cat("Excel file successfully created!\n")