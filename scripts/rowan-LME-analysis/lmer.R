if ("moments" %in% loadedNamespaces()) detach("package:moments", unload = TRUE, character.only = TRUE)
required_packages <- c(
  "readr", "openxlsx", "reshape2", "lme4", 
  "multcomp", "car", "lmerTest", "emmeans"
)
cran_mirror <- "http://cran.r-project.org"

# Check and install missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE, repos = cran_mirror)
    library(pkg, character.only = TRUE)
  }
}


library(readr)
library(openxlsx)
library(reshape2)
library(lme4)
library(multcomp)
library(car)
library(lmerTest)
library(emmeans)

# Define file paths
file_paths <- list(
  SL = "L:/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data/SL_category.xlsx",
  SI = "L:/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data/SI_category.xlsx",
  BA = "L:/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data/BA_category.xlsx",
  BA_UCB = "L:/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data/BAUCB_category.xlsx"
)
# file_paths <- list(
#   SL = "/media/labs/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data_threefry/SL_category_big.xlsx",
#   SI = "/media/labs/rsmith/lab-members/rhodson/Hill_AI/MATLAB/120trials_data_threefry/SI_category_big.xlsx"
# )

data_list <- list()

# Loop through files and prepare data
for (name in names(file_paths)) {
  # Read the CSV file
  df <- read.xlsx(file_paths[[name]], startRow = 2, colNames = FALSE)
  df[] <- lapply(df, function(x) as.integer(as.character(x)))
  # Melt the data frame into long format
  melted_df <- melt(df, measure.vars = colnames(df), variable.name = "Column", value.name = "value")
  
  # Generate column and row indices
  new_col_indices <- rep(1:ncol(df), each = nrow(df))
  new_row_indices <- rep(1:nrow(df), times = ncol(df))
  
  # Create a new data frame with required columns
  data_list[[name]] <- data.frame(
    Survival = melted_df$value,
    Id = new_col_indices,
    Iteration = new_row_indices,
    Model = name
  )
}

# Combine all data frames into one
combined_df <- do.call(rbind, data_list)
combined_df$Model <- factor(combined_df$Model) # Convert Model to factor


# Fit the Linear Mixed-Effects Model
model <- lmer(Survival ~ Iteration + Model + Iteration * Model + (1 | Id), data = combined_df)

# Summary of the model
summary(model)

# Tukey's Post Hoc Test
test <- glht(model, linfct = mcp(Model = "Tukey"))

# ANOVA for fixed effects
anova_results <- Anova(model, test = 'F')
write.csv(anova_results, file = "anova_big.csv", row.names = FALSE)
# Pairwise comparisons with estimated marginal means
em <- emmeans(model, pairwise ~ Model)
write.csv(em, file = "em_big.csv", row.names = FALSE)

print("Model Summary:")
print(summary(model, model.info = FALSE, model.fit = FALSE, pvals = FALSE))

print("ANOVA Results:")
print(anova_results)

print("Estimated Marginal Means:")
print(em)