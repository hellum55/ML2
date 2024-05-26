######################################################################################################
# Multivariate Adaptive Regression Splines (MARS)
# Reference: Ch.7, HOM with R
# Trevor Hastie, Stephen Milborrow. Derived from mda:mars by 
# Trevor Hastie and Rob Tibshirani. Uses Alan Miller’s Fortran utilities with 
# Thomas Lumley’s leaps wrapper. 2019. Earth: Multivariate Adaptive Regression Splines. 
# https://CRAN.R-project.org/package=earth.
# Advantages: automating tuning process, generalized cross-validation procedure
# automatic (GCV), automated feature selection, interaction effects and feature importance 
# Last update: 30 Jan 2023
######################################################################################################

# Case description: we use the ames data (see description in ch.1) 
# problem type: supervised regression
# response variable: Sale_Price (i.e., $195,000, $215,000)
# features: 80
# observations: 2,930
# objective: use property attributes to predict the sale price of a house
# access: provided by the AmesHousing package (Kuhn 2017a)
# more details: See ?AmesHousing::ames_raw

# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome plotting
library(rsample)     # for sample split

# Modeling packages
library(earth)     # for fitting MARS models
library(caret)     # for automating the tuning process

# Model interpretability packages
library(vip)       # for variable importance
library(pdp)       # for variable relationships


# Stratified sampling with the rsample package
ames <- AmesHousing::make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)


# Fit a basic MARS model
mars1 <- earth(
  Sale_Price ~ .,  
  data = ames_train   
)

# Print model summary
print(mars1)
# Discuss output
# at this stage pruning is based only on an approximation of cv model 
# performance on the training data (based on an expected change 
# in R-sq of less than 0.001)

# check out the first 10 coefficient terms
summary(mars1) %>% .$coefficients %>% head(10)

# plot
plot(mars1, which = 1)
# Discuss output




# Fit a basic MARS model with interactions (see degree = 2)
mars2 <- earth(
  Sale_Price ~ .,  
  data = ames_train,
  degree = 2
)

# check out the first 10 coefficient terms
summary(mars2) %>% .$coefficients %>% head(10)


# Tuning hyperparametrs:
 # the maximum degree of interactions (degree)  
 # the number of terms (i.e., hinge functions determined by the 
 # optimal number of knots across all features) (nprune)
 # we perform a grid search to identify the optimal combination of 
 # hyperparameters that minimize the cv prediction error (cv-RMSE)

hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)
hyper_grid


# Cross-validated model (it takes 5 min to compute)
set.seed(123) 
cv_mars <- train(
  x = subset(ames_train, select = -Sale_Price),
  y = ames_train$Sale_Price,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid
)

# View results
cv_mars$bestTune
# interpret

cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)
# interpret

ggplot(cv_mars)
# interpret

# Note: optimal nrpune around 56. As a next step, we could perform a grid search 
# that focuses in on a refined grid space for nprune (e.g., comparing 45–65 terms retained).

          hyper_grid_beta <- expand.grid(
            degree = 1:3, 
            nprune = 45:50
          )
          hyper_grid
          
          set.seed(123) 
          cv_mars_beta <- train(
            x = subset(ames_train, select = -Sale_Price),
            y = ames_train$Sale_Price,
            method = "earth",
            metric = "RMSE",
            trControl = trainControl(method = "cv", number = 10),
            tuneGrid = hyper_grid_beta
          )


cv_mars$resample
# interpret

# Variable importance plots
p1 <- vip(cv_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)
# interpret variable importance based on impact to GCV (left) and RSS (right)
# note: it measures the importance of original feature; it does not measure the impact 
# for particular hinge functions created for a given feature

# see coef.
cv_mars$finalModel %>% coef()
# filter for interaction terms (if any)
cv_mars$finalModel %>%
  coef() %>%  
  broom::tidy() %>%  
  filter(stringr::str_detect(names, "\\*")) 
# if interactions are not identified, then => A tibble: 0 × 2
# if interactions identified, it is difficult to give meaning 
# to the numbers because the coef. depend on the scale


# use partial dependence plots (PDPs) to better understand them
  # PDPs for individual feature
  # PDPs for two features to understand the interactions 

# Construct partial dependence plots
p1 <- partial(cv_mars, pred.var = "Gr_Liv_Area", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(cv_mars, pred.var = "Year_Built", grid.resolution = 10) %>% 
  autoplot()
p3 <- partial(cv_mars, pred.var = c("Gr_Liv_Area", "Year_Built"), 
              grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

# Display plots side by side
gridExtra::grid.arrange(p1, p2,  ncol = 2)
gridExtra::grid.arrange(p3, ncol = 1)


######################################################################################################
# MARS with binary outcome.
# Consider the dataset attrition, where DV is categorical.
# Implement a similar code to predict employee attrition. 
# Case description: 
# problem type: supervised binomial classification
# response variable: Attrition (i.e., “Yes”, “No”)
# features: 30
# observations: 1,470
# objective: use employee attributes to predict if they will attrit (leave the company)
# access: provided by the rsample package (Kuhn and Wickham 2019)
# more details: See ?rsample::attrition
######################################################################################################

# access data
library(modeldata)
data(attrition)
str(attrition)

set.seed(123)
churn_split <- initial_split(attrition, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)




