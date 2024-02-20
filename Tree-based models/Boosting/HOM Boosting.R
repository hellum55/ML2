##############################################################
# Gradient Boosting and Extreme Gradient Boosting (XGB)
# Ch.12, HOM with R
# Adv.: the most competitive ensemble of trees alg.
# Disadv: tuning require much more strategy than a random forest 
##############################################################
# Data: Ames 
# DV: Sale_Price (i.e., $195,000, $215,000)
# Features: 80
# Observations: 2,930
# Objective: use property attributes to predict the sale price of a house
# Access: provided by the AmesHousing package (Kuhn 2017a)
# more details: See ?AmesHousing::ames_raw


# Helper packages
library(dplyr)       # for data wrangling
library(ggplot2)     # for awesome plotting
library(doParallel)  # for parallel backend to foreach (to speed computation)
library(foreach)     # for parallel processing with for loops
library(rsample)     # for sample split

# Modeling packages
library(gbm)      # for original implementation of regular and stochastic GBMs
library(h2o)      # for a java-based implementation of GBM variants
library(xgboost)  # for fitting extreme gradient boosting
library(caret)    # for general modelling


ames <- AmesHousing::make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)


# run a basic GBM model (This model takes a little over 2 minutes to run)
set.seed(123)  # for reproducibility
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train,
  distribution = "gaussian",  # SSE loss function
  n.trees = 5000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)
best
# [1] 1119

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])
## [1] 22.402

# plot training (in black) and cross-validated MSE (in green) as n trees are 
# added to the GBM algorithm. 
gbm.perf(ames_gbm1, method = "cv")


# Tuning strategy 
# 1. Choose a relatively high learning rate. Generally, the default value of 0.1 works,
#    but somewhere between 0.05–0.2 should work across a wide range of problems.
# 2. Determine the optimum number of trees for this learning rate.
# 3. Fix tree hyperparameters and tune learning rate and assess speed vs. performance.
# 4. Tune tree-specific parameters for decided learning rate.
# 5. Once tree-specific parameters have been found, lower the learning rate to 
#    assess for any improvements in accuracy.
# 6. Use final hyperparameter settings and increase CV procedures to get more 
#    robust estimates. 
#    Often, the above steps are performed with a simple validation procedure or 
#    5-fold CV due to computational constraints. 
#    If you used k-fold CV throughout steps 1–5 then this step is not necessary.

# We did step 1 and 2 before
# Now we proceed with step 3. 
# The following grid search took us about 10 minutes.

# create grid search to tune the learning rate
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      formula = Sale_Price ~ .,
      data = ames_train,
      distribution = "gaussian",
      n.trees = 5000, 
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = 3, 
      n.minobsinnode = 10,
      cv.folds = 10 
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

# results HOM
arrange(hyper_grid, RMSE)
##   learning_rate  RMSE trees  time
## 1         0.050 21382  2375 129.5   -->> BEST
## 2         0.010 21828  4982 126.0
## 3         0.100 22252   874 137.6
## 4         0.005 23136  5000 136.8
## 5         0.300 24454   427 139.9

## results 2024 my output 
##  learning_rate     RMSE trees time   Time
## 1         0.050 21807.96  1565   NA 50.098 -->> BEST
## 2         0.010 22102.34  4986   NA 49.772
## 3         0.100 22402.07  1119   NA 49.947
## 4         0.005 23054.68  4995   NA 50.537
## 5         0.300 24411.95   269   NA 50.636




# Next, step 4) we’ll set our learning rate at the optimal level found (0.05) 
# and tune the tree specific hyperparameters (interaction.depth and n.minobsinnode). 
# The following grid search took us about 30 minutes.

# search grid to tune the interaction.depth and n.minobsinnode 
hyper_grid <- expand.grid(
  n.trees = 6000,
  shrinkage = 0.05, # note there is a typo in the book
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)

# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = Sale_Price ~ .,
    data = ames_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}

# perform search grid with functional programming https://purrr.tidyverse.org/
hyper_grid$rmse <- purrr::pmap_dbl(      # function used for parallel mapping over the rows of the hyper_grid
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,               # ...1 first column in the hyper_grid
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# The end result is a data frame (hyper_grid) augmented with an additional 
# column rmse, containing the RMSE values for each combination of hyperparameters. 
# This process allows for identifying the hyperparameter combination that yields
# the best model performance, as measured by the RMSE


# results HOM
arrange(hyper_grid, rmse)
##   n.trees shrinkage interaction.depth n.minobsinnode  rmse
## 1    4000      0.05                 5              5 20699  --->> BEST (Adjusting the tree-specific parameters provides us with an additional reduction in RMSE).
## 2    4000      0.05                 3              5 20723
## 3    4000      0.05                 7              5 21021
## 4    4000      0.05                 3             10 21382
## 5    4000      0.05                 5             10 21915
## 6    4000      0.05                 5             15 21924
## 7    4000      0.05                 3             15 21943
## 8    4000      0.05                 7             10 21999
## 9    4000      0.05                 7             15 22348


## results 2024 my output 
## n.trees shrinkage interaction.depth n.minobsinnode     rmse
## 1    6000      0.05                 5             10 21793.28 --->> BEST
## 2    6000      0.05                 3             10 21807.96
## 3    6000      0.05                 5              5 21976.76
## 4    6000      0.05                 3              5 22104.49
## 5    6000      0.05                 5             15 22156.30
## 6    6000      0.05                 3             15 22170.16
## 7    6000      0.05                 7             10 22268.51
## 8    6000      0.05                 7              5 22316.37
## 9    6000      0.05                 7             15 22595.51





# Step 5. After this procedure, one can take the top model’s hyperparameter settings, 
# reduce the learning rate from 0.05 to 0.005, and increased the number of trees 
# (8000 and then to 10000) to see if any additional improvement in accuracy. 

set.seed(123)  
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train,
  distribution = "gaussian",  
  n.trees = 10000,
  shrinkage = 0.005,
  interaction.depth = 5,
  n.minobsinnode = 10,
  cv.folds = 10
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)
best
# [1] 9975

# RMSE
sqrt(ames_gbm1$cv.error[best])
## [1] 21682.26

# plot training and cross-validated MSE as n trees are added to the GBM algorithm. 
gbm.perf(ames_gbm1, method = "cv")


# Step 6. Use final hyperparameter settings and increase CV procedures to get 
# more robust estimates. We have already implemented 10-fold CV in the process. 


# 12.4 Stochastic GBMs is left as OPTIONAL reading. 


# ######################################
# 12.5 XGBoost (Chen and Guestrin 2016) 
########################################
#   * It is similar to Gradient Boosting, but even more efficient 
#   * eXtreme Gradient Boosting (regularization to prevent overfitting + 
#     an efficient implementation: While regular gradient boosting uses the 
#     loss function of our base model (e.g. decision tree) 
#     as a proxy for minimizing the error of the overall model, 
#     XGBoost uses the 2nd order derivative as an approximation 
#  
#   * It is very popular in data science 
#   * It is consistently used to win machine learning competitions on Kaggle 👀
#   * xgboost is a very effective implementation of an old idea coming 
#     from Adaboost and gradient boosted trees


#   * xgboost - the tuning parameters are similar to those of RF and Boosting,
#     but it introduces some new parameters
#   - Number of Boosting Iterations (nrounds, numeric) - the number of decision
#      trees (i.e., base learners) to ensemble 
#   - Max Tree Depth (max_depth, numeric)-depth of a tree (default:6,typical:3-10)
#   - Shrinkage (eta, numeric) - reducing contribution of subsequent models by 
#      shrinking the weights (default: 0.3, typical: 0.01-0.2)
#      Learning rate or shrinkage controls how much each tree contributes to the
#      overall model, with smaller values leading to more conservative updates 
#      and larger values leading to more aggressive updates.
#   - Minimum Loss Reduction (gamma, numeric) - specifies a minimum loss 
#      reduction required to make a further partition on a leaf node of the tree
#   - Other regularization parameters (lambda and alpha, numeric)- 
#      correspond to L1 and L2 regularization; setting both of these to greater
#      than 0 results in an elastic net regularization; 
#   - Subsample Ratio of Columns (colsample_bytree, numeric)- fraction of features 
#      to be sampled randomly for each of the trees (default: 1, typical: 0.5-1)
#   - Minimum Sum of Instance Weight (min_child_weight, numeric) - minimum sum 
#      of weight for a child node; protects overfit (default: 1)
#   - Subsample Percentage (subsample, numeric) - fraction of observations to be
#      sampled randomly for each of the trees (default: 1, typical: 0.5-1)


#   * Here an application with "xgboost" package 
#   * NOTE:when using xgboost package it requires some data preparation:
    #   - to encode our categorical variables numerically
    #   - matrix input for the features and the response to be a vector


# Preliminary data preprocessing: 
# encode our categorical variables numerically
library(recipes)
xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = ames_train, retain = TRUE) %>%
  juice() # function to extract the preprocessed data from the recipe object

# convert the training data frame to a matrix
X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price


# A series of grid searches similar to the previous sections can be performed (not shown)
# Based on this, one sets the model hyperparameters (in the params() list)

set.seed(123)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  objective = "reg:squarederror",  # for classification 0/1 objective = "binary:logistic",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)  

# minimum test CV RMSE
min(ames_xgb$evaluation_log$test_rmse_mean)
## [1] 20488 --->> this RMSE is slightly lower than the best regular boosting run before

## 
## results 2024 my output 
## [1] 23341.22



# Next, assess if overfitting is limiting our model’s performance by performing a grid search 
# that examines various regularization parameters (gamma, lambda, and alpha).
# it takes very long time
# cf. authors: Due to the low learning rate (eta), this cartesian grid search takes a long time. 
# They stopped the search after 2 hours and only 98 of the 245 models had completed.
# So take the code below as an example (do not have to run it in class)

# hyperparameter grid
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3, 
  min_child_weight = 3,
  subsample = 0.5, 
  colsample_bytree = 0.5,
  gamma = c(0, 1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = 0,          # a place to dump RMSE results
  trees = 0          # a place to dump required number of trees
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i], 
      lambda = hyper_grid$lambda[i], 
      alpha = hyper_grid$alpha[i]
    ) 
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
}

# results
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()
## Observations: 98
## Variables: 10
## $ eta              <dbl> 0.01, 0.01, 0.01, 0.01, 0.01, 0.0…
## $ max_depth        <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, …
## $ min_child_weight <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, …
## $ subsample        <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
## $ colsample_bytree <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
## $ gamma            <dbl> 0, 1, 10, 100, 1000, 0, 1, 10, 10…
## $ lambda           <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
## $ alpha            <dbl> 0.00, 0.00, 0.00, 0.00, 0.00, 0.1…
## $ rmse             <dbl> 20488, 20488, 20488, 20488, 20488…
## $ trees            <dbl> 3944, 3944, 3944, 3944, 3944, 381…



# Assuming the optimal hyperparameters found before, we fit the final model.
# We take the optimal number of trees found during cross validation (3944)

# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 3944,
  objective = "reg:squarederror",
  verbose = 0
)


# test error
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal())
prepare <- prep(blueprint, training = ames_train)
baked_test <- bake(prepare, new_data = ames_test)
# convert the training data frame to a matrix
X_test <- as.matrix(baked_test[setdiff(names(baked_test), "Sale_Price")])
Y_test <- baked_test$Sale_Price

predictions = predict(xgb.fit.final, X_test, n.trees = 3944) 
(RMSE = sqrt(mean((Y_test-predictions)^2)))


# variable importance plot
vip::vip(xgb.fit.final) 




# One can run XGBoost in caret library
# I keep the grid to minimum to reduce time of convergence

library(caret)
set.seed(123)

# Define train control
train.param <- trainControl(method = "cv", number = 5) 

# Define parameter grid for tuning
tune.grid.xgboost <- expand.grid(nrounds=300, 
                                 max_depth=3, 
                                 gamma=c(0), 
                                 eta=c(0.03),
                                 subsample=0.5, 
                                 colsample_bytree=0.1, 
                                 min_child_weight = 1)
# Train the xgboost model
model.xgboost <- train(Sale_Price ~ ., ames_train,
                       method = "xgbTree",
                       tuneGrid = tune.grid.xgboost,
                       trControl = train.param)
model.xgboost
# RMSE      Rsquared  MAE     
# 26370.56  0.893265  16807.21
# not the best result without tuning the parameters

#plot(model.xgboost)
var.imp <- varImp(model.xgboost, scale = FALSE)
plot(var.imp, 5)
var.imp$importance


# If you tune the parameters, it takes some hours to convergence
# Define parameter grid for tuning
tune.grid.xgboost <- expand.grid(
  nrounds = 300,
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.03, 0.05),
  gamma = c(0, 1, 5),
  colsample_bytree = seq(0.1, 1, by = 0.1),
  min_child_weight = c(1, 5, 10),
  subsample = seq(0.5, 1, by = 0.1)
)

# Train the xgboost model with parameter tuning
model.xgboost <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "xgbTree",
  tuneGrid = tune.grid.xgboost,
  trControl = train.param
)

# Print the model
print(model.xgboost)


# Concl. 
# GBMs are one of the most powerful ensemble algorithms
# Alternative GBMs exist and are developed continuously
# CatBoost(Dorogush, Ershov, and Gulin 2018) and LightGBM (Ke et al. 2017) 


