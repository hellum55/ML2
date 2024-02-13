##############################################################
# Random forest (RF)
# Ch.11, HOM with R
# Adv.: greatly reduce instability and between-tree correlation
# Adv.: faster than bagging 
##############################################################
# Data: Ames 
# DV: Sale_Price (i.e., $195,000, $215,000)
# Features: 80
# Observations: 2,930
# Objective: use property attributes to predict the sale price of a house
# Access: provided by the AmesHousing package (Kuhn 2017a)
# more details: See ?AmesHousing::ames_raw


# Helper packages
library(dplyr)    # for data wrangling
library(ggplot2)  # for awesome graphics

# Modeling packages
library(ranger)   # a c++ implementation of random forest 
library(h2o)      # a java-based implementation of random forest
# NOTE: in ISL, we have use the original randomForest package (Liaw and Wiener 2002).
# ranger and h2o are the most modern implementations; they
# allow for parallelization to improve training time.
 

set.seed(123)

ames <- AmesHousing::make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)


# number of features
n_features <- length(setdiff(names(ames_train), "Sale_Price"))
n_features


# train a default random forest model using ranger library
ames_rf1 <- ranger(
  Sale_Price ~ ., 
  data = ames_train,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# get Out-of-bag RMSE (OOB RMSE)
(default_rmse <- sqrt(ames_rf1$prediction.error))


# Tunable hyperparameters that we should consider when training a RF model
  # create hyperparameter grid
  hyper_grid <- expand.grid(
    mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
    min.node.size = c(1, 3, 5, 10), 
    replace = c(TRUE, FALSE),                             
    sample.fraction = c(.5, .63, .8),                       
    rmse = NA                                               
  )
  
  # execute full cartesian grid search (where we assess every combination of hyperparameters of interest)
  for(i in seq_len(nrow(hyper_grid))) {
    # fit model for ith hyperparameter combination
    fit <- ranger(
      formula         = Sale_Price ~ ., 
      data            = ames_train, 
      num.trees       = n_features * 10,
      mtry            = hyper_grid$mtry[i],
      min.node.size   = hyper_grid$min.node.size[i],
      replace         = hyper_grid$replace[i],
      sample.fraction = hyper_grid$sample.fraction[i],
      verbose         = FALSE,
      seed            = 123,
      respect.unordered.factors = 'order',
    )
    # export OOB error 
    hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
  }
  
  # assess top 10 models
  hyper_grid %>%
    arrange(rmse) %>%
    mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
    head(10)

# ***
# An alternative implementation of RF using 
# h2o library (see HOM) is left as an exercise. 
# ***
  
# Feature importance 
  # re-run model with impurity-based variable importance
  rf_impurity <- ranger(
    formula = Sale_Price ~ ., 
    data = ames_train, 
    num.trees = 2000,  # notice the model is re-run with the optimal hyperparam identified before
    mtry = 32,
    min.node.size = 1,
    sample.fraction = .80,
    replace = FALSE,
    importance = "impurity",  # based on impurity
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
  )
  
  # re-run model with permutation-based variable importance
  rf_permutation <- ranger(
    formula = Sale_Price ~ ., 
    data = ames_train, 
    num.trees = 2000,
    mtry = 32,
    min.node.size = 1,
    sample.fraction = .80,
    replace = FALSE,
    importance = "permutation",  # based on permutation
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
  ) 
  
  # Typically, you will not see the same variable importance order
  # between the two options; however, you will often see similar 
  # variables at the top of the plots (and also the bottom). 
  
  p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE)
  p2 <- vip::vip(rf_permutation, num_features = 25, bar = FALSE)
  
  gridExtra::grid.arrange(p1, p2, nrow = 1)
  
  