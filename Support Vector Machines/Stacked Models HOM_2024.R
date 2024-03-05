#######################################################################
# Stacked models -----------------------------------------------------
# Ch.15, HOM with R
# Adv.: combine the predictions of several strong base learners
#.      to improve the prediction accuracy; it can reduce overfitting
# Disadv.: computationally very expensive;
#          requires careful consideration and experimentation 
# -- update 2024 --
########################################################################

# The process of stacking models typically involves the following steps:
  #  Training a set of base models on the training data.
  #  Using the base models to make predictions on the validation sets.
  #  Using the validation sets predictions as input to train a meta-model 
  #  (stacking model).
  #  Use the meta-model to make final predictions on the test data.



# Data: Ames 
# DV: Sale_Price (i.e., $195,000, $215,000)
# Features: 80
# Observations: 2,930
# Objective: use property attributes to predict the sale price of a house
# Access: provided by the AmesHousing package (Kuhn 2017a)
# more details: See ?AmesHousing::ames_raw



# Helper packages
library(rsample)   
library(recipes)   # to prepare the data in this script

# Modeling packages
library(h2o)       # for fitting stacked models in this script
                   # H2O is the scalable open source machine learning platform 
                   # that offers parallelized implementations of many supervised
                   # and unsupervised machine learning algorithms
                   # https://cran.r-project.org/web/packages/h2o/index.html


# Load and split the Ames housing data
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
summary(ames_train)


# Make sure we have consistent categorical levels
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)
# Any level of a categorical variable that does not make up at least 0.5% 
# of the total observations will be collapsed into an "other" category



# To use h2o from R, you must start or connect to the "H2O cluster", 
# the term we use to describe the backend H2O Java engine. To run H2O 
# on your local machine, call h2o.init without any arguments
h2o.init() # Connection successful!



# Create training & test sets for h2o
train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  juice() %>% # here juice() is superseded in favor of bake(). 
  # it is used to extract transformed training set given retain=TRUE
  as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>%
  bake(new_data = ames_test) %>% # code as in ML1
  as.h2o()

# Note that no actual data is stored in the R workspace; and no actual work is 
# carried out by R. R only saves the named objects, which uniquely identify the 
# data set, model, etc on the server. When the user makes a request, R queries 
# the server via the REST API, which returns a JSON file with the
# relevant information that R then displays in the console.


# Get response and feature names
Y <- "Sale_Price"
X <- setdiff(names(ames_train), Y)



# 15.3 Stacking existing algorithms  -------------------------------------------

# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo", 
# "Modulo" implies same fold assignment to ensure the same observations are used
  keep_cross_validation_predictions = TRUE, seed = 123)

# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000, mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate a GBM model

start.time <- proc.time()

best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 5000, learn_rate = 0.01,
  max_depth = 7, min_rows = 5, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
# GBM model takes about 360 seconds (6 min) to execute


# OBS: XGBoost model NOT AVAILABLE IN h2o FOR MAC AND WINDOWS  
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html#limitations



# Stack these models together using RF as a meta-learner
start.time <- proc.time()

ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm),
  metalearner_algorithm = "drf" # here using a RF as a meta-learning algorithm
                                # ?h2o.stackedEnsemble
)

stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
# it takes about 4 seconds to execute


# Test error 
  #  for individual base learners
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}
list(best_glm, best_rf, best_gbm) %>%
  purrr::map_dbl(get_rmse)
##[1] 30018.69 22247.60 20105.70

  # for stacked model
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE
## [1] 20698.79 

# based on the above, on the test data best_gbm is best


# Stacking never does worse than selecting the single best base 
# learner on the training data, but not necessarily the validation 
# or test data. The biggest gains are usually produced when stacking
# base learners that have high variability, and uncorrelated predicted values.
# The more similar the predicted values are between the base learners, 
# the less advantage there is to combining them. 

# If we assess the correlation of the CV predictions for our data we can see strong 
# correlation across the base learners, especially with three tree-based learners
data.frame(
  GLM_pred = as.vector(h2o.getFrame(best_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(best_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  GBM_pred = as.vector(h2o.getFrame(best_gbm@model$cross_validation_holdout_predictions_frame_id$name))
) %>% cor()
# This explains why stacking provides less advantage in this situation




# 15.4 Stacking a grid search --------------------------------------------------
# Define GBM hyperparameter grid
hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1),
  learn_rate_annealing = c(0.99, 1),
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1)
)

# Define random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 25
)

# Build random grid search 
start.time <- proc.time()

random_grid <- h2o.grid(
  algorithm = "gbm", grid_id = "gbm_grid", x = X, y = Y,
  training_frame = train_h2o, hyper_params = hyper_grid,
  search_criteria = search_criteria, ntrees = 500, stopping_metric = "RMSE",     
  stopping_rounds = 10, stopping_tolerance = 0, nfolds = 10, 
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123
)
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)
# about 365.503 seconds (aprox 6 min) to converge with ntrees=500.



# If we look at the grid search models we see that the cross-validated RMSE ranges from 20756–57826
# Sort results by RMSE
h2o.getGrid(
  grid_id = "gbm_grid", 
  sort_by = "rmse"
)



## Results for ntrees = 5000 (HOM)
## H2O Grid Details 
## ================
## 
## Grid ID: gbm_grid 
## Used hyper parameters: 
##   -  col_sample_rate 
##   -  learn_rate 
##   -  learn_rate_annealing 
##   -  max_depth 
##   -  min_rows 
##   -  sample_rate 
## Number of models: 25 
## Number of failed models: 0 
## 
## Hyper-Parameter Search Summary: ordered by increasing rmse
##   col_sample_rate learn_rate learn_rate_annealing max_depth min_rows sample_rate         model_ids               rmse
## 1             0.9       0.01                  1.0         3      1.0         1.0 gbm_grid_model_20  20756.16775065606
## 2             0.9       0.01                  1.0         5      1.0        0.75  gbm_grid_model_2 21188.696088824694
## 3             0.9        0.1                  1.0         3      1.0        0.75  gbm_grid_model_5 21203.753908665003
## 4             0.8       0.01                  1.0         5      5.0         1.0 gbm_grid_model_16 21704.257699437963
## 5             1.0        0.1                 0.99         3      1.0         1.0 gbm_grid_model_17 21710.275753497197
## 
## ---
##    col_sample_rate learn_rate learn_rate_annealing max_depth min_rows sample_rate         model_ids               rmse
## 20             1.0       0.01                  1.0         1     10.0        0.75 gbm_grid_model_11 26164.879525289896
## 21             0.8       0.01                 0.99         3      1.0        0.75 gbm_grid_model_15  44805.63843296435
## 22             1.0       0.01                 0.99         3     10.0         1.0 gbm_grid_model_18 44854.611500840605
## 23             0.8       0.01                 0.99         1     10.0         1.0 gbm_grid_model_21 57797.874642563846
## 24             0.9       0.01                 0.99         1     10.0        0.75 gbm_grid_model_10  57809.60302408739
## 25             0.8       0.01                 0.99         1      5.0        0.75  gbm_grid_model_4  57826.30370545089



# If we apply the best performing model to our test set, we achieve an RMSE of 21599.8.
# Grab the model_id for the top model, chosen by validation error
best_model_id <- random_grid@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
h2o.performance(best_model, newdata = test_h2o)
## H2ORegressionMetrics: gbm (results for ntrees = 5000)
## 
## MSE:  466551295
## RMSE:  21599.8
## MAE:  13697.78
## RMSLE:  0.1090604
## Mean Residual Deviance :  466551295



# Rather than use the single best model, we can combine all the models in our 
# grid search using a super learner
# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "ensemble_gbm_grid",
  base_models = random_grid@model_ids, metalearner_algorithm = "gbm"
)

# Eval ensemble performance on a test set
h2o.performance(ensemble, newdata = test_h2o)
## H2ORegressionMetrics: stackedensemble (results for ntrees= 5000)
## 
## MSE:  469579433
## RMSE:  21669.78 *
## MAE:  13499.93
## RMSLE:  0.1061244
## Mean Residual Deviance :  469579433

# * Evaluating RMSE in this example, our super learner does not provide 
# any performance gains.
# However, in cases where you see high variability across hyperparameter 
# settings for your leading models, stacking the grid search or even 
# the leaders in the grid search can provide significant performance gains.





# 15.5 Automated search (automated machine learning)  --------------------------

# Rather than search across a variety of parameters for a single base learner, 
# one can perform a search across a variety of hyperparameter settings for 
# many different base learners simultaneously and then stuck the resulting models

# h2o provides an open source implementation of AutoML with the h2o.automl() function
# By default, h2o.automl() will search for 1 hour 
# one can control how long it searches by adjusting a variety of stopping arguments
# the following performs an automated search for two hours, which ended up 
# assessing 80 models


# Use AutoML to find a list of candidate models (i.e., leaderboard)
auto_ml <- h2o.automl(
  x = X, y = Y, training_frame = train_h2o, nfolds = 5, 
  max_runtime_secs = 60 * 120, max_models = 20,     # need to reduce these to get the convergence fast
  keep_cross_validation_predictions = TRUE, sort_metric = "RMSE", seed = 123,
  stopping_rounds = 50, stopping_metric = "RMSE", stopping_tolerance = 0
)

# Assess the leader board; the following truncates the results to show the top 
# 25 models. 
auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)

# You can get the top model with auto_ml@leader
auto_ml@leader



#Results for max_runtime_secs = 60 * 120, max_models = 50 (HOM)

##                                               model_id   rmse
## 1                     XGBoost_1_AutoML_20190220_084553   22229.97
## 2            GBM_grid_1_AutoML_20190220_084553_model_1   22437.26
## 3            GBM_grid_1_AutoML_20190220_084553_model_3   22777.57
## 4                         GBM_2_AutoML_20190220_084553   22785.60
## 5                         GBM_3_AutoML_20190220_084553   23133.59
## 6                         GBM_4_AutoML_20190220_084553   23185.45
## 7                     XGBoost_2_AutoML_20190220_084553   23199.68
## 8                     XGBoost_1_AutoML_20190220_075753   23231.28
## 9                         GBM_1_AutoML_20190220_084553   23326.57
## 10           GBM_grid_1_AutoML_20190220_075753_model_2   23330.42
## 11                    XGBoost_3_AutoML_20190220_084553   23475.23
## 12       XGBoost_grid_1_AutoML_20190220_084553_model_3   23550.04
## 13      XGBoost_grid_1_AutoML_20190220_075753_model_15   23640.95
## 14       XGBoost_grid_1_AutoML_20190220_084553_model_8   23646.66
## 15       XGBoost_grid_1_AutoML_20190220_084553_model_6   23682.37
## ...                                                ...        ...
## 65           GBM_grid_1_AutoML_20190220_084553_model_5   33971.32
## 66           GBM_grid_1_AutoML_20190220_075753_model_8   34489.39
## 67  DeepLearning_grid_1_AutoML_20190220_084553_model_3   36591.73
## 68           GBM_grid_1_AutoML_20190220_075753_model_6   36667.56
## 69      XGBoost_grid_1_AutoML_20190220_084553_model_13   40416.32
## 70           GBM_grid_1_AutoML_20190220_075753_model_9   47744.43
## 71    StackedEnsemble_AllModels_AutoML_20190220_084553   49856.66
## 72    StackedEnsemble_AllModels_AutoML_20190220_075753   59127.09
## 73 StackedEnsemble_BestOfFamily_AutoML_20190220_084553   76714.90
## 74 StackedEnsemble_BestOfFamily_AutoML_20190220_075753   76748.40
## 75           GBM_grid_1_AutoML_20190220_075753_model_5   78465.26
## 76           GBM_grid_1_AutoML_20190220_075753_model_3   78535.34
## 77           GLM_grid_1_AutoML_20190220_075753_model_1   80284.34
## 78           GLM_grid_1_AutoML_20190220_084553_model_1   80284.34
## 79       XGBoost_grid_1_AutoML_20190220_075753_model_4   92559.44
## 80      XGBoost_grid_1_AutoML_20190220_075753_model_10  125384.88


# most of the leading models are GBM variants and achieve an RMSE in the 
# 22000–23000 range. Notice, this is not as good as some of our best models 
# trained by exploring the hyperparameters grid manually.

# However, we can use it as an exploratory tool when we have no idea where to 
# set the hyperparameters, and then continue to explore in that region. Thus
# the random search can serve as a starting point in defining our grid 
# values

# To extract the hyperparameters associated with the best model in random search
auto_ml@leader@parameters 

# To extract parameters for the model in the 4th position
leaderboard_df <- as.data.frame(auto_ml@leaderboard)
model_id_fourth <- leaderboard_df$model_id[4]
# Retrieve the model from H2O using its model_id
model_fourth <- h2o.getModel(model_id_fourth)
# Extract the parameters of this model
parameters_fourth <- model_fourth@parameters
parameters_fourth

