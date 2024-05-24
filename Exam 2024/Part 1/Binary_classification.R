library(readxl)
# Read the Excel file with specified column types
data_employee <- read_xls('Data.xls', 
                          na = c("", " ", "NA", "N/A", ".", "NaN", "MISSING"))

#Lets remove some of the variables we are sure are not giving anything to the analysis. For example ID variables and timestamps:
library(dplyr)
data_employee <- data_employee %>%
  select(-c(jobEndingYear, jobTitle.text, location.name, reviewId, reviewDateTime))

#In case of transforming a variable into binary.Lets convert the isCurrentJob variable into 1 and 0
data_employee$isCurrentJob <- ifelse(is.na(data_employee$isCurrentJob), 0, data_employee$isCurrentJob)
#data_employee$isCurrentJob <-ifelse(data_employee$isCurrentJob == "1",1,0)

#Transforming the ratingOverall variable into a binary variable. If the rating is => 4 the employee is satisfied
#If the rating is <= 3 the employee is not satisfied
job_satisfaction = rep(0, length(data_employee$ratingOverall))
job_satisfaction[data_employee$ratingOverall >= 4] = "Satisfied"
job_satisfaction[data_employee$ratingOverall <= 3] = "Not_Satisfied"
data_employee=data.frame(data_employee,job_satisfaction)
#Removing ratingOverall variable
data_employee <- subset(data_employee, select = -ratingOverall)

#Check the variable types of the variables:
str(data_employee)
# Convert the categorical variables into factors for the analysis:
#data_employee$ratingOverall <- factor(data_employee$ratingOverall)
data_employee$ratingCeo <- factor(data_employee$ratingCeo)
data_employee$ratingBusinessOutlook <- factor(data_employee$ratingBusinessOutlook)
#data_employee$ratingWorkLifeBalance <- factor(data_employee$ratingWorkLifeBalance)
#data_employee$ratingCultureAndValues <- factor(data_employee$ratingCultureAndValues)
#data_employee$ratingDiversityAndInclusion <- factor(data_employee$ratingDiversityAndInclusion)
#data_employee$ratingSeniorLeadership <- factor(data_employee$ratingSeniorLeadership)
data_employee$ratingRecommendToFriend <- factor(data_employee$ratingRecommendToFriend)
#data_employee$ratingCareerOpportunities <- factor(data_employee$ratingCareerOpportunities)
#data_employee$ratingCompensationAndBenefits <- factor(data_employee$ratingCompensationAndBenefits)
#data_employee$isCurrentJob <- factor(data_employee$isCurrentJob)
data_employee$employmentStatus <- factor(data_employee$employmentStatus)
#data_employee$jobEndingYear <- factor(data_employee$jobEndingYear)
str(data_employee)

#Delete all the NA's in the dataframe
data_employee <- na.omit(data_employee)

############################################ SPLITTING THE DATA ###########################################################
#Create the recipe
library(rsample)
set.seed(123)
split <- initial_split(data_employee, prop = 0.7, strata = "job_satisfaction") 

employee_train <- training(split)
employee_test <- testing(split)

# imbalanced
prop.table(table(employee_train$job_satisfaction))
prop.table(table(employee_test$job_satisfaction))
#It is a imbalanced data set with the majority of the observations => 3, which is good for the company but can be hard to predict on.

#If the target variable is binary we can make sure that the test split is balanced instead of stratisfied
#This can only be done with 2 levels. It ensures that the data are balanced.
library(ROSE)
employee_train <- ovun.sample(job_satisfaction~., data=employee_train, 
                              p=0.5, seed=2, 
                              method="over")$data

employee_train <- ovun.sample(job_satisfaction~., data=employee_train, 
                              p=0.5, seed=2, 
                              method="over")$data


prop.table(table(employee_train$job_satisfaction)) # more balanced after oversampling
prop.table(table(employee_test$job_satisfaction))
# reflection here (why is balancing necessary; 
#Later on it will help the relevance of the model. If the test set contains 80% of one class the model can simply just predict that outcome every time, but it
#will never be a good model, because it does not capture the other outcome, it just learns to predict one class. Not very nice!

############################################ BUILDING THE MODELS ###########################################################
library(recipes)
employee_recipe <- recipe(job_satisfaction ~ ., data = employee_train) %>%
  #step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)

str(baked_train)
#### A list for Classification results
modelComparison = NULL
modelComparison = data.frame()

# Decision trees ------------------------------------------------------------
library(rpart)       # direct engine for decision tree application
library(caret)       # meta engine for decision tree application

#Using caret to show Accuracy. It applies 10fold-CV with 20 alpha parameters to tune on. Lower alphas = deeper trees, and 
#helps to minimize error. 
tune.grid <- data.frame(.maxdepth = 3:10, 
                        .mincriterion = c(.1, .2, .3, .4))
# training parameters. Here we apply 5fold-CV
train.param <- trainControl(method = "cv", number = 5)

tree.model <- train(job_satisfaction ~., baked_train,
                    method = "ctree2",
                    metric = "Kappa",
                    trControl = train.param,
                    tuneGrid = tune.grid) 

tree.model$results$Accuracy
#The best model for the simple decision tree is a maxdepth of 9 and a mincriterion of 0.3 that yields
#an accuracy of 0.8781 and a kappa of 0.75

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
tree.class.pred <- predict(tree.model, 
                           baked_train, type = "raw") 

tree.scoring <- predict(tree.model, 
                        baked_train, 
                        type = "prob")[,"Satisfied"]


tree.conf <- confusionMatrix(tree.class.pred, real.pred,
                             positive = "Satisfied",
                             mode = "prec_recall")
tree.conf

# ROC and AUC
library(caTools)
tree.auc = colAUC(tree.scoring , real.pred, plotROC = TRUE) 

modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Basic Tree', 
                                   accuracy = tree.conf[[3]][[1]], 
                                   kappa = tree.conf[[3]][[2]], 
                                   precision = tree.conf[[4]][[5]], 
                                   recall_sensitivity = tree.conf[[4]][[6]], 
                                   specificity = tree.conf[[4]][[2]], 
                                   auc = tree.auc))

# MARS model ------------------------------------------------------------
# Cross-validated model
set.seed(123)  # for reproducibility
# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

cv_mars <- train(
  x = subset(baked_train, select = -job_satisfaction),
  y = baked_train$job_satisfaction,
  method = "earth",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = hyper_grid
)
# View results
cv_mars$bestTune
##    nprune degree
## 13     23      2

cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)
#degree nprune  Accuracy     Kappa  AccuracySD    KappaSD
#1      2     23 0.8787923 0.7575491 0.009631602 0.01927495

ggplot(cv_mars)
#The model is performing the best with around 12-13 terms, so a rather small model actually.

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
marsmodel.class.pred <- predict(cv_mars, 
                                baked_train, 
                                type = "raw")

marsmodel.scoring <- predict(cv_mars, 
                             baked_train, 
                             type = "prob")[, "Satisfied"]

marsmodel.conf <- confusionMatrix(marsmodel.class.pred, real.pred,
                                  positive = "Satisfied", 
                                  mode = "prec_recall")

# ROC and AUC
marsmodel.auc = colAUC(marsmodel.scoring, real.pred, plotROC = TRUE) 


#Save results
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'MARS', 
                                   accuracy = marsmodel.conf[[3]][[1]], 
                                   kappa = marsmodel.conf[[3]][[2]], 
                                   precision = marsmodel.conf[[4]][[5]], 
                                   recall_sensitivity = marsmodel.conf[[4]][[6]], 
                                   specificity = marsmodel.conf[[4]][[2]], 
                                   auc = marsmodel.auc))


# bagging ----------------------------------------------------------------------------------------------------------------------
bag_model <- train(
  job_satisfaction ~ .,
  data = baked_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5), # 5-fold CV increases the convergence time
  nbagg = 100,  
  control = rpart.control(minsplit = 2, cp = 0)
)
bag_model

# View results
bag_model$results
#parameter  Accuracy     Kappa AccuracySD   KappaSD
#1    none 0.9417062 0.8833684 0.00722381 0.0144619

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
bag_model.class.pred <- predict(bag_model, 
                                baked_train, 
                                type = "raw") 

bag_model.scoring <- predict(bag_model, 
                             baked_train, 
                             type = "prob")[,"Satisfied"]

bag_model.conf <- confusionMatrix(bag_model.class.pred, real.pred,
                                  positive = "Satisfied", 
                                  mode = "prec_recall")

bag_model.conf
# ROC and AUC
bag.auc = colAUC(bag_model.scoring, real.pred, plotROC = TRUE) 

modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Bagging', 
                                   accuracy = bag_model.conf[[3]][[1]], 
                                   kappa = bag_model.conf[[3]][[2]], 
                                   precision = bag_model.conf[[4]][[5]], 
                                   recall_sensitivity = bag_model.conf[[4]][[6]], 
                                   specificity = bag_model.conf[[4]][[2]], 
                                   auc = bag.auc))

# random forest -----------------------------------------------------------------------------
library(ranger)   # a c++ implementation of random forest 
library(h2o)      # a java-based implementation of random forest
#create tunegrid with 15 values from 1:15 for mtry to tunning model. 
#Our train function will change number of entry variable at each split according to tunegrid. 

tunegrid <- expand.grid(.mtry = (1:15))
train.param <- trainControl(method = "cv", number = 5)

rf.model <- train(job_satisfaction ~ ., baked_train,
                  method = "rf", 
                  ntree = 1000,
                  metric = "Accuracy",
                  trControl = train.param,
                  tune.grid = tunegrid)

rf.model

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
rfmodel.class.pred <- predict(rf.model, 
                              baked_train, 
                              type = "raw") 

rfmodel.scoring <- predict(rf.model, 
                           baked_train, 
                           type = "prob")[,"Satisfied"]

rfmodel.conf <- confusionMatrix(rfmodel.class.pred, real.pred,
                                positive = "Satisfied", 
                                mode = "prec_recall")

# ROC and AUC
rf.auc = colAUC(rfmodel.scoring, real.pred, plotROC = TRUE) 


#Save results
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'RF tree', 
                                   accuracy = rfmodel.conf[[3]][[1]], 
                                   kappa = rfmodel.conf[[3]][[2]], 
                                   precision = rfmodel.conf[[4]][[5]], 
                                   recall_sensitivity = rfmodel.conf[[4]][[6]], 
                                   specificity = rfmodel.conf[[4]][[2]], 
                                   auc = rf.auc))
# Gradient boosting --------------------------------------------------------------------------------------------------------------------------
library(gbm) 
set.seed(123)  # for reproducibility
ames_gbm1 <- gbm(
  formula = job_satisfaction ~ .,
  data = baked_train,
  distribution = "bernoulli",  # SSE loss function
  n.trees = 500,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 5
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])
## [1] 23240.38

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction
xgb.class.pred <- predict(model.xgboost, 
                          baked_train, 
                          type = "raw") 
xgb.scoring <- predict(model.xgboost, 
                       baked_train, 
                       type = "prob")[, "Satisfied"] 
xgb.conf <- confusionMatrix(data = xgb.class.pred, 
                            reference = real.pred, 
                            positive = "Satisfied", 
                            mode = "prec_recall") 

# ROC and AUC
xgb.auc = colAUC(xgb.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'xgb tree', 
                                   accuracy = xgb.conf[[3]][[1]], 
                                   kappa = xgb.conf[[3]][[2]], 
                                   precision = xgb.conf[[4]][[5]], 
                                   recall_sensitivity = xgb.conf[[4]][[6]], 
                                   specificity = xgb.conf[[4]][[2]], 
                                   auc = xgb.auc))


# xgboost -------------------------------------------------------------------------------------------------------------------------------------
#The hyperparamters are:
#Tree-depth: That controls the depth of the tree. Smaller trees are computed faster. Higher trees can capture unique interactions. (min3, max6)
#alpha/LR: Controls how quicly the tree proceeds and learns. Smaller vales makes the model robust, but can be stuck at a global minimum. More robust to overfitting.
#Training on a 0.5 subsample of the rows of the data set = lower tree correlation, higher accuracy.
# 0.1 subsample on the columns. 
#Gamma is a regularization strategy to reduce complexity. It specifies a minimum loss reduction required to make another node/leaf. XGBoost will grow the tree to the max
#and then pruning the tree and remove splits that do not meet the gamma.
#N_rounds = number of trees
model.xgboost <- train(job_satisfaction ~ ., baked_train,
                       method = "xgbTree",
                       metric = "Kappa",
                       tuneGrid = expand.grid(max_depth=3:6,
                                              gamma=c(0, 1, 2, 3, 5), 
                                              eta=c(0.03, 0.06, 0.1, 0.2), 
                                              nrounds=300,
                                              subsample=0.5, 
                                              colsample_bytree=0.1, 
                                              min_child_weight = 1),
                       trControl = train.param)
model.xgboost



# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction
xgb.class.pred <- predict(model.xgboost, 
                          baked_train, 
                          type = "raw") 
xgb.scoring <- predict(model.xgboost, 
                       baked_train, 
                       type = "prob")[, "Satisfied"] 
xgb.conf <- confusionMatrix(data = xgb.class.pred, 
                            reference = real.pred, 
                            positive = "Satisfied", 
                            mode = "prec_recall") 

# ROC and AUC
xgb.auc = colAUC(xgb.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'xgb tree', 
                                   accuracy = xgb.conf[[3]][[1]], 
                                   kappa = xgb.conf[[3]][[2]], 
                                   precision = xgb.conf[[4]][[5]], 
                                   recall_sensitivity = xgb.conf[[4]][[6]], 
                                   specificity = xgb.conf[[4]][[2]], 
                                   auc = xgb.auc))

# 4) SVM -----------------------------------------------------------------------------------
#Use only a part of the data
set.seed(123)

# Randomly select 15% of the rows from the data
data_employee_small <- data_employee[sample(nrow(data_employee), nrow(data_employee) * 0.15), ]

split_small <- initial_split(data_employee_small, prop = 0.7, strata = "job_satisfaction") 

employee_train_small <- training(split_small)
employee_test_small <- testing(split_small)

employee_train_small <- ovun.sample(job_satisfaction~., data=employee_train_small, 
                                    p=0.5, seed=2, 
                                    method="over")$data

prop.table(table(employee_train_small$job_satisfaction)) # more balanced after oversampling
prop.table(table(employee_test_small$Satisfied))

employee_recipe_small <- recipe(job_satisfaction ~ ., data = employee_train_small) %>%
  #step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare_small <- prep(employee_recipe_small, training = employee_train_small)
prepare_small$steps

baked_train_small <- bake(prepare_small, new_data = employee_train_small)
baked_test_small <- bake(prepare_small, new_data = employee_test_small)

train.param <-  trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Tune an SVM with radial basis kernel 
set.seed(1854)  
employee_svm <- train(
  job_satisfaction ~ ., 
  data = baked_train_small,
  method = "svmRadial",               
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 10
)
# Plot results
ggplot(employee_svm) + theme_light()

# Print results
employee_svm$results 
# notice the default is accuracy

# RE-run with trContol to get the class probabilities for AUC/ROC     
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,             
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  
employee_svm_auc <- train(
  job_satisfaction ~ ., 
  data = baked_train_small,
  method = "svmRadial",               
  metric = "ROC",  # explicitly set area under ROC curve as criteria        
  trControl = ctrl,
  tuneLength = 10
)

confusionMatrix(employee_svm_auc)
# interpret

# Feature importance 
# Create a wrapper to retain the predicted class probabilities for the class of interest (in this case, Yes)
library(kernlab)  # also for fitting SVMs 

# Model interpretability packages
library(pdp)      # for partial dependence plots, etc.
library(vip)      # for variable importance plots

prob_satisfied <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Satisfied"]
}

#variable importance plot
set.seed(2827)  # for reproducibility
vip::vip(employee_svm_auc, method = "permute", event_level = "second", nsim = 5, train = baked_train_small, 
    target = "job_satisfaction", metric = "roc_auc", reference_class = "Satisfied", 
    pred_wrapper = prob_satisfied)

#construct PDP (feature effect plots are on the probability scale)
features <- c("OverTime", "WorkLifeBalance", 
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
  partial(churn_svm_auc, pred.var = x, which.class = 2,        # since the predicted probabilities from our model come in two columns (No and Yes), we specify which.class = 2 so that our interpretation is in reference to predicting Yes
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})
grid.arrange(grobs = pdps,  ncol = 2)
# interpret

# confusion matrix + KAPPA 
real.pred <- baked_train_small$job_satisfaction #
svm.class.pred <- predict(svm_rad, baked_train_small, type = "raw") 
svm.scoring <- predict(svm_rad, baked_train_small, type = "prob") [, "Satisfied"] 

svm.conf <- confusionMatrix(data = svm.class.pred, reference = real.pred, positive = "Satisfied", mode = "prec_recall") 

# ROC and AUC
svm.auc = colAUC(svm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, data.frame(model = 'svm', accuracy = svm.conf[[3]][[1]], kappa = svm.conf[[3]][[2]], precision = svm.conf[[4]][[5]], recall_sensitivity = svm.conf[[4]][[6]], specificity = svm.conf[[4]][[2]], auc = svm.auc))
modelComparison

#The model with the highest and also highest kappa coefficients is the random forest model. with 99% accuracy and 98%, which is quite crazy. We might need some more data.
#The kappa coefficients is an estimate of how much better the model are rather than just random guessing. Lets have a look on how well it does on the test set:

#Random forest on test set --------------------------------------------------------------------------------------------------------------
# confusion matrix + KAPPA 
real.pred <- baked_test$job_satisfaction 
rfmodel.class.pred <- predict(rf.model, 
                              baked_test, 
                              type = "raw") 

rfmodel.scoring <- predict(rf.model, 
                           baked_test, 
                           type = "prob")[,"Satisfied"]

rfmodel.conf <- confusionMatrix(rfmodel.class.pred, real.pred,
                                positive = "Satisfied", 
                                mode = "prec_recall")
rfmodel.conf

# ROC and AUC
rf.auc = colAUC(rfmodel.scoring, real.pred, plotROC = TRUE) 
rf.auc
#The model has 90% accuracy on new data. very impressive.the AUC is 95%. it is always a discussion on where the trade off lays. 

#Lets have a look on the feature importance:
# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects

vip::vip(rf.model, num_features = 20, bar = FALSE)

# PDPs
# Construct partial dependence plots
p1 <- pdp::partial(
  rf.model, 
  pred.var = "ratingRecommendToFriend_POSITIVE",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  rf.model, 
  pred.var = "ratingCultureAndValues", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)
#When looking at the varibale importance for bagging models and random forest model it is not quite the same as with regression models or logistic models.
#RF and bagging consists of hundreds of trees and the way the variable importance are generated are with adding the amount a given predictor has decreased
#the gini index on the splits over all the different deep trees. The largest mean decrease in gini index for this data set
#is wheter you can recommend the job to a friend are the rating an employee gave the culture.