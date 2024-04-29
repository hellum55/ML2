# Case study: "Tous" 
# Machine Learning for BI II
# Version: 2023 - my version -----incl . LR, regularized reg in addition and svm solution

library(tidyverse)
library(caTools)
library(gtools)
library(dplyr)
library(tidytext)
library(ggplot2)
library(MASS)
library(ISLR)
library(caretEnsemble)
library(caret)
library(farff)
library(Matrix)
library(leaps)
library(mgcv)
library(gam)
library(party)
library(DMwR2)
library(DataExplorer) 
library(car)
library(rpart)
library(rpart.plot)
library(performanceEstimation)
library(AID)
library(psych)
library(factoextra)
library(randomForest)
library(gbm)
library(xgboost)
library(e1071)
library(glmnet)
library(pROC)
library(Hmisc)
library(corrplot)
library(kernlab)
library(xray)
library(ROSE)
library(PRROC)

#Read in the data
data_employee <- read.csv("data_employee.csv", stringsAsFactors=TRUE)

#Convert the ratingOverall into a binary class
Satisfied = rep(0, length(data_employee$ratingOverall))
Satisfied[data_employee$ratingOverall >= 4] = "Satisfied"
Satisfied[data_employee$ratingOverall <= 3] = "Not.Satisfied"
data_employee=data.frame(data_employee,Satisfied)
str(data_employee)
data_employee$Satisfied = as.factor(data_employee$Satisfied)
#Remove ratingOverall
data_employee <- subset(data_employee, select = -ratingOverall)

#Create the recipe
set.seed(123)
split <- initial_split(data_employee, prop = 0.8, strata = "Satisfied") 

employee_train <- training(split)
employee_test <- testing(split)

employee_recipe <- recipe(Satisfied ~ ., data = employee_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_integer_predictors()) %>%
  step_scale(all_integer(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)

# unbalanced
prop.table(table(baked_train$Satisfied))
#The baked_train dataset is unbalanced, which is good for the company. 65% of the employees are satisfied.

# oversampling
baked_train <- ovun.sample(Satisfied~., data=baked_train, 
                          p=0.5, seed=2, 
                          method="over")$data

prop.table(table(baked_train$Satisfied)) # more balanced after oversampling
prop.table(table(baked_test$Satisfied))
# reflection here (why is balancing necessary; 
# how it helps later in evaluating accuracy; why not used for testing data)

#### Classification
modelComparison = NULL 
modelComparison = data.frame() 

# Decision trees ------------------------------------------------------------

# training parameters
train.param <- trainControl(method = "cv", number = 5)

# 1) basic decision tree
tune.grid <- data.frame(.maxdepth = 3:10, 
                        .mincriterion = c(.1, .2, .3, .4))
tree.model <- train(Satisfied ~., baked_train,
                    method = "ctree2",
                    metric = "Kappa",
                    trControl = train.param,
                    tuneGrid = tune.grid) 

tree.model

# confusion matrix + KAPPA 
real.pred <- baked_test$Satisfied 
tree.class.pred <- predict(tree.model, 
                           baked_test, type = "raw") 

tree.scoring <- predict(tree.model, 
                        baked_test, 
                        type = "prob") [, "Satisfied"]

tree.conf <- confusionMatrix(tree.class.pred, real.pred,
                positive = "Satisfied",
                mode = "prec_recall")

# ROC and AUC
tree.auc = colAUC(tree.scoring , real.pred, plotROC = TRUE) 

modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Basic Tree', 
                                   accuracy = tree.conf[[3]][[1]], 
                                   kappa = tree.conf[[3]][[2]], 
                                   precision = tree.conf[[4]][[5]], 
                                   recall_sensitivity = tree.conf[[4]][[6]], 
                                   specificity = tree.conf[[4]][[2]], 
                                   auc = tree.auc))


# random forest -----------------------------------------------------------------------------
rf.model <- train(Satisfied ~ ., baked_train,
                  method = "rf", 
                  ntree = 1000,
                  metric = "Kappa",
                  trControl = train.param,
                  tune.grid = expand.grid(.mtry=c(1:28)))

rf.model

# confusion matrix + KAPPA 
real.pred <- baked_test$Satisfied 
rfmodel.class.pred <- predict(rf.model, 
                              baked_test, 
                              type = "raw") 

rfmodel.scoring <- predict(rf.model, 
                           baked_test, 
                           type = "prob") [, "Satisfied"] 

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
rfmodel.conf[[3]][[1]]
rfmodel.conf[[3]][[2]]
rfmodel.conf[[4]][[5]]
rfmodel.conf[[4]][[6]]
rfmodel.conf[[4]][[2]]
rf.auc

# 3) xgboost
model.xgboost <- train(Satisfied ~ ., baked_train,
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
real.pred <- baked_test$Satisfied 
xgb.class.pred <- predict(model.xgboost, 
                          baked_test, 
                          type = "raw") 
xgb.scoring <- predict(model.xgboost, 
                       baked_test, 
                       type = "prob") [, "Satisfied"] 
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
library(rsample)
library(recipes)
set.seed(123)

# Randomly select 50% of the rows from the data
data_employee_half <- data_employee[sample(nrow(data_employee), nrow(data_employee) * 0.15), ]

split <- initial_split(data_employee_half, prop = 0.8, strata = "Satisfied") 

employee_train <- training(split)
employee_test <- testing(split)

employee_recipe <- recipe(Satisfied ~ ., data = employee_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_integer_predictors()) %>%
  step_scale(all_integer(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)

train.param <-  trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(5628)  
svm_rad <- train(Satisfied ~ ., baked_train,
                 method = "svmRadial",               
                 metric = "Sens",  # kappa not available
                 trControl = train.param,
                 tuneLength = 20)
svm_rad

# confusion matrix + KAPPA 
real.pred <- baked_test$Satisfied #
svm.class.pred <- predict(svm_rad, baked_test, type = "raw") 
svm.scoring <- predict(svm_rad, baked_test, type = "prob") [, "Satisfied"] 

svm.conf <- confusionMatrix(data = svm.class.pred, reference = real.pred, positive = "Satisfied", mode = "prec_recall") 

# ROC and AUC
svm.auc = colAUC(svm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, data.frame(model = 'svm', accuracy = svm.conf[[3]][[1]], kappa = svm.conf[[3]][[2]], precision = svm.conf[[4]][[5]], recall_sensitivity = svm.conf[[4]][[6]], specificity = svm.conf[[4]][[2]], auc = svm.auc))






