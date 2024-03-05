# Case study: "Tous" 
# Machine Learning for BI II
# Lecturer: Ana Alina Tudoran
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



#Data
dataset <- read.csv("~/Cand. Merc./ML2/Support Vector Machines/DatasetTous.csv")
str(dataset)
dataset$best_seller <- as.factor(dataset$best_seller)


# data partitioning
set.seed(1337)
sample = sample.split(dataset$best_seller, SplitRatio = 0.7)
train.data = subset(dataset, sample == TRUE)
test.data = subset(dataset, sample == FALSE)

# unbalanced
prop.table(table(train.data$best_seller))

# oversampling
train.data <- ovun.sample(best_seller~., data=train.data, 
                                  p=0.5, seed=2, 
                                  method="over")$data

prop.table(table(train.data$best_seller)) # more balanced after oversampling
prop.table(table(test.data$best_seller))
# reflection here (why is balancing necessary; 
# how it helps later in evaluating accuracy; why not used for testing data)

str(train.data)
str(test.data)

#### Classification

# Defining the resampling method 
train.param <- trainControl(method = "cv", number = 10)

modelComparison = NULL 
modelComparison = data.frame() 

# 1) Logistic regression -----------------------------------------------------
log.model <- train(best_seller~.,
                   data = train.data, 
                   trControl = train.param,
                   method = "glm",
                   metric = "kappa",
                   preProcess = c("nzv", "center", "scale"))

log.model
summary(log.model)

# confusion matrix + KAPPA 
real.pred <- test.data$best_seller
log.pred  <- predict(log.model, 
                     test.data, 
                     type = "raw") # predicted class
log.scoring <- predict(log.model, 
                       test.data, 
                       type = "prob") [,"1"] # predicted probability
log.conf   <- confusionMatrix(data = log.pred, 
                              reference = real.pred, 
                              positive = "1", 
                              mode = "prec_recall")

# ROC and AUC
log.auc = colAUC(log.scoring , 
                 real.pred, 
                 plotROC = TRUE)
log.auc

# Saved LR result
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Logistic', 
                                   accuracy = log.conf[[3]][[1]], 
                                   kappa = log.conf[[3]][[2]], 
                                   precision = log.conf[[4]][[5]], 
                                   recall_sensitivity = log.conf[[4]][[6]], 
                                   specificity = log.conf[[4]][[2]], 
                                   auc = log.auc))

# 2) GLM Regularized (Elastic Net)----------------------------------------------
glmnet.model <- train(best_seller ~., 
                      data = train.data, 
                      trControl = train.param,
                      method = "glmnet",
                      metric = "kappa",
                      preProcess = c("nzv", "center", "scale"),
                      tuneGrid = expand.grid(alpha = seq(0, 1, length=5), 
                                             lambda = seq(0.0001, 1, length = 20)))

# confusion matrix + KAPPA 
real.pred   <- test.data$best_seller 
glmnet.pred    <- predict(glmnet.model, 
                          test.data, 
                          type = "raw") # predicted class
glmnet.scoring <- predict(glmnet.model, 
                          test.data, 
                          type = "prob")[,"1"] # predicted probability
glmnet.conf   <- confusionMatrix(data = glmnet.pred, 
                                 reference = real.pred, 
                                 positive = "1", 
                                 mode = "prec_recall")

# ROC and AUC
glmnet.auc = colAUC(glmnet.scoring , 
                    real.pred, 
                    plotROC = TRUE) 
glmnet.auc

# the predictors
coef(glmnet.model$finalModel)

# saved result
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Elastic net', 
                                   accuracy = glmnet.conf[[3]][[1]], 
                                   kappa = glmnet.conf[[3]][[2]], 
                                   precision = glmnet.conf[[4]][[5]], 
                                   recall_sensitivity = glmnet.conf[[4]][[6]], 
                                   specificity = glmnet.conf[[4]][[2]], 
                                   auc = glmnet.auc))

# GLM Regularized (Ridge)-------------------------------------------------------
ridge.model <- train(best_seller~., 
                      data = train.data, 
                      trControl = train.param,
                      method = "glmnet",
                      metric = "Kappa",
                      preProcess = c("nzv", "center", "scale"),
                      tuneGrid = expand.grid(alpha = 0, 
                                             lambda = seq(0.0001, 1, length = 20)))

# confusion matrix + KAPPA 
real.pred   <- test.data$best_seller 
ridge.pred    <- predict(ridge.model, 
                         test.data, 
                         type = "raw") 
ridge.scoring <- predict(ridge.model, 
                         test.data, 
                         type = "prob")[,"1"] 

ridge.conf   <- confusionMatrix(data = ridge.pred, 
                                reference = real.pred, 
                                positive = "1", 
                                mode = "prec_recall")

# ROC and AUC
ridge.auc = colAUC(ridge.scoring , 
                   real.pred, 
                   plotROC = TRUE) 

# Saved result
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Ridge regression', 
                                   accuracy = ridge.conf[[3]][[1]], 
                                   kappa = ridge.conf[[3]][[2]], 
                                   precision = ridge.conf[[4]][[5]], 
                                   recall_sensitivity = ridge.conf[[4]][[6]], 
                                   specificity = ridge.conf[[4]][[2]], 
                                   auc = ridge.auc))


# GLM Regularized (Lasso)-------------------------------------------------------
lasso.model <- train(best_seller ~., 
                      data = train.data, 
                      trControl = train.param,
                      method = "glmnet",
                      metric = "Kappa",
                      preProcess = c("nzv", "center", "scale"),
                      tuneGrid = expand.grid(alpha = 1, 
                                             lambda = seq(0.0001, 1, length = 20)))
coef(glmnet.model$finalModel)

# confusion matrix + KAPPA 
real.pred   <- test.data$best_seller 
lasso.pred    <- predict(lasso.model, 
                         test.data[,-1], 
                         type = "raw") 
lasso.scoring <- predict(lasso.model, 
                         test.data, 
                         type = "prob")[, "1"] 
lasso.conf   <- confusionMatrix(data = lasso.pred, 
                                reference = real.pred, 
                                positive = "1", 
                                mode = "prec_recall")

# ROC and AUC
lasso.auc = colAUC(lasso.scoring, real.pred, plotROC = TRUE) 
lasso.auc

# Saved result
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'Lasso', 
                                   accuracy = lasso.conf[[3]][[1]], 
                                   kappa = lasso.conf[[3]][[2]], 
                                   precision = lasso.conf[[4]][[5]], 
                                   recall_sensitivity = lasso.conf[[4]][[6]], 
                                   specificity = lasso.conf[[4]][[2]], 
                                   auc = lasso.auc))


# Plots 
# plot(glmnet.model$finalModel, xvar ="lam", label = "TRUE")
# plot(ridge.model$finalModel, xvar ="lam", label = "TRUE")
# plot(lasso.model$finalModel, xvar ="lam", label = "TRUE")



# 3) Decision trees ------------------------------------------------------------

# training parameters
train.param <- trainControl(method = "cv", number = 5)

# 1) basic decision tree
tune.grid <- data.frame(.maxdepth = 3:10, 
                        .mincriterion = c(.1, .2, .3, .4))
tree.model <- train(best_seller ~., train.data,
                    method = "ctree2",
                    metric = "Kappa",
                    preProcess = c("nzv", "center", "scale"),
                    trControl = train.param,
                    tuneGrid = tune.grid) 

tree.model

# confusion matrix + KAPPA 
real.pred <- test.data$best_seller 
tree.class.pred <- predict(tree.model, 
                           test.data, type = "raw") 
tree.scoring <- predict(tree.model, 
                        test.data, 
                        type = "prob") [, "1"] 
tree.conf <- confusionMatrix(data = tree.class.pred, 
                             reference = real.pred, 
                             positive = "1", 
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


# 2) random forest
rf.model <- train(best_seller ~ ., train.data,
                    method = "rf", 
                    ntree = 1000,
                    metric = "Kappa",
                    trControl = train.param,
                    tune.grid = expand.grid(.mtry=c(1:28)))

rf.model

# confusion matrix + KAPPA 
real.pred <- test.data$best_seller 
rfmodel.class.pred <- predict(rf.model, 
                              test.data, 
                              type = "raw") 
rfmodel.scoring <- predict(rf.model, 
                           test.data, 
                           type = "prob") [, "1"] 
rfmodel.conf <- confusionMatrix(data = rfmodel.class.pred, 
                                reference = real.pred, 
                                positive = "1", 
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

# 3) xgboost
model.xgboost <- train(best_seller ~ ., train.data,
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
real.pred <- test.data$best_seller 
xgb.class.pred <- predict(model.xgboost, 
                          test.data, 
                          type = "raw") 
xgb.scoring <- predict(model.xgboost, 
                       test.data, 
                       type = "prob") [, "1"] 
xgb.conf <- confusionMatrix(data = xgb.class.pred, 
                            reference = real.pred, 
                            positive = "1", 
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


# 4) SVM ---------
# 
train.data$best_seller <- as.factor(train.data$best_seller)
test.data$best_seller <- as.factor(test.data$best_seller)
levels(train.data$best_seller) <- c("No", "Yes") 
levels(test.data$best_seller) <- c("No", "Yes")


train.param <-  trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(5628)  
svm_rad <- train(best_seller ~ ., train.data,
                       method = "svmRadial",               
                       preProcess = c("center", "scale"),  
                       metric = "Sens",  # kappa not available
                       trControl = train.param,
                       tuneLength = 20)
svm_rad

# confusion matrix + KAPPA 
real.pred <- test.data$best_seller #
svm.class.pred <- predict(svm_rad, test.data, type = "raw") 
svm.scoring <- predict(svm_rad, test.data, type = "prob") [, "Yes"] 

svm.conf <- confusionMatrix(data = svm.class.pred, reference = real.pred, positive = "Yes", mode = "prec_recall") 

# ROC and AUC
svm.auc = colAUC(svm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, data.frame(model = 'svm', accuracy = svm.conf[[3]][[1]], kappa = svm.conf[[3]][[2]], precision = svm.conf[[4]][[5]], recall_sensitivity = svm.conf[[4]][[6]], specificity = svm.conf[[4]][[2]], auc = svm.auc))






