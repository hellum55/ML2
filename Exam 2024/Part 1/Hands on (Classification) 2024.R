# Machine Learning for BI II
##################################### LOAD THE LIBRARIES ####################################
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

######################################## DATA PREPROCESSING #################################################
library(readxl)
# Read the Excel file with specified column types
data_employee <- read_xls('Data.xls', 
                          na = c("", " ", "NA", "N/A", ".", "NaN", "MISSING"))
str(data_employee)
head(data_employee)
glimpse(data_employee)

#Lets check for missing variables for a start
library(visdat)
library(DataExplorer)
sapply(data_employee, function(x) sum(is.na(x)))
sum(is.na(data_employee))
plot_missing(data_employee)
#42086 observaions are missing. JObEndingYear has 61% missing values. When looking at the data it looks like
#there is a '1' of the employee quit and a 'NA' if the employee stayed. RatingCeo and BusinessOutlook have
#44.05 and 43.83% respectively. The plot_missing function suggests that it is bad and we should remove them.
#Lets see later if that is necessary. 

#Lets remove some of the variables we are sure are not giving anything to the analysis. For example ID variables and timestamps:
library(dplyr)
data_employee <- data_employee %>%
  select(-c(reviewId, reviewDateTime))

#In case of transforming a variable into binary.Lets convert the isCurrentJob variable into 1 and 0
data_employee$isCurrentJob <- ifelse(is.na(data_employee$isCurrentJob), 0, data_employee$isCurrentJob)
#data_employee$isCurrentJob <-ifelse(data_employee$isCurrentJob == "1",1,0)

# I select the columns i want to convert to factors
str(data_employee)
data_employee$ratingOverall <- factor(data_employee$ratingOverall)
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

#The overall task to do is to predict the overall rating of the company using the variables that are necessary. It is a large data set so it
#is important to pay attention to the feature engineering steps, so we have the best data to analyse on.

#Lets look at the data:
summary(data_employee)
#When looking at the data we can see that IsCurrentJob only has the observation of 1. We might delete this one. 
data_employee <- data_employee %>%
  select(-c(jobEndingYear, jobTitle.text, location.name))

#Lets look at the target variable and its distribution
par(mfrow=c(1,4))
hist(data_employee$ratingOverall, breaks = 20, col = "red", border = "red") 
#A bit left skewed. It is a good overall rating for the company with most observations within rating 4 and 5.
mean(data_employee$ratingOverall)
#The company has an overall rating of 3.75 - mediocre.

#Maybe we can normalize the responses to a more uniformly distribution
no_trans <- data_employee$ratingOverall
#Log transformation:
log_employee <- log(data_employee$ratingOverall)
#Yeo-Johnson
library(car)
employee_YJ <- yjPower(data_employee$ratingOverall, lambda = 0.5)
# Box-Cox transformation (lambda=0 is equivalent to log(x))
library(forecast)
employee_BC <- forecast::BoxCox(data_employee$ratingOverall, lambda="auto") 

hist(log_employee, breaks = 20, 
     col = "lightgreen", border = "lightgreen",
     ylim = c(0, 6000) )
hist(employee_BC, breaks = 20, 
     col = "lightblue", border = "lightblue", 
     ylim = c(0, 6000))
hist(employee_YJ, breaks = 20, 
     col = "black", border = "black", 
     ylim = c(0, 6000))

list(summary(no_trans),
     summary(log_employee),
     summary(employee_BC),
     summary(employee_YJ))
#Does not really seem to be worth normalizing the target variable. We can experiment when we get to the analysis, but the difference between the
#transformations are not significant. 

#We check for near zero variance 
library(caret)
caret::nearZeroVar(data_employee, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)
#IsCurentJob has zero variance, but the variable has not been transformed yet into 0, 1. Right now it only conatins 0's.

#Lets se the different correlations between the numeric variables.
library(corrplot)
num <- c("ratingOverall", "ratingWorkLifeBalance", "ratingCultureAndValues", "ratingDiversityAndInclusion", "ratingSeniorLeadership",
         "ratingCareerOpportunities", "ratingCompensationAndBenefits", "lengthOfEmployment")
cm <- cor(data_employee[num], use = "complete.obs")
cm
corrplot(cm)
corrplot(cm, type="upper", tl.pos="d", method="number", diag=TRUE)
#It is obvioes that many variables are correlated. And many variables are really strong correlated. It might be problematic, but it makes perfectly sense.
#It would be weird if one employee rates the culture 5, and the diveristyInclusion 5, and then WorkLife balance 1. Of course they are related. Whit is great
#to see is that the taret variable does not have a high correlation to any of the variables. We might consider deleting LenghtOfEmployment because it does have
#any correlation with the target variable. The correlation is -0.04. Not much.

#Lets look at the factors
library(DataExplorer)
plot_bar(data_employee)
#There is no need to look at JobTitle and LocationName
#Overall it looks fine. Continue!


########## We might need to do this #########

#It is a bit odd if we are expected to classify a multiclass model with 5 different outcomes. It seems really difficult
# so i will convert the ratingOverall into a binary class. Whether they are satisfied (rating > 3) or not satisfied (rating <= 3).
#I have no idea id this is right...
data_employee$ratingOverall <- as.numeric(data_employee$ratingOverall)
Satisfied = rep(0, length(data_employee$ratingOverall))
Satisfied[data_employee$ratingOverall >= 4] = "Satisfied"
Satisfied[data_employee$ratingOverall <= 3] = "Not.Satisfied"
data_employee=data.frame(data_employee,Satisfied)
str(data_employee)
data_employee$Satisfied = as.factor(data_employee$Satisfied)
#Remove ratingOverall
data_employee <- subset(data_employee, select = -ratingOverall)

############################################ BUILDING THE MODELS ###########################################################
#Create the recipe
library(rsample)
set.seed(123)
split <- initial_split(data_employee, prop = 0.8, strata = "ratingOverall") 

employee_train <- training(split)
employee_test <- testing(split)

# imbalanced
prop.table(table(employee_train$ratingOverall))
prop.table(table(employee_test$ratingOverall))
#It is a imbalanced data set with the majority of the observations => 3, which is good for the company but can be hard to predict on.

#If the target variable is binary we can make sure that the test split is balanced instead of stratisfied
#This can only be done with 2 levels. It ensures that the data are balanced.
library(ROSE)
employee_train <- ovun.sample(ratingOverall~., data=employee_train, 
                           p=0.5, seed=2, 
                           method="over")$data

prop.table(table(employee_train$Satisfied)) # more balanced after oversampling
prop.table(table(employee_test$Satisfied))
# reflection here (why is balancing necessary; 
#Later on it will help the relevance of the model. If the test set contains 80% of one class the model can simply just predict that outcome every time, but it
#will never be a good model, because it does not capture the other outcome, it just learns to predict one class. Not very nice!

library(recipes)
employee_recipe <- recipe(ratingOverall ~ ., data = employee_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)

str(baked_train)

#### Classification
modelComparison = NULL
modelComparison = data.frame() 

# Decision trees ------------------------------------------------------------
library(rpart)       # direct engine for decision tree application
library(caret)       # meta engine for decision tree application

# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects

#Lets have a look on the decision tree output:
dc_simple <- rpart(
  formula = ratingOverall ~ .,
  data    = baked_train,
  method  = "anova"
)
dc_simple
#What we can see from the DC output is that, we start with 13106 at the root node, adn the 
#first variable we split on is (the largest SSE reduction) is RatingRecommendToFried < 0.5, with
#3990 observations. On the 3 node it is the same variable RatingRecommendToFried with a mean of 4.18 in rating. which
#tells us that it is a very important variabel to predict the overall raring. 
#We can plot the tree:
rpart.plot(dc_simple)
plotcp(dc_simple)
#The rpart function by default performs 10fold-CV. The function also applies the cost complexity function (alpha),
#to prune the tree. The plot suggests to do 6 terminal nodes, because the dash in the dot reaches the horizontal line
#which means that it falls within 1SE. 

dc_simple1 <- rpart(
  formula = ratingOverall ~ .,
  data    = baked_train,
  method  = "anova", 
  control = list(cp = 0, xval = 6)
)
plotcp(dc_simple1)
abline(v = 11, lty = "dashed")

#Using caret to show Accuracy. It applies 10fold-CV with 20 alpha parameters to tune on. Lower alphas = deeper trees, and 
#helps to minimize error. 
dc_simple2 <- train(
  ratingOverall ~ .,
  data = baked_train,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 20
)
ggplot(dc_simple2)

#Lets have a look on the feature importance:
vip(dc_simple2, num_features = 14, bar = FALSE)

# training parameters. Here we apply 5fold-CV
train.param <- trainControl(method = "cv", number = 5)
#The top scoring features are ratings on senior leadership, career oppotunities, which is the features that results in the biggest reduction on the loss function.

# Construct partial dependence plots
p1 <- partial(dc_simple2, pred.var = "ratingSeniorLeadership") %>% autoplot()
p2 <- partial(dc_simple2, pred.var = "ratingCareerOpportunities") %>% autoplot()
p3 <- partial(dc_simple2, pred.var = c("ratingSeniorLeadership", "ratingCareerOpportunities")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, 
              colorkey = TRUE, screen = list(z = -20, x = -60))
# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

############################################# basic decision tree #########################################################################
library(rpart)
library(caret)       # meta engine for decision tree application
tune.grid <- data.frame(.maxdepth = 3:10, 
                        .mincriterion = c(.1, .2, .3, .4))
# training parameters. Here we apply 5fold-CV
train.param <- trainControl(method = "cv", number = 5)

tree.model <- train(ratingOverall ~., baked_train,
                    method = "ctree2",
                    metric = "Kappa",
                    trControl = train.param,
                    tuneGrid = tune.grid) 
tree.model

# confusion matrix + KAPPA 
real.pred <- baked_test$ratingOverall 
tree.class.pred <- predict(tree.model, 
                           baked_test, type = "raw") 

tree.scoring <- predict(tree.model, 
                        baked_test, 
                        type = "prob")[,"1"]


tree.scoring[,]


tree.conf <- confusionMatrix(tree.class.pred, real.pred,
                positive = "ratingOverall",
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
# MARS model ------------------------------------------------------------------------------
# Cross-validated model
set.seed(123)  # for reproducibility
# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

cv_mars <- train(
  x = subset(baked_train, select = -ratingOverall),
  y = baked_train$ratingOverall,
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
#degree nprune  Accuracy     Kappa  AccuracySD     KappaSD
#    2     23 0.5699675 0.3995818 0.004975854 0.007533007

ggplot(cv_mars)

# confusion matrix + KAPPA 
real.pred <- baked_test$ratingOverall 
marsmodel.class.pred <- predict(cv_mars, 
                              baked_test, 
                              type = "raw")

marsmodel.scoring <- predict(cv_mars, 
                           baked_test, 
                           type = "prob")

marsmodel.conf <- confusionMatrix(marsmodel.class.pred, real.pred,
                                positive = "ratingOverall", 
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
marsmodel.conf[[4]][[2]]


# bagging ----------------------------------------------------------------------------------
bag_model <- train(
  ratingOverall ~ .,
  data = baked_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5), # 5-fold CV increases the convergence time
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)
bag_model

#importance
vip::vip(ames_bag2, num_features = 40, bar = FALSE)

# PDPs
# Construct partial dependence plots
p1 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Area",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Frontage", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)

# confusion matrix + KAPPA 
real.pred <- baked_test$ratingOverall 
bag_model.class.pred <- predict(bag_model, 
                              baked_test, 
                              type = "raw") 

bag_model.scoring <- predict(bag_model, 
                           baked_test, 
                           type = "prob")

bag_model.conf <- confusionMatrix(bag_model.class.pred, real.pred,
                                positive = "ratingOverall", 
                                mode = "prec_recall")

bag_model.conf
# ROC and AUC
rf.auc = colAUC(bag_model.scoring, real.pred, plotROC = TRUE) 


# random forest -----------------------------------------------------------------------------
library(ranger)   # a c++ implementation of random forest 
library(h2o)      # a java-based implementation of random forest
#create tunegrid with 15 values from 1:15 for mtry to tunning model. Our train function will change number of entry variable at each split according to tunegrid. 
tunegrid <- expand.grid(.mtry = (1:15))
train.param <- trainControl(method = "cv", number = 5)

rf.model <- train(ratingOverall ~ ., baked_train,
                  method = "rf", 
                  ntree = 1000,
                  metric = "Accuracy",
                  trControl = train.param,
                  tune.grid = tunegrid)

rf.model

# confusion matrix + KAPPA 
real.pred <- baked_test$ratingOverall 
rfmodel.class.pred <- predict(rf.model, 
                              baked_test, 
                              type = "raw") 

rfmodel.scoring <- predict(rf.model, 
                           baked_test, 
                           type = "prob")

rfmodel.conf <- confusionMatrix(rfmodel.class.pred, real.pred,
                                positive = "ratingOverall", 
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
rfmodel.conf[[4]][[2]]

# 3) xgboost
#The hyperparamters are:
#Tree-depth: That controls the depth of the tree. Smaller trees are computed faster. Higher trees can capture unique interactions. (min3, max6)
#alpha/LR: Controls how quicly the tree proceeds and learns. Smaller vales makes the model robust, but can be stuck at a global minimum. More robust to overfitting.
#Training on a 0.5 subsample of the rows of the data set = lower tree correlation, higher accuracy.
# 0.1 subsample on the columns. 
#Gamma is a regularization strategy to reduce complexity. It specifies a minimum loss reduction required to make another node/leaf. XGBoost will grow the tree to the max
#and then pruning the tree and remove splits that do not meet the gamma.
#N_rounds = number of trees
model.xgboost <- train(ratingOverall ~ ., baked_train,
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
real.pred <- baked_test$ratingOverall 
xgb.class.pred <- predict(model.xgboost, 
                          baked_test, 
                          type = "raw") 
xgb.scoring <- predict(model.xgboost, 
                       baked_test, 
                       type = "prob") 
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

xgb.conf


# 4) SVM -----------------------------------------------------------------------------------
#Use only a part of the data
library(rsample)
library(recipes)
set.seed(123)

# Randomly select 15% of the rows from the data
data_employee_small <- data_employee[sample(nrow(data_employee), nrow(data_employee) * 0.15), ]

split_small <- initial_split(data_employee_small, prop = 0.7, strata = "Satisfied") 

employee_train_small <- training(split_small)
employee_test_small <- testing(split_small)

employee_train_small <- ovun.sample(Satisfied~., data=employee_train_small, 
                              p=0.5, seed=2, 
                              method="over")$data

prop.table(table(employee_train_small$Satisfied)) # more balanced after oversampling
prop.table(table(employee_test_small$Satisfied))

employee_recipe_small <- recipe(Satisfied ~ ., data = employee_train_small) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare_small <- prep(employee_recipe_small, training = employee_train_small)
prepare_small$steps

baked_train_small <- bake(prepare_small, new_data = employee_train_small)
baked_test_small <- bake(prepare_small, new_data = employee_test_small)

train.param <-  trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(5628)  
svm_rad <- train(Satisfied ~ ., baked_train_small,
                 method = "svmRadial",               
                 metric = "Sens",  # kappa not available
                 trControl = train.param,
                 tuneLength = 10)
svm_rad

# confusion matrix + KAPPA 
real.pred <- baked_test_small$Satisfied #
svm.class.pred <- predict(svm_rad, baked_test_small, type = "raw") 
svm.scoring <- predict(svm_rad, baked_test_small, type = "prob") [, "Satisfied"] 

svm.conf <- confusionMatrix(data = svm.class.pred, reference = real.pred, positive = "Satisfied", mode = "prec_recall") 

# ROC and AUC
svm.auc = colAUC(svm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, data.frame(model = 'svm', accuracy = svm.conf[[3]][[1]], kappa = svm.conf[[3]][[2]], precision = svm.conf[[4]][[5]], recall_sensitivity = svm.conf[[4]][[6]], specificity = svm.conf[[4]][[2]], auc = svm.auc))
modelComparison






