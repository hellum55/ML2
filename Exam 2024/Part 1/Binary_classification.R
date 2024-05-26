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
job_satisfaction[data_employee$ratingOverall >= 4] = "1"
job_satisfaction[data_employee$ratingOverall <= 3] = "0"
data_employee=data.frame(data_employee,job_satisfaction)
#Removing ratingOverall variable
data_employee <- subset(data_employee, select = -ratingOverall)
data_employee$job_satisfaction <- factor(data_employee$job_satisfaction)

#Check the distribution of the target variable
prop.table(table(data_employee$job_satisfaction))
#        0         1 
# 0.3492645 0.6507355 


#Check the variable types of the variables:
str(data_employee)
# Convert the categorical variables into factors for the analysis:
data_employee$job_satisfaction <- factor(data_employee$job_satisfaction)
data_employee$ratingBusinessOutlook <- factor(data_employee$ratingBusinessOutlook)
data_employee$ratingCeo <- factor(data_employee$ratingCeo)
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

#If the target variable is binary we can make sure that the test split is balanced instead of stratified
#This can only be done with 2 levels. It ensures that the data are balanced.
library(ROSE)
employee_train <- ovun.sample(job_satisfaction~., data=employee_train, 
                              p=0.5, seed=2, 
                              method="over")$data

employee_test <- ovun.sample(job_satisfaction~., data=employee_train, 
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
  step_dummy(all_nominal_predictors(), one_hot = T) %>%
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

#Using caret to show Accuracy. It applies 5fold-CV with 20 alpha parameters to tune on. Lower alphas = deeper trees, and 
#helps to minimize error. There is a maximum of 10 trees and a mini
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
ggplot(tree.model)

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
tree.class.pred <- predict(tree.model, 
                           baked_train, type = "raw") 

tree.scoring <- predict(tree.model, 
                        baked_train, 
                        type = "prob")[,"1"]


tree.conf <- confusionMatrix(tree.class.pred, real.pred,
                             positive = "1",
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
# create a tuning grid. The degree of freedom ranges from 1 to 3, where 3 is the maximum flexibility. Setting this higher could lead to overfitting, because it
#would fit the training data too good. The nprune argument decides how many terms the best model has, in combination with the degree of freedom.
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
## 13     23      1
#The best tune has 23 terms and only one degree, so no interactions between variables.

cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)
#degree nprune  Accuracy     Kappa  AccuracySD    KappaSD
#1    1     23 0.8838859 0.7677374 0.006360627 0.01274356

ggplot(cv_mars)
#The model is performing the best with around 12-13 terms, so a rather small model actually.
# variable importance plots
library(vip)       # for variable importance
library(pdp)       # for variable relationships
p1 <- vip(cv_mars, num_features = 20, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 20, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)
#From the feature plot we can tell that LentghOfEmployment has an importance of zero, and might not be used in any of the MARS functions.
#The GCV plot shows that, Recommend to a friend is very important both in the reduction of GCV and the reduction in rss. 
#The feature importance plots will not show the interactions/hinge-functions between variables but only the prediction error effect. 
# Construct partial dependence plots
p1 <- partial(cv_mars, pred.var = "ratingRecommendToFriend_NEGATIVE", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(cv_mars, pred.var = "ratingSeniorLeadership", grid.resolution = 10) %>% 
  autoplot()
p3 <- partial(cv_mars, pred.var = c("ratingRecommendToFriend_NEGATIVE", "ratingSeniorLeadership"), 
              grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
marsmodel.class.pred <- predict(cv_mars, 
                                baked_train, 
                                type = "raw")

marsmodel.scoring <- predict(cv_mars, 
                             baked_train, 
                             type = "prob")[, "1"]

marsmodel.conf <- confusionMatrix(marsmodel.class.pred, real.pred,
                                  positive = "1", 
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
#The bagging model are combining 100 trees. It might alright or too few trees to bag to see where the error levels off. 
bag_model <- train(
  job_satisfaction ~ .,
  data = baked_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5), # 5-fold CV increases the convergence time
  nbagg = 10,  
  control = rpart.control(minsplit = 2, cp = 0)
)

bag_model

# View results
bag_model$results
#parameter  Accuracy     Kappa AccuracySD    KappaSD
#1      none 0.9577544 0.9155159 0.00576135 0.01152046

#feature importance
vip::vip(bag_model, num_features = 20)
#for the bagging model it is senior leadership and culture and values that are the most important predictors. 

# PDPs
# Construct partial dependence plots
p1 <- pdp::partial(
  bag_model, 
  pred.var = "ratingSeniorLeadership",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  bag_model, 
  pred.var = "ratingCultureAndValues", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
bag_model.class.pred <- predict(bag_model, 
                                baked_train, 
                                type = "raw") 

bag_model.scoring <- predict(bag_model, 
                             baked_train, 
                             type = "prob")[,"1"]

bag_model.conf <- confusionMatrix(bag_model.class.pred, real.pred,
                                  positive = "1", 
                                  mode = "prec_recall")

bag_model.conf
#We might have overfitted the training data when we can get an accuracy of 99%. The bagging model does not penalize very complex tress, and it looks like the model
#has come up with very specific and complex to almost capture all the 1's and 0's.

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
tunegrid <- expand.grid(.mtry = (1:6), min.node.size = c(1, 3, 5, 10))
train.param <- trainControl(method = "cv", number = 5)

rf.model <- train(job_satisfaction ~ ., baked_train,
                  method = "rf", 
                  ntree = 200,
                  metric = "Accuracy",
                  trControl = train.param,
                  tune.grid = tunegrid)

rf.model

#Feature importance:
p1 <- vip::vip(rf.model, num_features = 25, bar = FALSE)
p2 <- vip::vip(rf.model, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1, p2, nrow = 1)
#I guess the importance when talking about decision trees relates to the factor of impurity. A lower impurity factor for variables indicates how good the
#variable can split up the data into 1's and 0's. If a leaf only contains 1's and no zeros the impurity level will be equal to 0.

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction 
rfmodel.class.pred <- predict(rf.model, 
                              baked_train, 
                              type = "raw") 

rfmodel.scoring <- predict(rf.model, 
                           baked_train, 
                           type = "prob")[,"1"]

rfmodel.conf <- confusionMatrix(rfmodel.class.pred, real.pred,
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
# Gradient boosting --------------------------------------------------------------------------------------------------------------------------
library(caret) 

set.seed(123)  # for reproducibility
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 1)
#The total number of trees in a sequence is set to 1000. It can set to much higher but  the GBM can overfit,
#but it is to costly for the computer. The tree fixes the last trees errors. A optimal number must be found so it does not overfit or underfit.
#The learning rate is set to 0.1 so it does not learn slowly but nor really quick. The depth of the trees are set to 3 which can be efficient, but there is a
#trade off, where it captures too much of the data to not be able to generalize, but catches too little of the complex data set.
#The minimum observation of each node is 10. Also controls the complexity.

gbmGrid <-  expand.grid(interaction.depth = c(3), 
                        n.trees = 1000, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

set.seed(825)
model_gbm <- train(job_satisfaction ~ ., data = baked_train, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
# View results
model_gbm$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
# 1500                 9       0.1             20
ggplot(model_gbm)

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction
gbm.class.pred <- predict(model_gbm, 
                          baked_train, 
                          type = "raw") 
xgb.scoring <- predict(model_gbm, 
                       baked_train, 
                       type = "prob")[, "1"] 
xgb.conf <- confusionMatrix(data = gbm.class.pred, 
                            reference = real.pred, 
                            positive = "1", 
                            mode = "prec_recall") 

# ROC and AUC
gbm.auc = colAUC(gbm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, 
                        data.frame(model = 'xgb tree', 
                                   accuracy = gbm.conf[[3]][[1]], 
                                   kappa = gbm.conf[[3]][[2]], 
                                   precision = gbm.conf[[4]][[5]], 
                                   recall_sensitivity = gbm.conf[[4]][[6]], 
                                   specificity = gbm.conf[[4]][[2]], 
                                   auc = gbm.auc))


# xgboost -------------------------------------------------------------------------------------------------------------------------------------
#The hyperparamters are:

#The depth of each trees is set between 3 and 6 to reduce the complexity of each tree. 
#Gamma is a regularization strategy to reduce complexity. It specifies a minimum loss reduction required to make another node/leaf. XGBoost will grow the tree to the max
#and then pruning the tree and remove splits that do not meet the gamma.
#alpha/LR: Controls how quickly the tree proceeds and learns. Smaller values make the model robust, but can be stuck at a global minimum. More robust to overfitting.
#Only eta/alpha is specified which means it is a L1 regularization which wil minimize the weights of the features.
#Training on a 0.5 subsample of the rows of the data set = lower tree correlation, higher accuracy.
# 0.1 subsample on the columns. 
#N_rounds = number of trees

train.param <- trainControl(method = "cv", number = 5)
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

# View results
model.xgboost$bestTune
#nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#64 300      3     0.2     3              0.1                1       0.5

#Feature importance:
p1 <- vip::vip(model.xgboost, num_features = 25, bar = FALSE)
p2 <- vip::vip(model.xgboost, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1, p2, nrow = 1)

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
library(caret)
set.seed(123)

# Randomly select 15% of the rows from the data
data_employee_small <- data_employee[sample(nrow(data_employee), nrow(data_employee) * 0.15), ]

train.param <-  trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Tune an SVM with radial basis kernel 
set.seed(1854)  
employee_svm <- train(
  job_satisfaction ~ ., 
  data = baked_train,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 5
)
employee_svm$results
ggplot(employee_svm)
confusionMatrix(employee_svm)

# Print results
employee_svm$results 
#sigma    C  Accuracy     Kappa AccuracySD    KappaSD
#1 0.05698483 0.25 0.8761052 0.7521935 0.00743444 0.01487482
#the model yields a low sigma and a low budget function (C). this implies that the model uses a larger margin. It becomes relatively flexible and potentially misclassify more.
#the sigma is the decision boundary for the predictions. The sigma is quite low resulting in decision boundaries which is smoother and considers observations that is far away.
#It seems like the prediction are easier to detect due to a low C and a low sigma value.

# Feature importance 
# Create a wrapper to retain the predicted class probabilities for the class of interest (in this case, Yes)
library(kernlab)  # also for fitting SVMs 

# Model interpretability packages
library(pdp)      # for partial dependence plots, etc.
library(vip)      # for variable importance plots
library(modeldata) # for data set Job attrition
baked_train$job_satisfaction <- as.factor(baked_train$job_satisfaction)
baked_test$job_satisfaction <- as.factor(baked_test$job_satisfaction)
levels(baked_train$job_satisfaction) <- c("Yes", "No") 
levels(baked_test$job_satisfaction) <- c("Yes", "No") 


prob_satisfied <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

#variable importance plot
set.seed(2827)  # for reproducibility
vip::vip(employee_svm, method = "permute", event_level = "first", nsim = 5, train = baked_train, 
    target = "job_satisfaction", metric = "roc_auc", reference_class = "Yes", 
    pred_wrapper = prob_satisfied)

#construct PDP (feature effect plots are on the probability scale)
features <- c("ratingCareerOpportunities", "ratingCultureAndValues", 
              "ratingSeniorLeadership", "ratingRecommendToFriend_NEGATIVE")
pdps <- lapply(features, function(x) {
  partial(employee_svm, pred.var = x, which.class = 1,        # since the predicted probabilities from our model come in two columns (No and Yes), we specify which.class = 2 so that our interpretation is in reference to predicting Yes
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})
gridExtra::grid.arrange(grobs = pdps,  ncol = 2)
# interpret

# confusion matrix + KAPPA 
real.pred <- baked_train$job_satisfaction #
svm.class.pred <- predict(employee_svm, baked_train, type = "raw") 
svm.scoring <- predict(employee_svm, baked_train, type = "prob") [, "Yes"] 

svm.conf <- confusionMatrix(data = svm.class.pred, reference = real.pred, positive = "Yes", mode = "prec_recall") 

# ROC and AUC
svm.auc = colAUC(svm.scoring, real.pred, plotROC = TRUE) 

#Save results
modelComparison = rbind(modelComparison, data.frame(model = 'svm', accuracy = svm.conf[[3]][[1]], kappa = svm.conf[[3]][[2]], precision = svm.conf[[4]][[5]], recall_sensitivity = svm.conf[[4]][[6]], specificity = svm.conf[[4]][[2]], auc = svm.auc))
modelComparison

#The model with the highest and also highest kappa coefficients is the random forest model. with 99% accuracy and 98%, which is quite crazy. We might need some more data.
#The kappa coefficients is an estimate of how much better the model are rather than just random guessing. Lets have a look on how well it does on the test set:
# ROC plot based on caTools library; AUC is displayed in the console
library(caTools)
par(mfrow=c(1,1))
satisfaction_prob <- predict(employee_svm, baked_test, type = "prob")$job_satisfaction
colAUC(satisfaction_prob, baked_test_small$job_satisfaction, plotROC = TRUE)

#Question d: ####
library(ROCR)
prob <- predict(employee_svm, baked_test, type = "prob")$Yes  # predicted prob.
perf <- prediction(prob, baked_test$job_satisfaction) 
roc_ROCR <- performance(perf, measure = "tpr", x.measure = "fpr")
plot(roc_ROCR, col = "red", lty = 2, main = "ROC curve")
abline(a = 0, b = 1)

#When we are looking at the ROC-curve it is simply the best to reach the left top corner where we predict all true positive. The important thing
#is to find a trade off between true positive rate and the false positive rate. As mentioned, if it very important that we predict all
#above classes correclty 
prob <- predict(employee_svm, baked_test, type = "prob")$No
probabilities = as.data.frame(prob)

# create empty accuracy table
accT = c()
# compute accuracy per cutoff
for (cut in seq (0, 1, 0.1)) {
  cm <- confusionMatrix(factor(ifelse(probabilities  > cut,"No", "Yes")), 
                        factor(baked_test$job_satisfaction))
  accT = c(accT, cm$overall[1])}

# plot 
plot(accT ~ seq(0,1, 0.1), xlab= "Cutoff value", ylab ="", type="l", ylim=c(0,1))
lines (1-accT ~ seq(0,1, 0.1), type = "l", lty=2)
legend ("center", c("accuracy", "overall error"), lty  = c(1,2), merge = TRUE)
#The accuracy for cutoff values from 0-0.8(ish) are pretty much the same. When the cutoff value exceeds 0.8-0.85, the accuracy begind to drop¨
#Which is not preferable because we wan a model that has the highest accuracy. 



```

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

remove.packages("h2o")
install.packages("h2o")

library(h2o)
h2o.init()

