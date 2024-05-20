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

# I select the columns i want to convert to factors
str(data_employee)
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
plot_bar(data_employee)
#There is no need to look at JobTitle and LocationName
#Overall it looks fine. Continue!


########## We might need to do this #########

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

####################################### Non-linearity ######################################################################
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

############################################### DECISION TREES ##########################################################
library(rpart)       # direct engine for decision tree application
library(caret)       # meta engine for decision tree application

# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects
train.param <- trainControl(method = "cv", number = 5)

tree.model <- train(ratingOverall ~., baked_train,
                    method = "rpart",
                    trControl = train.param,
                    tuneLength = 20) 
tree.model

#Lets have a look on the feature importance:
vip(tree.model, num_features = 14, bar = FALSE)
#The top scoring features are ratings on senior leadership, career opportunities, which is the features that results in the biggest reduction on the loss function.

# Construct partial dependence plots
p1 <- partial(tree.model, pred.var = "ratingSeniorLeadership") %>% autoplot()
p2 <- partial(tree.model, pred.var = "ratingCareerOpportunities") %>% autoplot()
p3 <- partial(tree.model, pred.var = c("ratingSeniorLeadership", "ratingCareerOpportunities")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, 
              colorkey = TRUE, screen = list(z = -20, x = -60))
# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

############################################ BAGGING ###################################################################
bag_model <- train(
  ratingOverall ~ .,
  data = baked_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 5), # 10-fold CV increases the convergence time
  nbagg = 100,  
  control = rpart.control(minsplit = 2, cp = 0)
)
bag_model

#importance
vip::vip(bag_model, num_features = 40, bar = FALSE)

# PDPs
# Construct partial dependence plots
p1 <- pdp::partial(
  bag_model, 
  pred.var = "Lot_Area",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  bag_model, 
  pred.var = "Lot_Frontage", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)

############################################# RANDOM FOREST #############################################################
#create tunegrid with 15 values from 1:15 for mtry to tunning model. Our train function will change number of entry variable at each split according to tunegrid. 
tunegrid <- expand.grid(.mtry = (1:15))
train.param <- trainControl(method = "cv", number = 5)

rf.model <- train(ratingOverall ~ ., baked_train,
                  method = "rf", 
                  ntree = 100,
                  metric = "RMSE",
                  trControl = train.param,
                  tune.grid = tunegrid)

rf.model

######################################## XGBOOST ########################################################################
#The hyperparamters are:
#Tree-depth: That controls the depth of the tree. Smaller trees are computed faster. Higher trees can capture unique interactions. (min3, max6)
#alpha/LR: Controls how quicly the tree proceeds and learns. Smaller vales makes the model robust, but can be stuck at a global minimum. More robust to overfitting.
#Training on a 0.5 subsample of the rows of the data set = lower tree correlation, higher accuracy.
# 0.1 subsample on the columns. 
#Gamma is a regularization strategy to reduce complexity. It specifies a minimum loss reduction required to make another node/leaf. XGBoost will grow the tree to the max
#and then pruning the tree and remove splits that do not meet the gamma.
#N_rounds = number of trees
train.param <- trainControl(method = "cv", number = 5)
model.xgboost <- train(ratingOverall ~ ., baked_train,
                       method = "xgbTree",
                       metric = "RMSE",
                       tuneGrid = expand.grid(max_depth=3:6,
                                              gamma=c(0, 1, 2, 3), 
                                              eta=c(0.03, 0.06, 0.1), 
                                              nrounds=30,
                                              subsample=0.5, 
                                              colsample_bytree=0.1, 
                                              min_child_weight = 1),
                       trControl = train.param)
model.xgboost

########################################################## SVM #######################################################################
#Use only a part of the data
library(rsample)
library(recipes)
set.seed(123)

# Randomly select 50% of the rows from the data
data_employee_small <- data_employee[sample(nrow(data_employee), nrow(data_employee) * 0.15), ]

split_small <- initial_split(data_employee_small, prop = 0.7, strata = "ratingOverall") 

employee_train_small <- training(split_small)
employee_test_small <- testing(split_small)

employee_recipe_small <- recipe(ratingOverall ~ ., data = employee_train_small) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare_small <- prep(employee_recipe_small, training = employee_train_small)
prepare_small$steps

baked_train_small <- bake(prepare_small, new_data = employee_train_small)
baked_test_small <- bake(prepare_small, new_data = employee_test_small)

train.param <-  trainControl(method = "cv", number = 5)

set.seed(5628)  
svm_rad <- train(ratingOverall ~ ., baked_train_small,
                 method = "svmRadial",               
                 metric = "RMSE",  # kappa not available
                 trControl = train.param,
                 tuneLength = 10)
svm_rad
