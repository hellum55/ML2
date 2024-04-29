#Non Linear regression:
#This first part is the target variable is treated as continuous. Later it is treated as categorical.
#We start of by splitting the data into train and test sets, and then peform desired feature engineering steps
library(recipes)
library(rsample)
library(ggridges)
library(earth)
library(caret)

data_employee <- read.csv("data_employee.csv", stringsAsFactors=TRUE)
str(data_employee)

set.seed(123)
split <- initial_split(data_employee, prop = 0.7, strata = "ratingOverall") 

employee_train <- training(split)
employee_test <- testing(split)

employee_recipe <- recipe(ratingOverall ~ ., data = employee_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_integer_predictors()) %>%
  step_scale(all_integer(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)
dim(baked_train)
dim(baked_test)


# Fit a basic MARS model
mars1 <- earth(
  ratingOverall~.,  
  data = baked_train   
)
# Print model summary
print(mars1)
#Selected 16 of 17 terms, and 7 of 9 predictors
#Termination condition: Reached nk 21
#Importance: ratingCultureAndValues, ratingCareerOpportunities, ...
#Number of terms at each degree of interaction: 1 15 (additive model)
#GCV 0.6139388    RSS 7001.409    GRSq 0.55131    RSq 0.553655
summary(mars1) %>% .$coefficients %>% head(10)

#This means that the model considers 16 predictors including the intercept.
#Looking at the terms of the MARS model, we can see that CultureAndValues is included
#with a knot at -1.648 h(CultureAndValues - (-1.648)) the coefficents corresponding is 0.39.
plot(mars1, which = 1)
#The vertical line at 16 terms tells us the point where marginal increases in GCV R2 are less than 0.001

#We can also look at different interactions between the terms by including a degree = 2.

# Fit a basic MARS model
mars2 <- earth(
  ratingOverall ~ .,  
  data = baked_train,
  degree = 2
)
# check out the first 10 coefficient terms
summary(mars2) %>% .$coefficients %>% head(10)
#The model includes an interaction between WorkLifeBalance and SeniorLeadership.

#When it comes to tuning a MARS model we can either tune the maximum degree of interactions and/or
#and the number of terms in the final model. To do that we have to perform a grid search to get the
#optimal combination. We will perform a "real" CV grid-search. 

# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)
head(hyper_grid)

# Cross-validated model
set.seed(123)  # for reproducibility
cv_mars <- train(
  x = subset(baked_train, select = -ratingOverall),
  y = baked_train$ratingOverall,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid
)

# View results
cv_mars$bestTune
##    nprune degree
## 3     23      1

cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)
# degree nprune      RMSE  Rsquared       MAE     RMSESD RsquaredSD MAESD
#1      1     23 0.7820735 0.5535122 0.5775255 0.02690001 0.01742313 0.01373252
ggplot(cv_mars)
#More or less no effect. They follow each other so no significant effect here.

# variable importance plots
library(vip)
p1 <- vip(cv_mars, num_features = 10, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 10, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)
#The two most important variables are Leadership and Culture/values. Of course it is. A good working environment
#is the most important things to alot of people. If a company scores high on these two factors people feel good,
#and do not feel stressed about their work and tasks. 

#In this example we will classify the target variable ratingOverall. We will convert the variable to a factor. 
data_employee$ratingOverall <- as.factor(data_employee$ratingOverall)

#We will split the data again and convert into baked_train and baked_test.
set.seed(123)
split <- initial_split(data_employee, prop = 0.7, strata = "ratingOverall") 

employee_train <- training(split)
employee_test <- testing(split)

employee_recipe <- recipe(ratingOverall ~ ., data = employee_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6) %>%
  step_center(all_integer_predictors()) %>%
  step_scale(all_integer(), -all_outcomes()) %>%
  step_dummy(all_nominal_predictors(), one_hot = F) %>%
  step_nzv(all_predictors(), -all_outcomes())

prepare <- prep(employee_recipe, training = employee_train)
prepare$steps

baked_train <- bake(prepare, new_data = employee_train)
baked_test <- bake(prepare, new_data = employee_test)

# cross validated model
# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor())

tuned_mars <- train(
  x = subset(baked_train, select = -ratingOverall),
  y = baked_train$ratingOverall,
  method = "earth",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid)

# best model
tuned_mars$bestTune ## nprune degree ##2 12 1
# plot results
ggplot(tuned_mars)






