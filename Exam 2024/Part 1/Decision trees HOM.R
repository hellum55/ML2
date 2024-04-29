##############################################################
# Decision Trees
# Ch.9, HOM with R
# Adv: simple and transparent 
# Disadv: unstable
##############################################################

# Helper packages
library(dplyr)       # for data wrangling
library(ggplot2)     # for awesome plotting
library(rsample)     # for sample split

# Modeling packages
library(rpart)       # direct engine for decision tree application
library(caret)       # meta engine for decision tree application

# Model interpretability packages
library(rpart.plot)  # for plotting decision trees
library(rattle)      # for plotting decision trees
library(vip)         # for feature importance
library(pdp)         # for feature effects
#Load in the data
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
library(recipes)
set.seed(123)
split <- initial_split(data_employee, prop = 0.7, strata = "Satisfied") 

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

# ***

# Alternative with caret 
# a) "method = rpart" implies cp-based pruning
rating_dt3 <- train(
  Satisfied ~ .,
  data = baked_train,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC",
  tuneLength = 20)

rating_dt3
ggplot(rating_dt3)
rating_dt3$bestTune

#plot tree
rating_dt3$finalModel
fancyRpartPlot(rating_dt3$finalModel, sub = NULL)
# vip
vip(rating_dt3, num_features = 40, bar = FALSE)

#Test it against the test set
pred.satisfied = predict(rating_dt3, baked_test)
confusionMatrix(pred.satisfied, baked_test$Satisfied)


# Construct partial dependence plots
p1 <- partial(rating_dt3, pred.var = "ratingSeniorLeadership") %>% autoplot()
p2 <- partial(ames_dt3, pred.var = "Year_Built") %>% autoplot()
p3 <- partial(ames_dt3, pred.var = c("Gr_Liv_Area", "Year_Built")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))
# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
# notice decision trees have rigid non-smooth prediction surfaces compared to MARS


# b) "method = rpart2", max tree depth tuning 
rating_dt4 <- train(
  Satisfied ~ .,
  data = baked_train,
  method = "rpart2",
  trControl = trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC",
  tuneLength = 20)

rating_dt4
ggplot(satisfied_dt4)
rating_dt4$bestTune

rating_dt4$finalModel
fancyRpartPlot(ames_dt4$finalModel, sub = NULL)


# Evaluate test RMSE for dt1, dt3 and dt4
#rpart library with rpart function, cp-based pruning
dt1_pred = predict(ames_dt1, ames_test)
dt1_RMSE = sqrt(mean((ames_test$Sale_Price - dt1_pred)^2))
dt1_RMSE

# caret with "method = rpart", cp-based pruning
dt3_pred = predict(ames_dt3, ames_test)
dt3_RMSE = sqrt(mean((ames_test$Sale_Price - dt3_pred)^2))
dt3_RMSE

# caret with "method = rpart2",max tree depth tuning 
dt4_pred = predict(ames_dt4, ames_test)
dt4_RMSE = sqrt(mean((ames_test$Sale_Price - dt4_pred)^2))
dt4_RMSE

# cp-based pruning is best
