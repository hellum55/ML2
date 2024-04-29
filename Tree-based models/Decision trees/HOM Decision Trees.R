##############################################################
# Decision Trees
# Ch.9, HOM with R
# Adv: simple and transparent 
# Disadv: unstable
##############################################################
# Data: Ames 
# DV: Sale_Price (i.e., $195,000, $215,000)
# Features: 80
# Observations: 2,930
# Objective: use property attributes to predict the sale price of a house
# Access: provided by the AmesHousing package (Kuhn 2017a)
# more details: See ?AmesHousing::ames_raw

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


ames <- AmesHousing::make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)


# model
ames_dt1 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova" 
)
ames_dt1

#plot tree
rpart.plot(ames_dt1)

# var imp
par(mar=c(8,4,3,2))
d <- as.data.frame(ames_dt1$variable.importance)

#plot cp
plotcp(ames_dt1) 

#rpart cross validation results
ames_dt1$cptable

# for demo see if cp=0
ames_dt2 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(cp = 0, xval = 10)
)
plotcp(ames_dt2)
abline(v = 11, lty = "dashed")



# ***

# Alternative with caret 
# a) "method = rpart" implies cp-based pruning
ames_dt3 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "rpart",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 20 
)
ames_dt3
ggplot(ames_dt3)
ames_dt3$bestTune


#plot tree
ames_dt3$finalModel
fancyRpartPlot(ames_dt3$finalModel, sub = NULL)


# vip
vip(ames_dt3, num_features = 40, bar = FALSE)

# Construct partial dependence plots
p1 <- partial(ames_dt3, pred.var = "Gr_Liv_Area") %>% autoplot()
p2 <- partial(ames_dt3, pred.var = "Year_Built") %>% autoplot()
p3 <- partial(ames_dt3, pred.var = c("Gr_Liv_Area", "Year_Built")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))
# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
# notice decision trees have rigid non-smooth prediction surfaces compared to MARS



# b) "method = rpart2", max tree depth tuning 
ames_dt4 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "rpart2",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 20 
)
ames_dt4
ggplot(ames_dt4)
ames_dt4$bestTune

ames_dt4$finalModel
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
