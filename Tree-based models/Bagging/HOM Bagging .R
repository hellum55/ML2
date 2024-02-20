##############################################################
# Bagging (Bootstrap aggregating)
# Ch.10, HOM with R
# Adv: improve the stability and accuracy of trees
# Adv.: works well with high-variance base learners
# Disadv.: tree correlation 
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
library(doParallel)  # for parallel backend to foreach (to speed computation)
library(foreach)     # for parallel processing with for loops
library(rsample)     # for sample split


# Modeling packages
library(caret)       # for general model fitting
library(rpart)       # for fitting decision trees
library(ipred)       # for fitting bagged decision trees
library(randomForest)# for fitting bagged decision trees

set.seed(123)

ames <- AmesHousing::make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# train bagged model (here with ipred library::bagging)
ames_bag1 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 100,  # trial with 100 bags
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0) # notice here
)

ames_bag1
# OOB RMSE = 28029.47 

ames_bag2 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 150,  # trial with 150 bags
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag2
# OOB RMSE = 27977.04

ames_bag3 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 300,  # trial with 300 bags
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag3
# OOB RMSE = 27664.28 

# Concl.: for this ames data,
# the error is stabilizing with just over 100 trees, 
# so we’ll likely not gain much improvement by simply bagging more trees.
# Also notice that by comparing the RMSE for the best individual pruned
# decision tree model (previous lecture) with the bagging models (this lecture), 
# the error has decreased significantly from around 40.000 to around 27.000. 


# train bagged model (here with caret library) 
# NOTE: run this at home; it takes 30 min to converge!
ames_bag4 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10), # 10-fold CV increases the convergence time
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)
ames_bag4

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


# Easily parallelize to speed computation (requires library(doParallel))
# Create a parallel socket cluster
   # A parallel socket cluster in R refers to a group of computing nodes 
   # connected by a network that work together to perform a computational 
   # task. The term "parallel" indicates that the task is split across 
   # multiple nodes to speed up processing, while "socket" refers to the 
   # networking interface used for communication. In R, the doParallel 
   # package provides functions for parallel processing on a single machine 
   # or in a cluster, allowing for faster completion of large and complex 
   # tasks.

cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend

# Fit trees in parallel and compute predictions on the test set
predictions <- foreach(
  icount(160), 
  .packages = "rpart", 
  .combine = cbind
) %dopar% {
  # bootstrap copy of training data
  index <- sample(nrow(ames_train), replace = TRUE)
  ames_train_boot <- ames_train[index, ]  
  
  # fit tree to bootstrap copy
  bagged_tree <- rpart(
    Sale_Price ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = ames_train_boot
  ) 
  
  predict(bagged_tree, newdata = ames_test)
}

predictions[1:5, 1:7]  # OOB RMSE
dim(predictions)

# Plot error curve for custom parallel bagging of 1-160 deep,
# unpruned decision trees.
predictions %>%
  as.data.frame() %>%
  mutate(
    observation = 1:n(),
    actual = ames_test$Sale_Price) %>%
  tidyr::gather(tree, predicted, -c(observation, actual)) %>%
  group_by(observation) %>%
  mutate(tree = stringr::str_extract(tree, '\\d+') %>% as.numeric()) %>%
  ungroup() %>%
  arrange(observation, tree) %>%
  group_by(observation) %>%
  mutate(avg_prediction = cummean(predicted)) %>%
  group_by(tree) %>%
  summarize(RMSE = RMSE(avg_prediction, actual)) %>%
  ggplot(aes(tree, RMSE)) +
  geom_line() +
  xlab('Number of trees')

# Shutdown parallel cluster
stopCluster(cl)

# Limitation of bagging trees 
# the trees in bagging are not completely independent of each 
# other since all the original features are considered at every 
# split of every tree; that is, if one or two feature dominates, 
# all tress will have that feature so the will be a similar structure 
# especially at the top of the tree. To avoid this limitation,
# random forest extend and improve upon bagged decision trees.


# Task: Consider the randomForest library from the textbook ISL. 
#       Implement a bagging model using the ames data and compare
#       the results. 




