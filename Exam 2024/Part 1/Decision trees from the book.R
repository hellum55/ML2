# *************************************************************************
#  Fitting Classification Trees
# *************************************************************************
library(tree)

data_employee <- read.csv("data_employee.csv", stringsAsFactors=TRUE)

# recode ratingOverall as a binary var and incorporate to the dataset
Satisfied=ifelse(data_employee$ratingOverall<=3,"0","1")
data_employee=data.frame(data_employee,Satisfied)
str(data_employee)
data_employee$Satisfied = as.factor(data_employee$Satisfied)

#Look at the distribution of the variable 'Satisfied'
plot_bar(data_employee$Satisfied)
#It is unbalanced. More employees are satisfied than dissatisfied, which of course is great for the company. 

# set a classsification tree
tree.rating = tree(Great ~ .-ratingOverall, data_employee, split = c("deviance", "gini"))
summary(tree.rating)

nrow(data_employee$isCurrentJob)

# plot tree, disply labels (text) and category names (pretty)
plot(tree.rating)
text(tree.rating,pretty=0)

# read the tree
tree.rating

# estimate the test error 
RNGkind("L'Ecuyer-CMRG")
set.seed(2)
train=sample(1:nrow(data_employee), (0.7*nrow(data_employee)))
rating.test=data_employee[-train,]
Great.test=Great[-train]
tree.rating=tree(Great~.-ratingOverall, data_employee,subset=train)
tree.pred=predict(tree.rating, rating.test, type="class")
table(tree.pred, Great.test)
# accuraccy: (87+56)/200 = 0.71; test error = 0.29
# it you get slightly different results, it is because of the split

# pruning the tree
RNGkind("L'Ecuyer-CMRG")
set.seed(3)
cv.rating=cv.tree(tree.rating,FUN=prune.misclass) # FUN: classsification error rate guides the cross-validation and pruning process; default: deviance 
names(cv.rating)
cv.rating
# in the output, despite the name, $dev is the cross-validation error rate
# size = numer of terminal nodes of each tree considered
# $k = cost-complexity parameter (alpha in the slides)

par(mfrow=c(1,2))
plot(cv.rating$size, cv.rating$dev,type="b")
plot(cv.rating$k, cv.rating$dev,type="b")
# the trees with 19 terminal nodes results in the lowest cross-validation error rate.
# this is the most complex model;  
# as an example, if we wish to prune the tree 
# let us select best = 13 

# plot pruned tree
prune.rating=prune.misclass(tree.rating,best=2)
plot(prune.rating)
text(prune.rating,pretty=0)

# estimate test error 
tree.pred=predict(prune.rating, rating.test,type="class")
table(tree.pred, Great.test)
# accuracy: (78+62)/200 =0.7; test error rate = 0.3
# here, pruning did not improve the accuracy.

# ps. our solution is not the same as in the textbook because of the random split.
# the instability of tree solutions when working with small datasets is acknowledged.


# *************************************************************************
# 8.3.2. Fitting Regression Trees
# *************************************************************************
library(MASS)
RNGkind("L'Ecuyer-CMRG")
set.seed(1)
train=sample(1:nrow(data_employee), (0.7*nrow(data_employee)))
tree.employee=tree(ratingOverall~., data_employee, subset=train)
summary(tree.employee)

par(mfrow=c(1,1))
plot(tree.employee)
text(tree.employee,pretty=0)
# most expensive houses are asssociated with 
# larger homes (rm > 7.45), for which the tree predicts 
# a median house price of 45.38

# or smaller houses (rm<5.49) but for which the weighted 
# mean of distances to five Boston employment centres (dis)
# is short (dis<3.25); in this case
# the tree predicts a median house price of 43.28 


# cv.tree to see if pruning the tree is better
cv.employee=cv.tree(tree.employee)
plot(cv.employee$size, cv.employee$dev,type='b')
# in this case 9 terminal nodes returns the smallest error; 
# this is the most complex model meaning that pruning the tree 
# if not useful in this case; 
# if pruning is required, as an example, let us choose best = 5
prune.employee=prune.tree(tree.employee,best=5)
plot(prune.employee)
text(prune.employee,pretty=0)


# test error
yhat=predict(tree.employee, newdata=data_employee[-train,])
employee.test=data_employee[-train,"ratingOverall"]
plot(yhat,employee.test)
abline(0,1)
mean((yhat-employee.test)^2) 
# [1] 24.59711 ~ 25.000  (=MSE)
# meaning that the model leads to test predictions that are within around 
# 5000 (sqrt(MSE)) of the true median home value for the suburb


# *************************************************************************
# 8.3.3. Bagging and Random Forests (same library and function)
# *************************************************************************
# Bagging
library(randomForest)
RNGkind("L'Ecuyer-CMRG")
set.seed(1)
bag.employee=randomForest(ratingOverall~.,data=data_employee, subset=train, mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)
# test MSE for bagging is around 16.37 (significantly lower than that for regression tree)

# increasing the number of trees grown ntree=1000
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=1000)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
# [1] 16.06824


# Random forest (mtry argument)
RNGkind("L'Ecuyer-CMRG")
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
# [1] 15.53946
# RF yielded an improvement over bagging
importance(rf.boston)
varImpPlot(rf.boston)
# Concl. across all the trees considered by the random forest, 
# the wealth level of community and the house size (rm)
# are far the two most important variables. 




# *************************************************************************
# 8.3.4 Boosting 
# *************************************************************************
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.boston)

par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")


yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
# [1] 15.83297 (MSE) similar to Rf

boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
# [1] 19.15354 (MSE) 
# changing the shrinkage parameter does not imporve the fit in this case. 
# creating a grid of values for this parameter and testing those values may help to 
# identify the best fit. 

# Next, see file XGBoost.R
