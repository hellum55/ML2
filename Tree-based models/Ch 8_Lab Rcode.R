# *************************************************************************
# Chapter 8 Lab: Trees - with discussions
# *************************************************************************


# *************************************************************************
# 8.3.1. Fitting Classification Trees
# *************************************************************************
library(tree)
library(ISLR)
attach(Carseats)

# recode sales as a binary var and incorporate to the dataset
High=ifelse(Carseats$Sales<=8,"0","1")
Carseats=data.frame(Carseats,High)
str(Carseats)
Carseats$High = as.factor(Carseats$High)


# set a classsification tree
tree.carseats = tree(High ~ .-Sales,Carseats, split = c("deviance", "gini"))
summary(tree.carseats)


# plot tree, disply labels (text) and category names (pretty)
plot(tree.carseats)
text(tree.carseats,pretty=0)

# read the tree
tree.carseats

# estimate the test error 
RNGkind("L'Ecuyer-CMRG")
set.seed(2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
# accuraccy: (87+56)/200 = 0.71; test error = 0.29
# it you get slightly different results, it is because of the split

# pruning the tree
RNGkind("L'Ecuyer-CMRG")
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass) # FUN: classsification error rate guides the cross-validation and pruning process; default: deviance 
names(cv.carseats)
cv.carseats
# in the output, despite the name, $dev is the cross-validation error rate
# size = numer of terminal nodes of each tree considered
# $k = cost-complexity parameter (alpha in the slides)

par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")
# the trees with 19 terminal nodes results in the lowest cross-validation error rate.
# this is the most complex model;  
# as an example, if we wish to prune the tree 
# let us select best = 13 

# plot pruned tree
prune.carseats=prune.misclass(tree.carseats,best=13)
plot(prune.carseats)
text(prune.carseats,pretty=0)

# estimate test error 
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
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
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)

par(mfrow=c(1,1))
plot(tree.boston)
text(tree.boston,pretty=0)
# most expensive houses are asssociated with 
# larger homes (rm > 7.45), for which the tree predicts 
# a median house price of 45.38

# or smaller houses (rm<5.49) but for which the weighted 
# mean of distances to five Boston employment centres (dis)
# is short (dis<3.25); in this case
# the tree predicts a median house price of 43.28 


# cv.tree to see if pruning the tree is better
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
# in this case 9 terminal nodes returns the smallest error; 
# this is the most complex model meaning that pruning the tree 
# if not useful in this case; 
# if pruning is required, as an example, let us choose best = 5
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)


# test error
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2) 
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
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
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
