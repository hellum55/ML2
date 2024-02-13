library(tree)
library(ISLR)
attach(Carseats)
View(Carseats)
#Question a####
set.seed(2)
train = sample(dim(Carseats)[1], dim(Carseats)[1] * 0.33)
Carseats.train = Carseats[train, ]
Carseats.test = Carseats[-train, ]

#Question b####
tree.carseats = tree(Sales~., data=Carseats.train)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty=0)
pred.carseats = predict(tree.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.carseats)^2)
#We can see from the summary function that "ShelveLoc", "Price", "CompPrice", "Income", "Age", "Population", "Advertising"
#are the variables that are considered with 17 nodes and a deviance of 2.1.
#The test MSE is 5.48

#Question c####
cv.carseats = cv.tree(tree.carseats, FUN=prune.tree)
par(mfrow=c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type="b")
plot(cv.carseats$k, cv.carseats$dev, type="b")
# Best size = 6
pruned.carseats = prune.tree(tree.carseats, best=6)
par(mfrow=c(1, 1))
plot(pruned.carseats)
text(pruned.carseats, pretty=0)
pred.pruned = predict(pruned.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.pruned)^2)
#In this case the pruning with CV results in a slight decrease in test MSE.
#(5.43) The decision tree is a little bit shallower, so we didnt get so much out of it.

#Question d####
library(randomForest)
bag.carseats = randomForest(Sales~., data=Carseats.train, mtry=10, ntree=500, importance=T)
bag.pred = predict(bag.carseats, Carseats.test)
mean((Carseats.test$Sales - bag.pred)^2)
importance(bag.carseats)
#A significantly decrease regarding the test MSE.

#Question e####
set.seed(123)
rf.carseats = randomForest(Sales ~ ., data=Carseats.train, mtry = 5, ntree = 500, 
                           importance = T)
rf.pred = predict(rf.carseats, Carseats.test)
mean((Carseats.test$Sales - rf.pred)^2)
#A huge decrease compared to simple regression tree, but not better than bagging approach.
#The main difference between bagging and random forest is that we do not use all the variables within the data
#but a subset of it. rule of thumb is to use a third a the varibles if p>3.
