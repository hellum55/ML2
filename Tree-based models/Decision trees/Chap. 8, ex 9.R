library(tree)
library(ISLR)
attach(OJ)
View(OJ)

#Question a####
set.seed(2)
train = sample(dim(OJ)[1], 800)
OJ.train = OJ[train, ]
OJ.test = OJ[-train, ]

#Question b####
tree.OJ = tree(Purchase~., data=OJ.train)
summary(tree.OJ)
#The tree uses two variables which is LoyalCH and PriceDiff and it has 9 terminal nodes.
#The misclassifications error / train-error is 0.15.

#Question c####
tree.OJ
#Here we can choose the terminal node that is labelled number 9: The splitting variable is LoyalCH.
#The splitting value of the node 0.03 and there are 56 points in the subtree.
#The deviance for all points in the wihtin the subtree 106.6. The star after the parantheses tells us that
#it is in fact a terminal node. 0.17 % of the points in the subtree have CH as value of Sales and the remaining
#81% points have MM as value of Sales

#Question d####
plot(tree.OJ)
text(tree.OJ, pretty=0)
View(OJ)
#From the plot we can observe that LoyalCH is the most important variable to explain Purchase. It is so important that the
#first 3 nodes contains LoyalCH. If LoyalCH < 0.27 the tree predicts MM and if LoyalCH > 0.76 it predicts Ch.
#For values inbetween these values PriceDiff also plays a role.

#Question e####
OJ.pred = predict(tree.OJ, OJ.test, type = "class")
table(OJ.test$Purchase, OJ.pred)
error.rate <- (148+70)/(148+15+37+70)
#0.8 error rate

#Question f####
cv.oj = cv.tree(tree.OJ, FUN = prune.tree)
cv.oj$size

#Question g####
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")
#Size 4 gives the lowest CV-error

#Question i####
oj.pruned = prune.tree(tree.OJ, best = 4)

#Question j####
summary(oj.pruned)
#The MIS-error rate has increased again.

#Question k####
#Unpruned
pred.unpruned = predict(tree.OJ, OJ.test, type = "class")
misclass.unpruned = sum(OJ.test$Purchase != pred.unpruned)
misclass.unpruned/length(pred.unpruned)

#Pruned
pred.pruned = predict(oj.pruned, OJ.test, type = "class")
misclass.pruned = sum(OJ.test$Purchase != pred.pruned)
misclass.pruned/length(pred.pruned)
#The pruned error rate is larger than the unpruned