library(ISLR)
#Question a####
train = 1:1000
Caravan$Purchase = ifelse(Caravan$Purchase == "Yes", 1, 0)
Caravan.train = Caravan[train, ]
Caravan.test = Caravan[-train, ]

#Question b####
library(gbm)
set.seed(342)
boost.caravan = gbm(Purchase ~ ., data = Caravan.train, n.trees = 1000,
                    shrinkage = 0.01, 
                    distribution = "bernoulli")
summary(boost.caravan)
#The most important variables in this model is Ppersaut, MKOOPKLA, and MOPLHOOG. 

#Question c####
boost.prob = predict(boost.caravan, Caravan.test, n.trees = 1000, type = "response")
boost.pred = ifelse(boost.prob > 0.2, 1, 0)
table(Caravan.test$Purchase, boost.pred)