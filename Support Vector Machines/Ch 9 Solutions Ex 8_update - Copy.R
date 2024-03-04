# *************************************************************************
# # Chapter 9 SVM: Solution to Applied Ex 8
# *************************************************************************
library(ISLR)
library(e1071)

View(OJ)
dim(OJ)

# Familiarize again with the dataset: 
# The data contains 1070 purchases where the customer either purchased Citrus Hill (CH) or Minute Maid Orange Juice (MM)
# A number of characteristics of the customer and product are recorded
# Open ISLR library and read more on this dataset and its variables
# Once you are ready, start the analysis.


# a). Create a training set containing a random sample of 800 observations

set.seed(1)
train <- sample(nrow(OJ), 800)
OJ.train <- OJ[train, ]
OJ.test <- OJ[-train, ]


# b). Fit a SV classifier using cost=0.01

svm.linear <- svm(Purchase ~ ., data = OJ.train, kernel = "linear", cost = 0.01)
summary(svm.linear)

# SVC creates 432 support vectors out of 800 training points. 
# Out of these, 217 belong to level MM and remaining 215 belong to level CH



# c). Training and test error rates

train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)

# training error rate is: 
(78+55)/800 
# [1] 0.16625


test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)

# testing error rate is: 
(31+18)/270
# [1] 0.1814815



# d). tune() the cost parameter

set.seed(2)
tune.out <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "linear", ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out)


# e). Training and test error rates for the best (tuned) cost

svm.linear <- svm(Purchase ~ ., kernel = "linear", data = OJ.train, cost = tune.out$best.parameter$cost)
train.pred <- predict(svm.linear, OJ.train)
table(OJ.train$Purchase, train.pred)

(71+56)/800
# 0.15875

test.pred <- predict(svm.linear, OJ.test)
table(OJ.test$Purchase, test.pred)

(32+19)/270
# 0.1888889



# f). Repeating the steps b)-e) using SVM with radial kernel, for default gamma

svm.rad <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost=0.01)
summary(svm.rad) 

train.pred.rad <- predict(svm.rad, OJ.train)
table(OJ.train$Purchase, train.pred.rad)

306/800
# [1] 0.3825


test.pred.rad <- predict(svm.rad, OJ.test)
table(OJ.test$Purchase, test.pred.rad)

111/270
# [1] 0.4111111


set.seed(1)
tune.out.rad <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "radial", ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out.rad)

svm.rad2 <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out.rad$best.parameter$cost)

train.pred.rad2 <- predict(svm.rad2, OJ.train)
table(OJ.train$Purchase, train.pred.rad2)

(77+39)/800
# 0.145

test.pred.rad2 <- predict(svm.rad2, OJ.test)
table(OJ.test$Purchase, test.pred.rad2)

(29+16)/270
# 0.1666667



# g). Repeating the steps b)-e) for polinomial kernel with degree=2
  
svm.poly <- svm(Purchase ~ ., kernel = "polynomial", data = OJ.train, degree = 2)
summary(svm.poly) 
  
train.pred.poly <- predict(svm.poly, OJ.train)
table(OJ.train$Purchase, train.pred.poly)

(105+33)/800
# [1] 0.1725


test.pred.poly <- predict(svm.poly, OJ.test)
table(OJ.test$Purchase, test.pred.poly)

(41+10)/270
# [1] 0.1888889


set.seed(2)
tune.out.poly <- tune(svm, Purchase ~ ., data = OJ.train, kernel = "polynomial", degree = 2, ranges = list(cost = 10^seq(-2, 1, by = 0.25)))
summary(tune.out.poly)

svm.poly2 <- svm(Purchase ~ ., kernel = "polynomial", degree = 2, data = OJ.train, cost = tune.out.poly$best.parameter$cost)
summary(svm.poly2)

train.pred.poly2 <- predict(svm.poly2, OJ.train)
table(OJ.train$Purchase, train.pred.poly2)

(72+44)/800
# [1] 0.145

test.pred.poly2 <- predict(svm.poly2, OJ.test)
table(OJ.test$Purchase, test.pred.poly2)

(31+19)/270
# [1] 0.1851852


# h).Using the error rate of testing set as a criterion of evaluation, 
# the best model is the SVM with radial kernel


###############################################################################################################################################
# ROC curve and evaluate AUC for each model using "ROCR" library
###############################################################################################################################################
#   Below, for comparison reasons, I evaluate ROC for both training and testing data sets; 
#   in practice, ROC for testing set is the important one. 

library(ROCR)

OJ.train$Purchase = as.ordered(OJ.train$Purchase)
OJ.test$Purchase = as.ordered(OJ.test$Purchase)

rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

# ROC for training data and with the two models (radial and poly SVM)
svm.rad2 <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out.rad$best.parameter$cost,decision.values=T)
fitted.rad2=attributes(predict(svm.rad2,OJ.train,decision.values=TRUE))$decision.values
rocplot(fitted.rad2,OJ.train$Purchase,main="Training Data")
abline(a=0,b=1)

svm.poly2 <- svm(Purchase ~ ., kernel = "polynomial", degree = 2, data = OJ.train, cost = tune.out.poly$best.parameter$cost)
fitted.poly2=attributes(predict(svm.poly2,OJ.train,decision.values=T))$decision.values
rocplot(fitted.poly2,OJ.train$Purchase,add=T,col="red")


# ROC for testing data and with the two models (radial and poly SVM)
fitted.rad2.test=attributes(predict(svm.rad2,OJ.test,decision.values=T))$decision.values
rocplot(fitted.rad2.test,OJ.test$Purchase,main="Test Data")

fitted.poly2.test=attributes(predict(svm.poly2,OJ.test,decision.values=T))$decision.values
rocplot(fitted.poly2.test,OJ.test$Purchase,add=T,col="red")

          
# Calculating AUC
   # * run the models setting the option "probability = TRUE" and predict with the option "probability = TRUE" 
   # * below I consider CH the "1" class ([,1] ) when making predictions; if MM is of interest then use ([,2]); 
   #  if omitted, the software will display AUC for both classes 

# * AUC for radial SVM
svm.rad2 <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out.rad$best.parameter$cost,probability=TRUE)
yhat.opt.rad2 = predict(svm.rad2,OJ.test, probability =TRUE)
pred.rad2 <- prediction(attributes(yhat.opt.rad2)$probabilities[,1], OJ.test$Purchase) 
perf.rad2 <- performance(pred.rad2,"tpr","fpr")
par(mfrow=c(1,2))
plot(perf.rad2,colorize=TRUE, main="Radial SVM Test Data")

auc_ROCR_rad2 <- performance(pred.rad2, measure = "auc")
auc_ROCR_rad2 <- auc_ROCR_rad2@y.values[[1]]
auc_ROCR_rad2 

# * AUC for poly SVM
svm.poly2 <- svm(Purchase ~ ., kernel = "polynomial", degree = 2, data = OJ.train, cost = tune.out.poly$best.parameter$cost, probability=TRUE)
yhat.opt.poly2=predict(svm.poly2,OJ.test,probability =TRUE)

pred.poly2 <- prediction(attributes(yhat.opt.poly2)$probabilities[,1], OJ.test$Purchase) 
perf.poly2 <- performance(pred.poly2,"tpr","fpr")
plot(perf.poly2,colorize=TRUE, main="Poly SVM Test Data")

auc_ROCR_poly2 <- performance(pred.poly2, measure = "auc")
auc_ROCR_poly2 <- auc_ROCR_poly2@y.values[[1]]
auc_ROCR_poly2 

# see at the end of the code other advantages of using ROCR



###############################################################################################################################
# * ROC and AUC using "caTools" library 
# * This method was shown in Bayesian Networks course
###############################################################################################################################

library(caTools) 
#  * ROC and AUC for linear SV
 svm <- svm(Purchase ~ ., kernel = "linear", data = OJ.train, cost = tune.out$best.parameter$cost, probability=TRUE)
 yhat.opt.lin=predict(svm,OJ.test,probability =TRUE)
 colAUC(attributes(yhat.opt.lin)$probabilities[,1], OJ.test$Purchase, plotROC = TRUE) 
 
#  * ROC and AUC for poly SVM
  svm.poly2 <- svm(Purchase ~ ., kernel = "polynomial", degree = 2, data = OJ.train, cost = tune.out.poly$best.parameter$cost, probability=TRUE)
  yhat.opt.poly=predict(svm.poly2,OJ.test,probability =TRUE)
  colAUC(attributes(yhat.opt.poly)$probabilities[,1], OJ.test$Purchase, plotROC = TRUE) 

# * ROC and AUC for radial SVM
  svm.rad2  <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out.rad$best.parameter$cost, probability=TRUE)
  yhat.opt.rad=predict(svm.rad2,OJ.test,probability=TRUE)
  colAUC(attributes(yhat.opt.rad)$probabilities[,1], OJ.test$Purchase, plotROC = TRUE)

# * We conclude that the three models perform well and similar (AUC ~ 0.88).
# END  

  
  
  
# ****
# Extra code on the advantages and functions offered in "ROCR" library
# ***** 
  # Selecting and evaluating models performance based on AUC is common practice. 
  # But, at least after the best model is selected and evaluated, one also has 
  # to decide which is the optimal cut-off to make the classification
  # Imagine this is the model
  svm.rad2 <- svm(Purchase ~ ., kernel = "radial", data = OJ.train, cost = tune.out.rad$best.parameter$cost, probability=TRUE) # requires probability = TRUE
  predictions = predict(svm.rad2,OJ.test, probability=TRUE) 
  predictions_prob = attributes(predictions)$probabilities[,1] # extract the column probabilities from the list
  
  
  # Two objects required in ROCR:
  # prediction object
  pred = prediction(predictions_prob, OJ.test$Purchase, label.ordering = NULL)
  #class(pred)
  slotNames(pred)
  pred@cutoffs
  pred@labels
  pred@fp
  
  # performance object, allowing to vary the measure  
  roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
  plot(roc.perf)
  abline(a=0, b= 1)
  
  # now,
      # i) Getting the "optimal" cutoff such that it weighs both sensitivity and specificity equally 
      opt.cut = function(perf, pred){
        cut.ind = mapply(FUN=function(x, y, p){
          d = (x - 0)^2 + (y-1)^2
          ind = which(d == min(d))
          c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
            cutoff = p[[ind]])
        }, perf@x.values, perf@y.values, pred@cutoffs)
      }
      print(opt.cut(roc.perf, pred))
      
      
      # ii) Getting the "optimal" cutoff to minimize the total cost 
      # imagine that we have different costs for FP and FN.  
      # assume cost of false positive is half of the cost of false negative
      cost.perf = performance(pred, "cost", cost.fp = 2, cost.fn = 1) 
      pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
      # it may result in a differenet optimal cut-point, corresponding to the minimum total cost
      
      
      # iii) Getting the "optimal" cutoff to maximize the accuracy 
      acc.perf = performance(pred, measure = "acc")
      plot(acc.perf)
      
      i = which.max( slot(acc.perf, "y.values")[[1]] )
      acc = slot(acc.perf, "y.values")[[1]][i]
      cutoff = slot(acc.perf, "x.values")[[1]][i]
      print(c(accuracy= acc, cutoff = cutoff))
      
      
 
