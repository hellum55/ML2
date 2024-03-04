# *************************************************************************
# # Chapter 9 SVM Solutions to Applied Ex. 4-7
#  (with explanations AAT)
# *************************************************************************
library(ISLR) # for dataset
library(e1071) # for implementing SVMs

# Ex.4 

# We begin by creating a data set with non-linear separation between the two classes.
set.seed(1)
x <- rnorm(100)
y <- 4 * x^2 + 1 + rnorm(100)
plot(x,y)

set.seed(1)
class <- sample(100, 50)
y[class] <- y[class] + 3
y[-class] <- y[-class] - 3

plot(x[class], y[class], col = "red", xlab = "X", ylab = "Y", ylim = c(-6, 30))
points(x[-class], y[-class], col = "blue")

# Now, we fit a support vector classifier on the training data.
# create the dependent variable as -1 and 1 
z <- rep(-1, 100)
z[class] <- 1
# build my data
data <- data.frame(y = y,x = x, z = as.factor(z))


# Split the dataset
set.seed(123)
train <- sample(100, 50)
data.train <- data[train, ]
data.test <- data[-train, ]

# Apply the model
svm.linear <- svm(z ~ ., data = data.train, kernel = "linear", cost = 10)
plot(svm.linear, data.train, ylim = c(-6, 25), xlim = c(-2,1.90) )
# In the plot, points that are represented by an “x” are the support vectors
# or the points that directly affect the classification line
# The points marked with an “o” are the other points, which don’t affect the calculation of the line.

# Check prediction
table(predict = predict(svm.linear, data.train), truth = data.train$z)
# Concl: The support vector classifier makes 6 errors on the training data.
# (one may get another output if the seed is not preserved)


# Now let us fit a support vector machine with a polynomial kernel.
svm.poly <- svm(z ~ ., data = data.train, kernel = "polynomial", cost = 10)
plot(svm.poly, data.train)
table(predict = predict(svm.poly, data.train), truth = data.train$z)
#Concl: 10 errors on the training data


# Now let us fit a support vector machine with a radial kernel.
svm.rad <- svm(z ~ ., data = data.train, kernel = "radial", cost = 10)
plot(svm.rad, data.train)
table(predict = predict(svm.rad, data.train), truth = data.train$z)
#Concl: 0 training errors


# Let us evaluate the performance of the 3 models in the testing data
table(predict = predict(svm.linear, data.test), truth = data.test$z)
table(predict = predict(svm.poly, data.test), truth = data.test$z)
table(predict = predict(svm.rad, data.test), truth = data.test$z)
# Concl: linear, polynomial and radial support vector machines 
# classify respectively 7, 12 and 5 observations incorrectly. 
# in my run, radial kernel performs best on the test data.


# visualize 
print(svm.rad)
plot(svm.rad, data.test, ylim = c(-6, 25), xlim = c(-2.1,2.5)) 
# notice the incorrectly classified points that were "1"
# but classified as "-1" 


      # An alternative with package "kernlab" 
      # library(kernlab)
      # kernfit <- ksvm(z ~ ., data = data.train, kernel = 'vanilladot', C = 10) # vanilladot = linear kernel
      # table(predict = predict(kernfit, data.test), truth = data.test$z)
      # print(kernfit)
      # plot(kernfit,data=data.test, ylim = c(-6, 25), xlim = c(-2.1,2.5))
        # the color gradient that indicates how confidently a new point 
        # would be classified based on its features;
        # in this package, the support vectors are marked as filled-in points, 
        # while the classes are denoted by different shapes.
  


# Ex.5 
# LR using non-linear transformations of the features 

set.seed(1)
x1 <- runif(500) - 0.5
x2 <- runif(500) - 0.5
y <- 1 * (x1^2 - x2^2 > 0)

#b)
plot(x1, x2, col = (2-y), xlab = "X1", ylab = "X2", pch=(3-y))

#c)
logit.fit <- glm(y ~ x1 + x2, family = "binomial")
summary(logit.fit)


#d)
data <- data.frame(x1 = x1, x2 = x2, y = y) 
probs <- predict(logit.fit, data, type = "response") # consider "data" as training data
preds <- rep(0, 500) #initialize the vector with 500 observations
preds[probs > 0.50] <- 1 # cutoff at 0.50 and I assigned a label of 1 to all prediction that surpass that cutoff
plot(data[preds == 1, ]$x1, data[preds == 1, ]$x2, col = (4 - 1), pch = (3 - 1), xlab = "X1", ylab = "X2",ylim = c(-0.5, 0.5), xlim = c(-0.5,0.5) )
points(data[preds == 0, ]$x1, data[preds == 0, ]$x2, col = (4 - 0), pch = (3 - 0))
# notice the decision boundary is linear; 
# this is normal, as we have discussed that a logistic model is in essence linear. 


#e) apply a polynomial LR model
logitnl.fit <- glm(y ~ poly(x1, 2) + poly(x2, 2) + I(x1 * x2), family = "binomial")
summary(logitnl.fit)


#f)
probs <- predict(logitnl.fit, data, type = "response")
preds <- rep(0, 500)
preds[probs > 0.50] <- 1
plot(data[preds == 1, ]$x1, data[preds == 1, ]$x2, col = "red", pch = 0, xlab = "X1", ylab = "X2")
points(data[preds == 0, ]$x1, data[preds == 0, ]$x2, col = "blue", pch = 2)
# The non-linear decision boundary is similar to the true decision boundary 
# (here we know the true decision boundary because we simulated the data).


#g) apply a svm model LINEAR
data$y <- as.factor(data$y) # do not forget to set y as a factor 
svm.fit <- svm(y ~ x1 + x2, data, kernel = "linear", cost = 0.01)
plot(svm.fit, data)
preds <- predict(svm.fit, data)
table(predict = predict(svm.fit, data), truth = data$y)
# This support vector classifier (even with low cost) classifies all points to a single class. 
# svm classified all points as zeros

plot(data[preds == 0, ]$x1, data[preds == 0, ]$x2, col = "red", pch = 0, xlab = "X1", ylab = "X2")
points(data[preds == 1, ]$x1, data[preds == 1, ]$x2, col = "blue", pch = 2) 
# points is a generic function to draw a sequence of points at the specified coordinates. 
# The specified character(s) are plotted, centered at the coordinates.
# as seen, there was no "1" predicted so the nothing is visualized (svm classfied all points as zeros) 


#g) apply a svm model RADIAL
data$y <- as.factor(data$y) # do not forget to set y as a factor
svmnl.fit <- svm(y ~ x1 + x2, data, kernel = "radial", gamma = 1) # gamma can be tuned; cost is here the default
plot(svmnl.fit, data)
table(predict = predict(svmnl.fit, data), truth = data$y)
preds <- predict(svmnl.fit, data)

#plot(data[preds == 0, ]$x1, data[preds == 0, ]$x2, col = "red", pch = 0, xlab = "X1", ylab = "X2")
#points(data[preds == 1, ]$x1, data[preds == 1, ]$x2, col ="blue" , pch = 2) 
# the non-linear decision boundary is similar to the true decision boundary.


#i) Comment on your results.

# We may conclude that:
# 1. SVM with non-linear kernel and logistic regression (LR) with interaction terms 
#    are equally very powerful for finding non-linear decision boundaries. 
# 2. SVM with linear kernel and LR without any interaction terms
#    are very bad when it comes to finding non-linear decision boundaries. 
# 3. An argument in favor of using SVM instead of LR, is that LR requires 
#    some manual tuning to find the right interaction terms, 
#    while when using SVM we only need to tune the gamma.




# Ex. 6
# a.) # We randomly generate 1000 points and scatter them across line x=y with wide margin. 
# We also create noisy points along the line. 
# These points make the classes barely separable and also shift the maximum margin classifier.

set.seed(1)
x.one <- runif(500, 0, 90)
y.one <- runif(500, x.one + 10, 100)
x.one.noise <- runif(50, 20, 80)
y.one.noise <- 5/4 * (x.one.noise - 10) + 0.1

x.zero <- runif(500, 10, 100)
y.zero <- runif(500, 0, x.zero - 10)
x.zero.noise <- runif(50, 20, 80)
y.zero.noise <- 5/4 * (x.zero.noise - 10) - 0.1

class.one <- seq(1, 550)
x <- c(x.one, x.one.noise, x.zero, x.zero.noise)
y <- c(y.one, y.one.noise, y.zero, y.zero.noise)

plot(x[class.one], y[class.one], col = "green", pch = "+", ylim = c(0, 100))
points(x[-class.one], y[-class.one], col = "orange", pch = 4)




#b) 
set.seed(2)
z <- rep(0, 1100)
z[class.one] <- 1
data <- data.frame(x = x, y = y, z = as.factor(z))
dim(data)

# tune () function is using 10-fold cross validation (default)
tune.out <- tune(svm, z ~ ., data = data, kernel = "linear", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100, 1000, 10000)))
summary(tune.out)
# notice in terms of cost, the lowest cv-error is corresponding to a cost of 10000
# we may calculate and display the numbers as 
data.frame(cost = tune.out$performance$cost, misclass = tune.out$performance$error * 1100)



# c) Testing with another dataset
x.test <- runif(1000, 0, 100)
class.one <- sample(1000, 500)
y.test <- rep(NA, 1000)
# Set y > x for class.one
for (i in class.one) {
  y.test[i] <- runif(1, x.test[i], 100)
}
# set y < x for class.zero
for (i in setdiff(1:1000, class.one)) {
  y.test[i] <- runif(1, 0, x.test[i])
}
plot(x.test[class.one], y.test[class.one], col = "blue", pch = "+")
points(x.test[-class.one], y.test[-class.one], col = "red", pch = 4)

set.seed(3)
z.test <- rep(0, 1000)
z.test[class.one] <- 1
data.test <- data.frame(x = x.test, y = y.test, z = as.factor(z.test))

costs <- c(0.01, 0.1, 1, 5, 10, 100, 1000, 10000)

train.err <- rep(NA, length(costs))
test.err <- rep(NA, length(costs))
for (i in 1:length(costs)) {
  svm.fit <- svm(z ~ ., data = data, kernel = "linear", cost = costs[i])
  predtrain <- predict(svm.fit, data) # train error
  pred <- predict(svm.fit, data.test) # test error
  train.err[i] <- sum(predtrain != data$z)
  test.err[i] <- sum(pred != data.test$z)
}
options(scipen=999)
data.frame(cost = costs, misclass_train = train.err)
data.frame(cost = costs, misclass_test = test.err)



#d) We observe an overfitting phenomenon for linear kernel. 
# A large cost (meaning a narrower band in the svm() function - see below) tries to fit 
# correctly noisy-points and hence overfits the train data. 
# A small cost, however, makes a few errors on the noisy training points 
# and performs better on test data.


#  NOTE: explanation link to the above conclusion, based on the question I received in class:  
#  it is reported in the LAB section that svm() function interprets the C parameter 
#  in the opposite way than used in the theory
#  p.359: svm() uses a slightly different formulation from (9.14) and (9.25)
#   C in svm() function means the cost of a violation to the margin
#   - when C (the cost) is small, that margin will be wide
#   - when C is large, than the margin will be narrow

# in the theory, C is presented as budget (p.347). in this perspective the interpretation is reversed: 
#   - as C decreses, we are less tolerant to violations to the margin, so the margin narrows
#   - as C increses, we are more tolerant, so the margin is wider

# hope this helps.




# Ex.7 

# a) 
View(Auto)
str(Auto)
# Discretize the variable mpg into two groups
var <- ifelse(Auto$mpg > median(Auto$mpg), 1, 0)
Auto$mpglevel <- as.factor(var) # for svm library make sure dep variable is factor

Auto <- Auto[, -1] # remove the original mpg
str(Auto)

# b) 
set.seed(111)
tune.out <- tune(svm, mpglevel ~ ., data = Auto, kernel = "linear", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100, 1000)))
summary(tune.out)
# Lowest cv-error is obtained for c=1;
# best performance: ~ 0.08
# one can also unselect variable "name" (factor with 304 levels)

# c) 
set.seed(1)
tune.out <- tune(svm, mpglevel ~ ., data = Auto, kernel = "polynomial", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100), degree = c(2, 3, 4)))
summary(tune.out)
# For a polynomial kernel, the lowest cv-error is obtained 
# for a degree of 2 and a cost of 100.
# best performance: ~ 0.31

set.seed(1)
tune.out <- tune(svm,mpglevel ~ ., data = Auto, kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 5, 10, 100), gamma = c(0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)
# For a radial kernel, the lowest cv-error is obtained for a gamma 0.1 and a cost of 10.
# best performance: ~0.07

# d)
# run the models with the tuned parameters (best models)
svm.linear <- svm(mpglevel ~ ., data = Auto, kernel = "linear", cost = 1)
svm.poly <- svm(mpglevel ~ ., data = Auto, kernel = "polynomial", cost = 100, degree = 2)
svm.radial <- svm(mpglevel ~ ., data = Auto, kernel = "radial", cost = 10, gamma = 0.1)

# because the models have more than two input variables, we
# need to fix 2 selected dimensions; an example:
plot(svm.linear, Auto, weight ~ acceleration)
plot(svm.poly, Auto, weight  ~ acceleration)
plot(svm.radial, Auto, weight ~ acceleration)
# for more informative plots, see HOM, ch.14











