# 9.6.1. Support Vector Classifier
###########################################################################################################################

# Generating some data. In practice, the data will be given. 

set.seed(1) 
x=matrix(rnorm(20*2), ncol=2) # set up a matrix x of 20 obs, in two classes and 2 variables 
plot(x)

y=c(rep(-1,10), rep(1,10)) # generates the y variable (-1 and + 1) (there are 10 obs in each class)
x[y==1,]=x[y==1,] + 1 # for class +1, we move the mean from 0 to 1 in each of the coordinates
plot(x, col=(3-y)) # plot x by coloring the classes
# As we can see, the obs generated are not linearly separable



# Create the data and encode the reponse as a factor variable:

dat=data.frame(x=x, y=as.factor(y))



# Fit the model, specifying the ´cost´ parameter and linear kernel

library(e1071)
svmfit = svm(y~., data=dat, kernel="linear", cost=10,scale = FALSE) # scale = FALSE means the variables are not standardized
plot(svmfit, dat) # the plot
# support vectors
svmfit$index 
# summary info
summary(svmfit)



##     * Extra code for better visualization     

# * The plot above is fine but it can be improved.
# * The svm() function in e1071 library does not explicitly output neither the coefficients nor the width of the margin 
# * This can be done manually
# * The folowing code is based on the Lab1: SVM for classification, available on Blackboard.

#    - create a grid of values for X1 and x2; we write a function for it. 
#    - it uses the function expand.grid and produces coordinates of ´n*n´ points on a lattice covering the domain of x
#    - once we have the lattice, we make a prediction at each point on the lattice
#    - we then plot the lattice, color-coded according to the classification
#    - now we can see the decision boundary
#    - the support points (points on the margin, or on the wrong side of the margin) are indexed in the $index component of the fit

make.grid = function(x, n=75){
  grange=apply(x,2,range)
  x1=seq(from=grange[1,1], to=grange[2,1], length=n)
  x2=seq(from=grange[1,2], to=grange[2,2], length=n)
  expand.grid(x.1=x1, x.2=x2)
}

xgrid=make.grid(x)
ygrid=predict(svmfit,xgrid)
plot(xgrid, col=c("red", "blue")[as.numeric(ygrid)], pch=20, cex=.2)
points(x,col=3-y, pch=19)
points(x[svmfit$index,], pch=5, cex=2) # support vectors are highlighted

#     - Extracting the linear coefficients that describe the lienar bounday to be able to build the margin 
#     - Full details on this function is available on Ch. 12 ESL "Elements of Statistical Learning

beta=drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0=svmfit$rho
plot(xgrid, col=c("red", "blue")[as.numeric(ygrid)],pch=20, cex=0.2)
points(x, col=3-y, pch=19)
points(x[svmfit$index,], pch=5, cex=2) 
abline(beta0/beta[2], -beta[1]/beta[2])
abline((beta0-1)/beta[2], -beta[1]/beta[2], lty=2)
abline((beta0+1)/beta[2], -beta[1]/beta[2], lty=2)

# the output differs a bit from the Lab1 video because of the way we generated the data. 
# Lab 2 shows an example with nonlinear kernel.


##

# We continue now with the code from the book: 

#    - Change the cost of violating the margins to 0.1 => allow wide margins

svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat)
svmfit$index

#     - Cross-validation for a range of models where we vary the cost

set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)
# Concl: the best model is for c=0.1; we can access it by: 
bestmod = tune.out$best.model
summary(bestmod)


# Let us test the best model using a Testing data set

#   - Creating a test data
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
plot(xtest, col=(3-ytest)) #visualize the testing data
testdat=data.frame(x=xtest, y=as.factor(ytest))

ypred=predict(bestmod,testdat) # make predictions
table(predict=ypred, truth=testdat$y) # contingency table


#   - for cost c=0.01:

svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)




# Now let us consider a sit. when the classes are linearly separable

#  - Generate the data 
x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)

# - We set a very high cost C, and fit the model
dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)
plot(svmfit, dat)

# We can see that there are only 3 support vectors; 
# We also see there were no training errors, as the classes are separable
# BUT, we also notice the observations that are not support vectors - indicated as circles --->
# --> are very close to the decision boundary. 
# This indicates that the model might perform bad on a testing data


#  - Let us try a smaller cost (c=1) - thus allowing a wider margin

svmfit=svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)
plot(svmfit,dat)

# Concl: in this model we misclassify an observation (see the plotted figure) 
# However, we also obtain a wider margin and make use of seven support vectors
# This model might perform better than the previous one; we shall see later. 




# 9.6.2. Support Vector Machine
###########################################################################################################################

# - Generate data

set.seed(1)
x=matrix(rnorm(200*2), ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))
plot(x, col=y) # it is clear that the data is not linerly separable in the original space

train=sample(200,100) # randomly select 100 numbers out of 200 to be used for splitting the data into training and testing

# - Fit the model

svmfit=svm(y~., data=dat[train,], kernel="radial",  gamma=1, cost=1) #note we set kernel = radial; gamma and the cost;
plot(svmfit, dat[train,]) # the plot reveals a non-linear boundary between classes; it also reveals there are a fair number of training errors; 
summary(svmfit)

# - Let us increase the cost

svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1,cost=1e5)
plot(svmfit,dat[train,])

# - A cross validation model to tune the parameter ´cost´ 

set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)

# Concl: Check in the output the best model is for cost=1 and gamma =2
# Now, let us use this model to make predictions in the testing data

table(true=dat[-train,"y"], pred=predict(tune.out$best.model,newdata=dat[-train,]))
# 10 observations out of 100 are missclassfied. 



# 9.6.3. ROC Curves
###########################################################################################################################

library(ROCR)

rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

svmfit.opt=svm(y~., data=dat[train,], kernel="radial",gamma=2, cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values
par(mfrow=c(1,2))
rocplot(fitted,dat[train,"y"],main="Training Data")

# increasing gamma to 50

svmfit.flex=svm(y~., data=dat[train,], kernel="radial",gamma=50, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted,dat[train,"y"],add=T,col="red")

# doing the same for testing data

fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")
fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")

# Concl: the model with gamma 2 appear to give the most accurate results. 



# 9.6.4 SVM with Multiple Classes (one-versus-one approach)
###########################################################################################################################

set.seed(1)
x=rbind(x, matrix(rnorm(50*2), ncol=2)) # adds 50 obs more with raw bind
y=c(y, rep(0,50))
x[y==0,2]=x[y==0,2]+2
dat=data.frame(x=x, y=as.factor(y))
par(mfrow=c(1,1))
plot(x,col=(y+1))
svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1)
plot(svmfit, dat)


# 9.6.5 Application to Gene Expression Data
###########################################################################################################################

library(ISLR)
names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)
table(Khan$ytrain)
table(Khan$ytest)
dat=data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain)) # a lot of features and few obs. 

# fit the svm
out=svm(y~., data=dat, kernel="linear",cost=10, scale=FALSE)
plot(out,dat) # we cannot plot this because of multiple dimensions; The plot() will only run automatically if your data= argument has exactly three columns (one of which is a response)
plot(out, dat, x.1~x.3)
summary(out)
table(out$fitted, dat$y) # check contingency table

# test the model
dat.te=data.frame(x=Khan$xtest, y=as.factor(Khan$ytest))
pred.te=predict(out, newdata=dat.te)
table(pred.te, dat.te$y)
