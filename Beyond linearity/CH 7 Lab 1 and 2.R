#test#
# *************************************************************************
# LAB with discussions - Chapter 7 Non-linear Modeling -
# *************************************************************************
library(ISLR) 
attach(Wage)
dim(Wage)
str(Wage)
View(Wage)

# ***********************************************
# 7.8.1. Polynomial Regression and Step Functions
# ***********************************************

# ***
# Objective: Predict "Wage" based on Age.  
# Wage is a continuous (numerical) random var. Age is also continuous.
plot(age, wage) # the relationships does not seem to be linear
# We try a 4-degree polynomial linear regression lm()
# The poly() function:
# * helps us to avoid typing long formulas with powers of age
# * generates a basis for orthogonal polynomials terms (i.e. combinations of the original terms); 
# * orthogonal means uncorrelated predictors. 
fit=lm(wage~poly(age,4),data=Wage)  
summary(fit)
# if raw = T, poly() uses the polynomials terms (age, age^2, age^3, age^4) - it does not create orthogonal terms. 
fit2=lm(wage ~ poly(age,4,raw=T),data=Wage)  
fit2$model # Check
summary(fit2)
# We get different coefficients using the two methods. Why? 
# Still, the predictions will be almost the same - see below.


# Predictions 
# Create a grid of values for age for which we want to predict salary  
agelims=range(age)
age.grid=seq(from=agelims[1],to=agelims[2])
# Generate the predicted values + standard error bands 
preds=predict(fit,newdata=list(age=age.grid),se=TRUE) # preds has two objects "preds$fit" and "preds$se.fit"
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)
# Plot the data and the predictions 
par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0)) 
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Degree-4 Polynomial",outer=T)
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
# gam package will do this plot automatically (see later) 


# Other equivalent ways of fitting a polynomial model
fit2a = lm(wage ~ age + I(age^2) + I(age^3) + I(age^4),data=Wage) # I() is a wrapper function to protect ^ to be correctly interpreted
coef(fit2a) # the coeficients are similar to fit2 (non-orthogonal); but the predictions are the same, as we will see in a moment. 
fit2b = lm(wage ~ cbind(age,age^2,age^3,age^4),data=Wage)
coef(fit2b) # IDEM 


# Let us use fit2a to predict 
preds2a = predict(fit2a,newdata=list(age=age.grid),se=TRUE)
max(abs(preds$fit-preds2a$fit)) # the maximum difference btw. predictions fit fit and fit2a is very very small
# Concl:  the fitted values obtained in either case are almost identical

# Let us use fit2b to predict 
preds2b = predict(fit2b,newdata=list(age=age.grid),se=TRUE)
max(abs(preds$fit-preds2b$fit))
mean(preds$fit-preds2b$fit)
# Concl:  the fitted values obtained in either case are almost identical
# Final note: here we do not interpret each beta coefficient separately; 
# rather we are interested in the form of the overall function. How age determines salary, overall? 



# ****************************************************************************************
# When applying polynomial, we need to decide on the DEGREE of the polynomial 

# a.) One way to do this is using hypothesis tests such as Analysis-of-variance (anova), 
# where we look at the decrease in RSS between models evaluated 
# H0: is that a model M1 is sufficient to explain the data
# H1: is that a more complex model M2 is required
# anova can be used only with nested models: meaning the predictors in M1 must be a subset of the predictors in M2
fit.1=lm(wage~age,data=Wage)
fit.2=lm(wage~poly(age,2),data=Wage)
fit.3=lm(wage~poly(age,3),data=Wage)
fit.4=lm(wage~poly(age,4),data=Wage)
fit.5=lm(wage~poly(age,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
# Concl: Model 3 (maybe 4) is sufficient. The decrease in RSS is not anymore significant at 5% for model 4 and 5
coef(summary(fit.5)) # In fit.5 model, the p-value for the age^ 5 also reflects that. 


# Including education in the model
fit.1=lm(wage~education+age,data=Wage)
fit.2=lm(wage~education+poly(age,2),data=Wage)
fit.3=lm(wage~education+poly(age,3),data=Wage)
anova(fit.1,fit.2,fit.3)
# Let us conclude


# b.) We can use cross-validation (ch. 5) to select the best model degree, where we aim to choose 
# the model with the lowest test MSE (mean square error); MSE = RSS/n.
# K = 10-fold cross-validation 
library(boot)
set.seed(19)
cv.error = rep (0, 5)
for (i in 1:5)
{
  fit.i=glm(wage~poly(age,i),data=Wage)  # notice glm here in conjunction with cv.glm function
  cv.error[i]=cv.glm(Wage, fit.i, K=10)$delta[1]
}
cv.error # the CV errors of the five polynomials models
# Concl: A 5 order model is not justified.



# ****************************************************************************************
# Polynomial logistic regression (glm())
# Background: Plotting the wage data it seems there are two subpopulations we can distingush: 
# Low earners (<250.000) and High earners (>250.000)
# hist(wage)
# Objective: set a model to predict if an individual earns more than $250.000 per year

fit = glm(I(wage>250) ~ poly(age,4),data=Wage,family=binomial) # function I() creates automatically the binary variable Wage 
preds = predict(fit,newdata=list(age=age.grid),se=T) # predict 
preds # note the predictions; they are negative values ; they represent "logits" of wage - not probabilities - because we used the default (type = "link")
# we need to tranform logits into probabilities - see formula p. 291 ISL
pfit=exp(preds$fit)/(1+exp(preds$fit)) 
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit) # add the the bands +-2*SE with cbind
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit)) # transform them into probabilities

# Alternatively, if we explicitly set type = "response" in the predict() function, we could directly compute the probabilities 
# preds=predict(fit,newdata=list(age=age.grid),type="response",se=T) 
# But we cannot get the 95% CI in terms of probabilities in this case.
# In the online video, the author shows another method.
# Now let us plot the predicted vs the actual (Figure 7.1 right hand from ISL) 
plot(age,I(wage>250),xlim=agelims,type="n",ylim=c(0,.2))
points(jitter(age), I((wage>250)/5),cex=.5,pch="|",col="darkgrey")
lines(age.grid,pfit,lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
# be patient ....
# Grey marks (top and bottom) are ages of high earners and low earners; blue line represent pfit (fitted probabilities). 



# ****************************************************************************************
# Step functions 
# Discretize the independent variables and fit a constant in each bin 
table(cut(age,4)) # cut() function discretize the variable in 4 intervals of equal length 
fit = lm(wage ~ cut(age,4), data = Wage)  
# one category us the reference category ; in this case age < 33.5 is the ref category
contrasts (cut(age,4)) # ?contrasts to learn about other contrasts and how to set them. 
round(coef(summary(fit)),2)
# $94.16 is the avg. wage for the ref category; the rest are interpreted as ...see full explanation in ISL p.292



# ***********************
# 7.8.2 Splines
# ***********************
library(splines)
# a) Regression splines
# Choosing 3 knots manually: 25, 40, 60 
fit=lm(wage~bs(age,knots=c(25,40,60)),data=Wage) # bs() function 
pred=predict(fit,newdata=list(age=age.grid),se=T)
# Plot
plot(age,wage,col="gray")
lines(age.grid,pred$fit,lwd=2)
lines(age.grid,pred$fit+2*pred$se,lty="dashed")
lines(age.grid,pred$fit-2*pred$se,lty="dashed")

# Set knots at uniform quatiles of the data by setting the degrees of freedom, df() 
dim(bs(age,knots=c(25,40,60)))
dim(bs(age,df=6))
attr(bs(age,df=6),"knots") # the corresponding knots are 33.75, 42.0 and 51.0
# ...and re-fit the model
fit=lm(wage~bs(age,df=6),data=Wage) 
pred=predict(fit,newdata=list(age=age.grid),se=T)
# ...and plot 
plot(age,wage,col="gray")
lines(age.grid,pred$fit,lwd=2)
lines(age.grid,pred$fit+2*pred$se,lty="dashed")
lines(age.grid,pred$fit-2*pred$se,lty="dashed")



# b) Natural splines 
# e.g. for df = 4  (or we can specify the knots directly using the knots() function as before)
dim(ns(age,df=4))
attr(ns(age,df=4),"knots") 
fit2=lm(wage~ns(age,df=4),data=Wage)# using ns() function  
pred2=predict(fit2,newdata=list(age=age.grid),se=T)
lines(age.grid, pred2$fit,col="red",lwd=2) # we plot on the top of the regression splines so we can compare



# c) Smoothing splines 
# they have a smoothing parameter, lamda, which can be specified using degrees of freedon (df)
fit=smooth.spline(age,wage,df=16) # we select subjectively 16 df
fit2=smooth.spline(age,wage,cv=TRUE) # the software selects automatically df by using cross validation (cv = TRUE); 
# the default in most spline software is either leave-one-out CV
fit2
# PRESS is the “prediction sum of squares”, i.e., the sum of the squared leave- one-out prediction errors.
fit2$df # selected df based on cv is 6.79 
fit2$lambda # selected lambda
# λ → ∞, having any curvature at all becomes infinitely penalized, and only linear functions are allowed
# as λ → 0, we decide that we don’t care about curvature
# we select λ by cross-validation but smooth.spline does not let us control λ directly
# check also: $x component, re-arranged in increasing order
# a $y component of fitted values, 
# a $yin component of original values, etc. 
# See help(smooth.spline) for more



# Plot 
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Smoothing Spline")
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)



# d) Local linear regression with loess () 
fit=loess(wage~age,span=.2,data=Wage) # span = 0.2 meanind neighbourhood consists of 20% of the obs
fit2=loess(wage~age,span=.5,data=Wage) # span = 0.5 meanind neighbourhood consists of 50% of the obs
# ...and plot
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Local Regression")
lines(age.grid,predict(fit,data.frame(age=age.grid)),col="red",lwd=2)
lines(age.grid,predict(fit2,data.frame(age=age.grid)),col="blue",lwd=2)
legend("topright",legend=c("Span=0.2","Span=0.5"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
# note that the larger the span the smoother the fit



# ****************************************************
# 7.8.3 GAMs model - # more than one predictor 
# ****************************************************
# Objective: to predict Wage using functions of Year and Age, and 
# treating Education as qualitative with n levels 
# (it will be converted in n-1 dummy variables)

# A GAM using a natural spline 
gam1=lm(wage~ns(year,4)+ns(age,5)+education,data=Wage)
summary(gam1)

# A GAM using a smooth spline - called s() in gam library
library(gam) # make sure it is installed
gam.m3=gam(wage~s(year,4)+s(age,5)+education,data=Wage) 
summary(gam.m3)
# plot
par(mfrow=c(1,3))
plot(gam.m3, se=TRUE,col="blue",main="GAM  using smooth splines" )

# plot also gam1
par(mfrow=c(1,3))
plot.Gam(gam1, se=TRUE, col="red", main="GAM  using natural splines") 

# Now, which model is best? Let us run two more models and use anova to compare them
gam.m1=gam(wage ~ s(age,5)+education,data=Wage) # GAM that excludes year (called, M1)
gam.m2=gam(wage ~ year+s(age,5)+education,data=Wage) # GAM that includes a linear function of year (M2)
anova(gam.m1,gam.m2,gam.m3,test="F")
# M2 is preferred. 
# A GAM with a linear function of year is better that a GAM that excludes the year. A GAM with a nonlinear function of year is not needed. 
summary(gam.m3) # looking at the p-values in the ANOVA for Nonparametric Effects, it reinforces our previous conclusion 
# Let us make predictions using predict () employing our best model 
preds=predict(gam.m2,newdata=Wage) # note here we make predictions using the actual training data set; ideally split it before training.


# Optional 
# Using GAM with local regression lo() 
#gam.lo=gam(wage~s(year,df=4)+lo(age,span=0.7)+education,data=Wage)
#plot.Gam(gam.lo, se=TRUE, col="green")
# Using GAM with lo() and interaction : year X age
#gam.lo.i = gam(wage~lo(year,age,span=0.5)+education,data=Wage)
#library(akima) # install it first
#par(mfrow=c(1,1))
#plot(gam.lo.i)



# ****************************************************************************************** 
# Fitting a logistic regression GAM (if Y is binary)
# We fit a GAM to the wage data to predict the probability that an individual exceeds $250.000 per year 
gam.lr = gam(I(wage>250) ~ year + s(age,df=5) + education,family=binomial,data = Wage) #  a smooth splines with df = 5 for age and a step function for education  
par(mfrow=c(1,3))
plot(gam.lr,se=T,col="green") # last plot looks suspicios 
table(education,I(wage>250)) # there are no high earners in the HS category

# Refit the model using all except this category of education
gam.lr.s = gam(I(wage>250) ~ year + s(age,df=5) + education,family=binomial,data = Wage,subset=(education!="1. < HS Grad"))
plot(gam.lr.s,se=T,col="green")
# As all plots have an identical scale, we can assess the relative contributions of the 3 variables: 
# Age and education have a relatively larger effect on the probability of earning more than 250.000 per year (p. 287 ISL)


# Do we need a nonlinear term for year? 
# Use anova for comparing the previous model with a model that includes a smooth spline of year with df=4
gam.y.s = gam(I(wage>250) ~ s(year, 4) + s(age,5) + education,family=binomial,data = Wage,subset=(education!="1. < HS Grad")) 
anova(gam.lr.s,gam.y.s, test="Chisq") #  Chi-square test as Dep variable is categorical 
# We do not need a non-linear term for year.

# End 
# In addition to the textbook, there are good discussions of splines in:
# 1) Simonoff (1996, ch.5), Smoothing Methods in Statistics. Berlin: Springer- Verlag.
# 2) Hastie et al. (2009, ch.5) The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Berlin: Springer, 
#    2nd edn. URL http://www-stat.stanford.edu/~tibs/ ElemStatLearn/.
# 3) Wasserman (2006, ch.5.5). All of Nonparametric Statistics. Berlin: Springer- Verlag.





