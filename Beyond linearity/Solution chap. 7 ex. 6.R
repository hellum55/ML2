#Chapter 7 exercise 6:####
#a)
library(ISLR)
attach(Wage)
dim(Wage)
str(Wage)
View(Wage)

library(boot)
set.seed(1)
cv.error = rep (NA, 10)
for (i in 1:10)
{
  fit.i=glm(wage~poly(age,i),data=Wage)  # notice glm here in conjunction with cv.glm function
  cv.error[i]=cv.glm(Wage, fit.i, K=10)$delta[2]
}
plot(1:10, cv.error, xlab="Degree", ylab="CV error", type="l", pch=20, lwd=2, ylim=c(1590, 1700))
min.point = min(cv.error)
sd.points = sd(cv.error)
abline(h=min.point + 0.2 * sd.points, col="red", lty="dashed")
abline(h=min.point - 0.2 * sd.points, col="red", lty="dashed")
legend("topright", "0.2-standard deviation lines", lty="dashed", col="red")
cv.error
which.min(cv.error)
# degree 4: 1594.003

#One way to do this is using hypothesis tests such as Analysis-of-variance (anova), 
# where we look at the decrease in RSS between models evaluated 
# H0: is that a model M1 is sufficient to explain the data
# H1: is that a more complex model M2 is required
# anova can be used only with nested models: meaning the predictors in M1 must be a subset of the predictors in M2
fit.1=lm(wage~age,data=Wage)
fit.2=lm(wage~poly(age,2),data=Wage)
fit.3=lm(wage~poly(age,3),data=Wage)
fit.4=lm(wage~poly(age,4),data=Wage)
fit.5=lm(wage~poly(age,5),data=Wage)
fit.6=lm(wage~poly(age,6),data=Wage)
fit.7=lm(wage~poly(age,7),data=Wage)
fit.8=lm(wage~poly(age,8),data=Wage)
fit.9=lm(wage~poly(age,9),data=Wage)
fit.10=lm(wage~poly(age,10),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5, fit.6, fit.7, fit.8, fit.9, fit.10)
#Anova shows that all polynomials above degree $3$ are insignificant at $1%$ significance level. But interestincly
#the 9 degree model are significant as we saw with the CV-model.
#We can plot the poly prediction on the data:
plot(wage~age, data=Wage, col="darkgrey")
#Predictions:
agelims = range(Wage$age)
age.grid = seq(from=agelims[1], to=agelims[2])
lm.fit = lm(wage~poly(age, 9), data=Wage)
lm.pred = predict(lm.fit, newdata=list(age=age.grid), se=TRUE)

se.bands <- cbind(lm.pred$fit + 2 * lm.pred$se.fit ,
                  lm.pred$fit - 2 * lm.pred$se.fit)

plot(wage~age, data=Wage, col="darkgrey")
lines(age.grid, lm.pred$fit, col="blue", lwd=2)
matlines(age.grid , se.bands, lwd = 1, col = "blue", lty = 3)

#b)
# Step functions 
cv.error1 = rep(NA, 10)
for (i in 2:10) {
  Wage$age.cut = cut(Wage$age, i)
  lm.fit1 = glm(wage~age.cut, data = Wage)
  cv.error1[i] = cv.glm(Wage, lm.fit1, K=10)$delta[2]
}
cv.error1
plot(2:10, cv.error1[-1], xlab="Number of cuts", ylab="CV error", type="l", pch=20, lwd=2)
which.min(cv.error1)
#When applying CV on the step function model the lowest CV-error is when k=8
#We can train the model with the whole data set with k = 8 and plot it:

lm.fit2 = glm(wage~cut(age, 8), data = Wage)
agelimits = range(Wage$age)
age.grid = seq(from=agelimits[1], to=agelimits[2])
lm.pred2 = predict(lm.fit2, data.frame(age=age.grid))
plot(wage~age, data=Wage, col="blue")
lines(age.grid, lm.pred2, col="red", lwd=2)









