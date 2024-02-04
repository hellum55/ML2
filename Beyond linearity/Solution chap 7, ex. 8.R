library(ISLR)
set.seed(1)
pairs(Auto)
#mpg appears inversely proportional to cylinders, displacement, horsepower, weight.

## Polynomial
rss = rep(NA, 10)
fits = list()
for (d in 1:10) {
  fits[[d]] = lm(mpg~poly(displacement, d), data=Auto)
  rss[d] = deviance(fits[[d]])
}
rss
anova(fits[[1]], fits[[2]], fits[[3]], fits[[4]])
#When increasing the degree of poly it is only the 2nd degree polynomial that are significant from the others

library(glmnet)
library(boot)
cv.errs = rep(NA, 15)
for (d in 1:15) {
  fit = glm(mpg~poly(displacement, d), data=Auto)
  cv.errs[d] = cv.glm(Auto, fit, K=10)$delta[2]
}
which.min(cv.errs)
cv.errs
#When applying CV-validation the lowest error can be seen when DF = 10. 
#CV.error = 17.71750

## Step functions
cv.errs = rep(NA,10)
for (c in 2:10) {
  Auto$dis.cut = cut(Auto$displacement, c)
  fit = glm(mpg~dis.cut, data=Auto)
  cv.errs[c] = cv.glm(Auto, fit, K=10)$delta[2]
}
which.min(cv.errs)
cv.errs
#Lowest CV-error = 18.90573

## Splines
library(splines)
cv.errs = rep(NA,10)
for (df in 3:10) {
  fit = glm(mpg~ns(displacement, df=df), data=Auto)
  cv.errs[df] = cv.glm(Auto, fit, K=10)$delta[2]
}
which.min(cv.errs)
cv.errs
#Lowest CV-error = 17.66726

## GAM
library(gam)
fit = gam(mpg~s(displacement, 4) + s(horsepower, 4), data=Auto)
summary(fit)
