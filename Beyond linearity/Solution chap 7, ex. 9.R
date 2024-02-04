set.seed(1)
library(MASS)
attach(Boston)

#Question d: ####
range(Boston$dis)
dis.grid = seq(from=range(dis)[1], to=range(dis)[2], by=0.1)
#The limits is around 1-13 regarding the IDV. We can then split it into 4 intervals 4-7-11.
#Note: bs function in R expects either df or knots argument. If both are specified, knots are ignored.
library(splines)
sp.fit = lm(nox~bs(dis, df=4, knots=c(4, 7, 11)), data=Boston)
summary(sp.fit)
#All terms in the spline function appears to be significant (very significant)
sp.pred = predict(sp.fit, list(dis=dis.grid))
plot(nox~dis, data=Boston, col="darkgrey")
lines(dis.grid, sp.pred, col="red", lwd=2)
#The smoothing line fits the data quite well. A little upward spike at the end but else, it looks
#pretty fair. It does not predict the extreme values so well.

#Question e: ####
#We fit regression splines with dfs between 3 and 16.
all.cv = rep(NA, 16)
for (i in 3:16) {
  lm.fit = lm(nox~bs(dis, df=i), data=Boston)
  all.cv[i] = sum(lm.fit$residuals^2)
}
plot(all.cv[-c(1, 2)])
which.min(all.cv[-c(1, 2)])
#From the plot we can observe that the RSS for the different DF, decreases until around DF 11,
#where it starts to increase a bit again. The best one might be around 11-12 ish.

#Question f: ####
library(boot)
all.cv = rep(NA, 16)
for (i in 3:16) {
  lm.fit = glm(nox~bs(dis, df=i), data=Boston)
  all.cv[i] = cv.glm(Boston, lm.fit, K=10)$delta[2]
}
plot(3:16, all.cv[-c(1, 2)], lwd=2, type="l", xlab="df", ylab="CV error")
which.min(all.cv)
#the DF with the lowest CV-error is when DF is 12, which might be preferable. 
