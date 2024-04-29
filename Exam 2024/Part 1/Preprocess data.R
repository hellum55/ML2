#Read in the data and treat the strings as factors. We have to do this manually :(
library(readxl)
# Read the Excel file with specified column types
data_employee <- read_xls('Data.xls', 
                          na = c("", " ", "NA", "N/A", ".", "NaN", "MISSING"))

#Lets remove some of the variables we are sure are not giving anything to the analysis. For example ID variables and timestamps:
library(dplyr)
data_employee <- data_employee %>%
  select(-c(reviewId, reviewDateTime))

# I select the columns i want to convert to factors
str(data_employee)
data_employee$ratingCeo <- factor(data_employee$ratingCeo)
data_employee$ratingBusinessOutlook <- factor(data_employee$ratingBusinessOutlook)
data_employee$ratingRecommendToFriend <- factor(data_employee$ratingRecommendToFriend)
data_employee$employmentStatus <- factor(data_employee$employmentStatus)
str(data_employee)

data_employee$isCurrentJob <- ifelse(is.na(data_employee$isCurrentJob), 0, data_employee$isCurrentJob)
#The job title and location might not be needed, but lets keep them and see if they still make sense.

#The overall task to do is to predict the overall rating of the company using the variables that are necessary. It is a large data set so it
#is important to pay attention to the feature engineering steps, so we have the best data to analyse on.

#Lets look at the missing observations. It seems like there could be a lot:
library(visdat)
library(DataExplorer)
sum(is.na(data_employee))
#42086 observaions are missing

#Lets look at it visually (e.g. plot)
vis_miss(data_employee, cluster = TRUE) #visdat library
#It seems that RatingCeo, RatingBusinessOutlook, RatingRecoToFriend are missing at the same time. Maybe it is questions that some employees do not
#find really relevant or dont have an opinion on that. 

plot_missing(data_employee)
#The three variables mentioned have a high missing value percentage, and can arguably be discarded. JobEndingYear has 61% missing values which is high.
#It makes sense though because not all the employees have left the company of course. It is questionable what this variable can give as insight.
#Lets look at the data:
summary(data_employee)
#When looking at the data we can see that IsCurrentJob only has the observation of 1. We might delete this one. 
data_employee <- data_employee %>%
  select(-c(jobEndingYear, jobTitle.text, location.name))

#Lets look at the target varible and its distribution
par(mfrow=c(1,4))
hist(data_employee$ratingOverall, breaks = 20, col = "red", border = "red") 
#A bit left skewed. It is a good overall rating for the company with most observations within rating 4 and 5.
mean(data_employee$ratingOverall)
#The company has an overall rating of 3.75 - mediocre.

#Maybe we can normalize the responses to a more uniformly distribution
no_trans <- data_employee$ratingOverall
#Log transformation:
log_employee <- log(data_employee$ratingOverall)
#Yeo-Johnson
library(car)
employee_YJ <- yjPower(data_employee$ratingOverall, lambda = 0.5)
# Box-Cox transformation (lambda=0 is equivalent to log(x))
library(forecast)
employee_BC <- forecast::BoxCox(data_employee$ratingOverall, lambda="auto") 

hist(log_employee, breaks = 20, 
     col = "lightgreen", border = "lightgreen",
     ylim = c(0, 6000) )
hist(employee_BC, breaks = 20, 
     col = "lightblue", border = "lightblue", 
     ylim = c(0, 6000))
hist(employee_YJ, breaks = 20, 
     col = "black", border = "black", 
     ylim = c(0, 6000))

list(summary(no_trans),
     summary(log_employee),
     summary(employee_BC),
     summary(employee_YJ))
#Does not really seem to be worth normalizing the target variable. We can experiment when we get to the analysis, but the difference between the
#transformations are not significant. 

#We check for near zero variance 
library(caret)
caret::nearZeroVar(data_employee, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)
#We dont have any. We might have deleted them alredy. Good Job!

#scatterplot   
#pairs(data_employee)

#Lets se the different correlations between the numeric variables.
str(data_employee)
library(corrplot)
num <- c("ratingOverall", "ratingWorkLifeBalance", "ratingCultureAndValues", "ratingDiversityAndInclusion", "ratingSeniorLeadership",
         "ratingCareerOpportunities", "ratingCompensationAndBenefits", "lengthOfEmployment")
cm <- cor(data_employee[num], use = "complete.obs")
cm
corrplot(cm)
corrplot(cm, type="upper", tl.pos="d", method="number", diag=TRUE)
#It is obvioes that many variables are correlated. And many variables are really strong correlated. It might be problematic, but it makes perfectly sense.
#It would be weird if one employee rates the culture 5, and the diveristyInclusion 5, and then WorkLife balance 1. Of course they are related. Whit is great
#to see is that the taret variable does not have a high correlation to any of the variables. We might consider deleting LenghtOfEmployment because it does have
#any correlation with the target variable. The correlation is -0.04. Not much.

#Lets look at the factors
plot_bar(data_employee)
#There is no need to look at JobTitle and LocationName
#Overall it looks fine. Continue!

#Lets have a quick look at the assumptions:
library(car)
library(performance)
library(easystats)
library(see)
test_lm <- lm(ratingOverall ~ ratingOverall+ratingWorkLifeBalance+ratingCultureAndValues+ratingDiversityAndInclusion+ratingSeniorLeadership+
              ratingCareerOpportunities+ratingCompensationAndBenefits+lengthOfEmployment, data = data_employee)
summary(test_lm)
check_model(test_lm)
hist(test_lm$residuals)
plot(predict(test_lm), rstudent(test_lm))

library(readr)
write_csv(data_employee,"data_employee.csv")
