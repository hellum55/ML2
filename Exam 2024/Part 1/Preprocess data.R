#Read in the data and treat the strings as factors. We have to do this manually :(
library(readxl)
library(dplyr)
library(visdat)
library(DataExplorer)
library(ggplot2)

# Read the Excel file with specified column types
data_employee <- read_xls('Data.xls', 
                          na = c("", " ", "NA", "N/A", ".", "NaN", "MISSING"))
str(data_employee)
head(data_employee)
glimpse(data_employee)

#Lets check for missing variables for a start
sapply(data_employee, function(x) sum(is.na(x)))
sum(is.na(data_employee))
plot_missing(data_employee)
#42086 observaions are missing. JObEndingYear has 61% missing values. When looking at the data it looks like
#there is a '1' of the employee quit and a 'NA' if the employee stayed. RatingCeo and BusinessOutlook have
#44.05 and 43.83% respectively. The plot_missing function suggests that it is bad and we should remove them.
#Lets see later if that is necessary. 

#Lets remove some of the variables we are sure are not giving anything to the analysis. For example ID variables and timestamps:
data_employee <- data_employee %>%
  select(-c(reviewId, reviewDateTime))

#Lets convert the isCurrentJob variable into 1 and 0
data_employee$isCurrentJob <- ifelse(is.na(data_employee$isCurrentJob), 0, data_employee$isCurrentJob)
#data_employee$isCurrentJob <-ifelse(data_employee$isCurrentJob == "1",1,0)

# Calculate turnover rate
data_employee %>%
  count(isCurrentJob)

data_employee %>%
  summarize(Turnover_level = 1-(mean(isCurrentJob)))
#39.1% od the respondants are not from a current employee - Which is quite many, but the data is gathered
#for a long period of time.

#We can look at which location has the most turnover
#Department with Turnover
employee_status <- data_employee %>%
  group_by(employmentStatus) %>%
  summarize(Turnover_level = mean(isCurrentJob))

ggplot(employee_status, aes(x = employmentStatus, y = Turnover_level, fill=employmentStatus)) + 
  geom_col()+ggtitle("Location&Turnover") + 
  theme(plot.title = element_text(size = 20, face = "bold"))

# I select the columns i want to convert to factors
str(data_employee)
data_employee$ratingOverall <- factor(data_employee$ratingOverall)
data_employee$ratingCeo <- factor(data_employee$ratingCeo)
data_employee$ratingBusinessOutlook <- factor(data_employee$ratingBusinessOutlook)
data_employee$ratingWorkLifeBalance <- factor(data_employee$ratingWorkLifeBalance)
data_employee$ratingCultureAndValues <- factor(data_employee$ratingCultureAndValues)
data_employee$ratingDiversityAndInclusion <- factor(data_employee$ratingDiversityAndInclusion)
data_employee$ratingSeniorLeadership <- factor(data_employee$ratingSeniorLeadership)
data_employee$ratingRecommendToFriend <- factor(data_employee$ratingRecommendToFriend)
data_employee$ratingCareerOpportunities <- factor(data_employee$ratingCareerOpportunities)
data_employee$ratingCompensationAndBenefits <- factor(data_employee$ratingCompensationAndBenefits)
data_employee$isCurrentJob <- factor(data_employee$isCurrentJob)
data_employee$employmentStatus <- factor(data_employee$employmentStatus)
data_employee$jobEndingYear <- factor(data_employee$jobEndingYear)
str(data_employee)

#The overall task to do is to predict the overall rating of the company using the variables that are necessary. It is a large data set so it
#is important to pay attention to the feature engineering steps, so we have the best data to analyse on.

#Lets look at the data:
summary(data_employee)
#When looking at the data we can see that IsCurrentJob only has the observation of 1. We might delete this one. 
data_employee <- data_employee %>%
  select(-c(jobEndingYear, jobTitle.text, location.name))

#Lets look at the target variable and its distribution
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


########## We might need to do this #########

#It is a bit odd if we are expected to classify a multiclass model with 5 different outcomes. It seems really difficult
# so i will convert the ratingOverall into a binary class. Whether they are satisfied (rating > 3) or not satisfied (rating <= 3).
#I have no idea id this is right...
data_employee$ratingOverall <- as.numeric(data_employee$ratingOverall)
Satisfied = rep(0, length(data_employee$ratingOverall))
Satisfied[data_employee$ratingOverall >= 4] = "Satisfied"
Satisfied[data_employee$ratingOverall <= 3] = "Not.Satisfied"
data_employee=data.frame(data_employee,Satisfied)
str(data_employee)
data_employee$Satisfied = as.factor(data_employee$Satisfied)
#Remove ratingOverall
data_employee <- subset(data_employee, select = -ratingOverall)
