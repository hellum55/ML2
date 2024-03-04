##############################################################
# Support Vector Machines (SVM)
# Ch.14, HOM with R
# Adv: guarantee to find a global optimum; robust to outliers
# Disadv: slow to train n tall data n >> p
##############################################################
# Data: Attrition data
# Problem type: supervised binomial classification
# Response variable: Attrition (i.e., “Yes”, “No”)
# Features: 30
# Observations: 1,470
# Objective: use employee attributes to predict if they will attrit (leave the company)
# Access: provided by the rsample package (Kuhn and Wickham 2019)
# More details: See ?rsample::attrition


# Note: In the ISL, implementation using library(e1071)
# New here: implementing SVMs in library (caret)


# Helper packages
library(dplyr)    # for data wrangling
library(ggplot2)  # for awesome graphics
library(rsample)  # for data splitting
library(modeldata) # for data set Job attrition


# Modeling packages
library(caret)    # meta-engine for SVMs
library(kernlab)  # also for fitting SVMs 

# Model interpretability packages
library(pdp)      # for partial dependence plots, etc.
library(vip)      # for variable importance plots



# Access data
data(attrition)
dim(attrition)
## [1] 1470   31

head(attrition$Attrition)
prop.table(table(attrition$Attrition)) 
#No       Yes 
#0.8387755 0.1612245 # unbalanced

str(attrition) # evaluate variables' types 
df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE) 
str(df)


# Create training (70%) and test (30%) sets
set.seed(123)  # for reproducibility
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)


# Tune and fit an SVM with a radial basis kernel (C and sigma as hyperparameters)
# below we use caret’s train() function to tune and train an SVM using the radial basis 
# kernel function with autotuning for the sigma parameter (i.e., "svmRadialSigma") and 10-fold CV.

# Tune an SVM with radial basis kernel 
set.seed(1854)  
churn_svm <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Plot results
ggplot(churn_svm) + theme_light()


# Print results
churn_svm$results 
# notice the default is accuracy


# RE-run with trContol to get the class probabilities for AUC/ROC     
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,             
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  
churn_svm_auc <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # explicitly set area under ROC curve as criteria        
  trControl = ctrl,
  tuneLength = 10
)

confusionMatrix(churn_svm_auc)
# interpret


# Feature importance 
# Create a wrapper to retain the predicted class probabilities for the class of interest (in this case, Yes)
prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}
#variable importance plot
set.seed(2827)  # for reproducibility
vip(churn_svm_auc, method = "permute", nsim = 5, train = churn_train, 
    target = "Attrition", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes)

#construct PDP (feature effect plots are on the probability scale)
features <- c("OverTime", "WorkLifeBalance", 
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
  partial(churn_svm_auc, pred.var = x, which.class = 2,        # since the predicted probabilities from our model come in two columns (No and Yes), we specify which.class = 2 so that our interpretation is in reference to predicting Yes
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})
grid.arrange(grobs = pdps,  ncol = 2)
# interpret




