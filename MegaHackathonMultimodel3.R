# Loading packages
library(lubridate)
library(dplyr)
library(xgboost)
library(mice)
library(Matrix)
library(missForest)
library(mi)
library(Hmisc)

# Reading data


train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")

str(train)
colnames(train)

train$Datetime <- ymd(train$Datetime)
test$Datetime <- ymd(test$Datetime)


## Removing Item_IDs not in the testing set

train <- train[c(train$Item_ID %in% unique(test$Item_ID)),]


## Creating new Variables for the model

trainingdata2014 <- train %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime)) %>%
        mutate(LogSales = log(Number_Of_Sales)) %>%
        mutate(LogPrice = log(Price)) %>%
        filter(Year == 2014)

trainingdata2015 <- train %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime)) %>%
        mutate(LogSales = log(Number_Of_Sales)) %>%
        mutate(LogPrice = log(Price)) %>%
        filter(Year == 2015)

trainingdata2016 <- train %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime)) %>%
        mutate(LogSales = log(Number_Of_Sales)) %>%
        mutate(LogPrice = log(Price)) %>%
        filter(Year == 2016)


trainingdata201415 <- train %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime)) %>%
        mutate(LogSales = log(Number_Of_Sales)) %>%
        mutate(LogPrice = log(Price)) %>%
        filter(Year %in% c(2014,2015))


trainingdata201516 <- train %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime)) %>%
        mutate(LogSales = log(Number_Of_Sales)) %>%
        mutate(LogPrice = log(Price)) %>%
        filter(Year %in% c(2015,2016))


testingdata <- test %>%
        mutate(Day = lubridate::day(Datetime)) %>%
        mutate(Month = lubridate::month(Datetime)) %>%
        mutate(Year = lubridate::year(Datetime))



## Dropping unnecessary variables

trainingdata_final2014 <- trainingdata2014[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year", "LogSales", "LogPrice")]

trainingdata_final2015 <- trainingdata2015[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year", "LogSales", "LogPrice")]

trainingdata_final2016 <- trainingdata2016[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year", "LogSales", "LogPrice")]

trainingdata_final201415 <- trainingdata201415[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year", "LogSales", "LogPrice")]

trainingdata_final201516 <- trainingdata201516[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year", "LogSales", "LogPrice")]

testingdata_final <- testingdata[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]


## XGBoost model

xgbSales2014 <- xgboost(data = data.matrix(trainingdata_final2014[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                        label = trainingdata_final2014$LogSales, 
                        eta = 0.1, #learning rate
                        max_depth = 10, #depth of tree limited to 10
                        nrounds = 200, #No of iterations 200
                        subsample = 0.5, #subsample ratio for training
                        objective = "reg:linear", 
                        booster = "gbtree", 
                        colsample_bytree = 0.7, #subsample ratio for each tree
                        nfold = 5, #CV folds
                        maximize = FALSE)

xgbSales2015 <- xgboost(data = data.matrix(trainingdata_final2015[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                        label = trainingdata_final2015$LogSales, 
                        eta = 0.1, 
                        max_depth = 10,
                        nrounds = 200, 
                        subsample = 0.5,
                        objective = "reg:linear", 
                        booster = "gbtree", 
                        colsample_bytree = 0.7,
                        nfold = 5,
                        maximize = FALSE)


xgbSales2016 <- xgboost(data = data.matrix(trainingdata_final2016[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                        label = trainingdata_final2016$LogSales, 
                        eta = 0.1, 
                        max_depth = 10,
                        nrounds = 200, 
                        subsample = 0.5,
                        objective = "reg:linear", 
                        booster = "gbtree", 
                        colsample_bytree = 0.7,
                        nfold = 5,
                        maximize = FALSE)

xgbSales201415 <- xgboost(data = data.matrix(trainingdata_final201415[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                        label = trainingdata_final201415$LogSales, 
                        eta = 0.1, 
                        max_depth = 10,
                        nrounds = 200, 
                        subsample = 0.5,
                        objective = "reg:linear", 
                        booster = "gbtree", 
                        colsample_bytree = 0.7,
                        nfold = 5,
                        maximize = FALSE)


xgbSales201516 <- xgboost(data = data.matrix(trainingdata_final201516[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                          label = trainingdata_final201516$LogSales, 
                          eta = 0.1, 
                          max_depth = 10,
                          nrounds = 200, 
                          subsample = 0.5,
                          objective = "reg:linear", 
                          booster = "gbtree", 
                          colsample_bytree = 0.7,
                          nfold = 5,
                          maximize = FALSE)

xgb_Price2014 <- xgboost(data = data.matrix(trainingdata_final2014[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                         label = trainingdata_final2014$LogPrice, 
                         eta = 0.1, 
                         max_depth = 10, 
                         nrounds = 200, 
                         subsample = 0.5, 
                         objective = "reg:linear", 
                         booster = "gbtree", 
                         colsample_bytree = 0.7, 
                         nfold = 5, 
                         maximize = FALSE)

xgb_Price2015 <- xgboost(data = data.matrix(trainingdata_final2015[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                         label = trainingdata_final2015$LogPrice, 
                         eta = 0.1, 
                         max_depth = 10, 
                         nrounds = 200, 
                         subsample = 0.5, 
                         objective = "reg:linear", 
                         booster = "gbtree", 
                         colsample_bytree = 0.7, 
                         nfold = 5, 
                         maximize = FALSE)

xgb_Price2016 <- xgboost(data = data.matrix(trainingdata_final2016[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                         label = trainingdata_final2016$LogPrice, 
                         eta = 0.1, 
                         max_depth = 10, 
                         nrounds = 200, 
                         subsample = 0.5, 
                         objective = "reg:linear", 
                         booster = "gbtree", 
                         colsample_bytree = 0.7, 
                         nfold = 5, 
                         maximize = FALSE)

xgb_Price201415 <- xgboost(data = data.matrix(trainingdata_final201415[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                         label = trainingdata_final201415$LogPrice, 
                         eta = 0.1, 
                         max_depth = 10, 
                         nrounds = 200, 
                         subsample = 0.5, 
                         objective = "reg:linear", 
                         booster = "gbtree", 
                         colsample_bytree = 0.7, 
                         nfold = 5, 
                         maximize = FALSE)

xgb_Price201516 <- xgboost(data = data.matrix(trainingdata_final201516[c("Item_ID", "Category_1", "Category_2", "Category_3", "Day", "Month", "Year")]), 
                           label = trainingdata_final201516$LogPrice, 
                           eta = 0.1, 
                           max_depth = 10, 
                           nrounds = 200, 
                           subsample = 0.5, 
                           objective = "reg:linear", 
                           booster = "gbtree", 
                           colsample_bytree = 0.7, 
                           nfold = 5, 
                           maximize = FALSE)


## Sales Predictions 

predSales2014 <- predict(xgbSales2014, data.matrix(testingdata_final))
Pred_Sales_final2014 <- exp(predSales2014)

predSales2015 <- predict(xgbSales2015, data.matrix(testingdata_final))
Pred_Sales_final2015 <- exp(predSales2015)

predSales2016 <- predict(xgbSales2016, data.matrix(testingdata_final))
Pred_Sales_final2016 <- exp(predSales2016)

predSales201415 <- predict(xgbSales201415, data.matrix(testingdata_final))
Pred_Sales_final201415 <- exp(predSales201415)

predSales201516 <- predict(xgbSales201516, data.matrix(testingdata_final))
Pred_Sales_final201516 <- exp(predSales201516)

## Price Predictions

predPrice2014 <- predict(xgb_Price2014, data.matrix(testingdata_final))
pred_Price_final2014 <- exp(predPrice2014)

predPrice2015 <- predict(xgb_Price2015, data.matrix(testingdata_final))
pred_Price_final2015 <- exp(predPrice2015)

predPrice2016 <- predict(xgb_Price2016, data.matrix(testingdata_final))
pred_Price_final2016 <- exp(predPrice2016)

predPrice201415 <- predict(xgb_Price201415, data.matrix(testingdata_final))
pred_Price_final201415 <- exp(predPrice201415)

predPrice201516 <- predict(xgb_Price201516, data.matrix(testingdata_final))
pred_Price_final201516 <- exp(predPrice201516)


## Submission with higher weights for more recent predictions
submission1 <- data.frame(ID = testingdata$ID, Number_Of_Sales = 0.82*(0.1*Pred_Sales_final2014 + 0.15*Pred_Sales_final2015 + 0.75*Pred_Sales_final2016) + 0.11*(0.2*Pred_Sales_final201415 + 0.8*Pred_Sales_final201516), Price = 0.82*(0.1*pred_Price_final2014 + 0.15*pred_Price_final2015 + 0.75*pred_Price_final2016) + 0.11*(0.2*pred_Price_final201415 + 0.8*pred_Price_final201516))

write.csv(submission1, file = "submission39.csv", row.names = F)

