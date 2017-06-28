#Udacity Time Series
library(tseries)
library(zoo)
library(xts)
library(lubridate)
library(ggplot2)

setwd("/home/fractaluser/Documents/Time Series")

tsdata <- read.csv("single-family-home-sales.csv")

tsdata$Date <- paste(tsdata$Month, "-01", sep = "")
tsdata$Date <- ymd(tsdata$Date)

ggplot(data = tsdata, aes(x = Date)) + geom_line(aes(y = Home.Sales))
head(tsdata)

tsdatafin <- ts(tsdata[c("Date", "Home.Sales")], start = 1990, frequency = 12)

HoltWinters(tsdatafin, alpha = 0.8, beta = FALSE, gamma = FALSE)

HoltWinters(tsdatafin, alpha = 0.6, beta = FALSE, gamma = FALSE)

HoltWinters(tsdatafin, alpha = 0.2, beta = FALSE, gamma = FALSE)

