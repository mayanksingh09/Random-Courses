#Udacity Time Series
library(forecast)
library(zoo)
library(tseries)
library(xts)
library(ggplot2)
library(lubridate)


setwd("C:/Users/Mayank/Desktop/R_Programming")

chp_data <- read.csv("champagne-sales.csv")
head(chp_data)

chp_data$Date <- ymd(paste0("0", chp_data$Month, "-01"))
ts_data <- chp_data[c("Date", "Champagne.Sales")]

ts_final <- ts(data = chp_data$Champagne.Sales, start = 2001, frequency = 12)


#HOLT-WINTERS MODEL

##The data plot
ggplot(data = ts_data, aes(x = Date)) + geom_line(aes(y = ts_data$Champagne.Sales))

##plotting ts series
plot.ts(ts_final)

log_ts <- log(ts_final)


##plotting log ts series
plot.ts(log_ts)


##Decomposing TS series
ts_decomposed <- decompose(ts_final)

##Plotting the decomposition
plot(ts_decomposed)


##forecasting
ts_forecasts <- HoltWinters(ts_final, beta = FALSE, seasonal = "multiplicative")

plot(ts_forecasts)

ts_forecasts$fitted

##Forecasts for next 6 months
ts_forecastpoints <- forecast(ts_forecasts, h = 6)

##Plotting forecasted values
plot(ts_forecastpoints)



#ARIMA Model






#Toothbrush sales#

tb_sales <- read.csv("tb-sales.csv")
tb_sales$DATE <- ymd(tb_sales$DATE)

tb_sales_ts <- ts(tb_sales$Toothbrush.Sales, frequency = 12, start = 2009)

plot.ts(tb_sales_ts)

tb_sales_decomposed <- decompose(tb_sales_ts)

plot(tb_sales_decomposed)
