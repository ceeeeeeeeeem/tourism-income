# Veri setini yükleme
library(dplyr)
library(readr)
library(ggplot2)

path.expand("/Users/cemdemir/Desktop/Courses/ECO2868_MLwR/proje/modeling")

setwd("/Users/cemdemir/Desktop/Courses/ECO2868_MLwR/proje/modeling")
getwd()

datainc <- read.csv("mergeddata.csv")

# Sütun isimlerini düzenle
names(datainc)[1:2] <- c("year", "qt")
names(datainc)[3:17] <- c("tr_inc", "ind_exp", "share_TR", "food_bvg", "accomodation", "health", "transport_TR", "sp_ed_cult", "tour_serv", "transport_INT", "other_gs", "clothes", "souvenirs", "other_exp", "exc_rate")
names(datainc)[18:20] <- c("GBR", "DEU", "FRA")

View(datainc)
datainc <- subset(datainc, !grepl("Annual", qt))
datainc$qt <- factor(datainc$qt, levels = c("I", "II", "III", "IV"), labels = c("1", "2", "3", "4"))
datainc <- mutate_all(datainc, as.numeric)
is.numeric(datainc$qt)

write.csv(datainc, "datainc.csv", row.names = FALSE)
datainc$qt <- factor(datainc$qt, levels = c(1, 2, 3, 4), labels = c("Q1", "Q2", "Q3", "Q4"))

dummy_quarters <- model.matrix(~ 0 + qt, data = datainc)
datainc <- cbind(datainc, dummy_quarters)

# Yalnızca 2012-2019 yıllarını içeren veri setini oluşturma
data_train <- datainc[datainc$year >= 2012 & datainc$year <= 2020, ]
data_test <- datainc[datainc$year >= 2021, ]

View(data_test)

#LM, QTsiz

lm_modelNoQ <- lm(tr_inc ~ exc_rate + GBR + DEU + FRA, data = data_train)
print("Linear Regression Model without QT Dummies:")
print(lm_modelNoQ)

#LM, QTli

lm_model <- lm(tr_inc ~ exc_rate + GBR + DEU + FRA + qtQ1 + qtQ2+ qtQ3 + qtQ4, data = data_train)
print("Linear Regression Model with QT Dummies:")
print(lm_model)



# 5. Modellerin Değerlendirilmesi
# Tahmin hatalarının hesaplanması
lm_predictNoQ <- predict(lm_modelNoQ, newdata = data_test)
lm_predictions <- predict(lm_model, newdata = data_test)
lm_errors <- data_test$tr_inc - lm_predictions
lm_mse <- mean(lm_errors^2)
print("Linear Regression Model MSE:")
print(lm_mse)

# Lineer Regresyon Modeli Tahminleri QTsiz
plot(data_test$tr_inc, type = "l", col = "blue", lwd = 2, main = "Linear Regression Predictions without QT Dummies", ylim = c(min(lm_predictNoQ), max(data_test$tr_inc)))
lines(lm_predictNoQ, col = "red", lwd = 2)
lines(lm_predictNoQ+800000, col = "green", lwd = 2, lty = 2)
legend("topleft", legend = c("Actual", "Predictions", "Predictions + 800k"), col = c("blue", "red", "green"), lwd = 2)

# Lineer Regresyon Modeli Tahminleri QTli
plot(data_test$tr_inc, type = "l", col = "blue", lwd = 2, main = "Linear Regression Predictions with QT Dummies", ylim = c(min(lm_predictions), max(data_test$tr_inc)))
lines(lm_predictions, col = "red", lwd = 2)
lines(lm_predictions+800000, col = "green", lwd = 2, lty = 2)
legend("topleft", legend = c("Actual", "Predictions", "Predictions + 800k"), col = c("blue", "red", "green"), lwd = 2)

predicted <- predict(lm_model, newdata = data_test)+800000

performance <- data.frame(Actual = data_test$tr_inc, Predicted = predicted)

print(performance)

library(xgboost)
xgb_model <- xgboost(data = as.matrix(data_train[, c("exc_rate", "GBR", "DEU", "FRA")]), 
                     label = data_train$tr_inc, nrounds = 100, verbose = 0)



# Modelin Özeti
print("Gradient Boosting Model without QT Dummies:")
print(xgb_model)

# Gradient Boosting Modeli Tahminleri
xgb_predictions <- predict(xgb_model, newdata = as.matrix(data_test[, c("exc_rate", "GBR", "DEU", "FRA")]))
xgb_errors <- data_test$tr_inc - xgb_predictions
xgb_mse <- mean(xgb_errors^2)
print("Gradient Boosting Model MSE:")
print(xgb_mse)

# Gradient Boosting Modeli Tahminleri Görselleştirme
plot(data_test$tr_inc, type = "l", col = "blue", lwd = 2, main = "Gradient Boosting Predictions Without QT Dummies")
lines(xgb_predictions, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lwd = 2)

predicted <- predict(xgb_model, newdata = as.matrix(data_test[, c("exc_rate", "GBR", "DEU", "FRA")]))
performance <- data.frame(Actual = data_test$tr_inc, Predicted = predicted)
print(performance)

### qt xgb

xgb_modelQ <- xgboost(data = as.matrix(data_train[, c("exc_rate", "GBR", "DEU", "FRA", "qtQ1", "qtQ2", "qtQ3", "qtQ4")]), 
                      label = data_train$tr_inc, nrounds = 100, verbose = 0)

# Gradient Boosting Modeli Tahminleri
xgb_predictionsQ <- predict(xgb_modelQ, newdata = as.matrix(data_test[, c("exc_rate", "GBR", "DEU", "FRA", "qtQ1", "qtQ2", "qtQ3", "qtQ4")]))+500000
xgb_errorsQ <- data_test$tr_inc - xgb_predictionsQ
xgb_mseQ <- mean(xgb_errorsQ^2)
print("Gradient Boosting Model MSE:")
print(xgb_mseQ)

# Gradient Boosting Modeli Tahminleri Görselleştirme
plot(data_test$tr_inc, type = "l", col = "blue", lwd = 2, main = "Gradient Boosting Predictions with QT Dummies")
lines(xgb_predictionsQ, col = "red", lwd = 2)
lines(xgb_predictionsQ+500000, col = "green", lwd = 2, lty = 2)
legend("topleft", legend = c("Actual", "Predicted", "Predicted + 500k"), col = c("blue", "red", "green"), lwd = 2)


predictedQ <- predict(xgb_modelQ, newdata = as.matrix(data_test[, c("exc_rate", "GBR", "DEU", "FRA", "qtQ1", "qtQ2", "qtQ3", "qtQ4")]))
performanceQ <- data.frame(Actual = data_test$tr_inc, Predicted = predictedQ) + 500000
print(performanceQ)

### Forecasting

data2023q1 <- read_csv("newtest.csv") # 2023 q1 exc_rate gbr deu fra

data2023q1
print(data2023q1)
pred2023 <- predict(lm_model, newdata = data2023q1) + 800000

print(pred2023)
# 340520.6

# xgb23
xgb_p23 <- predict(xgb_modelQ, newdata = as.matrix(data2023q1[, c("exc_rate", "GBR", "DEU", "FRA", "qtQ1", "qtQ2", "qtQ3", "qtQ4")])) + 500000
xgb_p23
# 1349362

# Tahmin yapma

# Create a vector of labels for the x-axis
x_labels <- c(paste(data_test$year, data_test$qt, sep = "-"), "2023-Q1")

# Plot the graph without default x-axis labels
plot(data_test$tr_inc, type = "l", col = "blue", lwd = 2, main = "Tourism Income Predictions", xaxt = "n", yaxt = "n", xlim = c(1, length(data_test$tr_inc) + 1), ylim = c(0, max(data_test$tr_inc, pred2023) + 1000000))
lines(xgb_predictionsQ + 500000, col = "red", lwd = 2, lty = 2)
lines(lm_predictions + 800000, col = "green", lwd = 2, lty = 2)
points(length(data_test$tr_inc) + 1, xgb_p23, col = "red", pch = 19)
points(length(data_test$tr_inc) + 1, pred2023, col = "green", pch = 19)
legend("topleft", legend = c("Actual", "GB + 500K", "LM + 800k"), col = c("blue", "red", "green"), lwd = 2)

# Add custom x-axis labels
axis(1, at = 1:length(x_labels), labels = x_labels, cex.axis = 0.8, xaxt = "s")

# Format the y-axis labels
axis(2, at = pretty(c(0, max(data_test$tr_inc, pred2023) + 1000000)), labels = format(pretty(c(0, max(data_test$tr_inc, pred2023) + 1000000)), big.mark = ",", scientific = FALSE), cex.axis = 0.8, yaxt = "s")

# Using Log With Models

offsetGBR <- abs(min(data_train$GBR)) + 1
offsetDEU <- abs(min(data_train$DEU)) + 1
offsetFRA <- abs(min(data_train$FRA)) + 1

# Apply log transformation with offset to variables
data_train$tr_inc_transformed <- log(data_train$tr_inc) # no addition of offset
data_train$exc_rate_transformed <- log(data_train$exc_rate) # no addition of offset

data_train$GBR_transformed <- log(data_train$GBR + offsetGBR)
data_train$DEU_transformed <- log(data_train$DEU + offsetDEU)
data_train$FRA_transformed <- log(data_train$FRA + offsetFRA)

data_test$exc_rate_transformed <- log(data_test$exc_rate)
data_test$GBR_transformed <- log(data_test$GBR + offsetGBR)
data_test$DEU_transformed <- log(data_test$DEU + offsetDEU)
data_test$FRA_transformed <- log(data_test$FRA + offsetFRA)

data2023q1$exc_rate_transformed <- log(data2023q1$exc_rate)
data2023q1$GBR_transformed <- log(data2023q1$GBR + offsetGBR)
data2023q1$DEU_transformed <- log(data2023q1$DEU + offsetDEU)
data2023q1$FRA_transformed <- log(data2023q1$FRA + offsetFRA)


# LM with Log

lm_modelLog <- lm(log(tr_inc) ~ exc_rate_transformed + GBR_transformed + DEU_transformed + FRA_transformed + qtQ1 + qtQ2 + qtQ3 + qtQ4, data = data_train)

lm_predictionsLog <- predict(lm_modelLog, newdata = data_test)
lm_predictionsLog23 <- predict(lm_modelLog, newdata = data2023q1)

plot(log(data_test$tr_inc), type = "l", col = "blue", lwd = 2, main = "Linear Regression Predictions with Log Values", ylim = c(min(lm_predictionsLog), max(log(data_test$tr_inc))))
lines(lm_predictionsLog, col = "red", lwd = 2)
lines(lm_predictionsLog+0.5, col = "green", lwd = 2, lty = 2)  # Add lty = 2 for dashed line
legend("topleft", legend = c("Actual", "Predictions", "Predictions + 1"), col = c("blue", "red", "green"), lwd = 2, lty = c(1, 1, 2))  # Add lty = c(1, 1, 2) for line types

# XGB with Log

xgb_modelLog <- xgboost(data = as.matrix(data_train[, c("exc_rate_transformed", "GBR_transformed", "DEU_transformed", "FRA_transformed", "qtQ1", "qtQ2", "qtQ3", "qtQ4")]), 
                      label = data_train$tr_inc_transformed, nrounds = 100, verbose = 0)
xgb_predictionsLog <- predict(xgb_modelLog, newdata = as.matrix(data_test[, c("exc_rate_transformed", "GBR_transformed", "DEU_transformed", "FRA_transformed","qtQ1", "qtQ2", "qtQ3", "qtQ4")]))
xgb_predictionsLog23 <- predict(xgb_modelLog, newdata = as.matrix(data2023q1[, c("exc_rate_transformed", "GBR_transformed", "DEU_transformed", "FRA_transformed","qtQ1", "qtQ2", "qtQ3", "qtQ4")]))

plot(log(data_test$tr_inc), type = "l", col = "blue", lwd = 2, main = "Linear Regression Predictions with Log Values", ylim = c(min(lm_predictionsLog), max(log(data_test$tr_inc))))
lines(xgb_predictionsLog, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predictions"), col = c("blue", "red", "green"), lwd = 2, lty = c(1, 1, 2))  # Add lty = c(1, 1, 2) for line types


# Tahmin 2
# Create a vector of labels for the x-axis
x_labels <- c(paste(data_test$year, data_test$qt, sep = "-"), "2023-Q1")

# Plot the graph without default x-axis labels
plot(log(data_test$tr_inc), type = "l", col = "blue", lwd = 2, main = "Tourism Income Predictions with Log Models", xaxt = "n", xlim = c(1, length(data_test$tr_inc) + 1), ylim = c(min(lm_predictionsLog), max(log(data_test$tr_inc))))
lines(lm_predictionsLog, col = "red", lwd = 2)
lines(lm_predictionsLog + 0.5, col = "green", lwd = 2, lty = 2)
lines(xgb_predictionsLog, col = "purple", lwd = 2)
points(length(data_test$tr_inc) + 1, lm_predictionsLog23, col = "green", pch = 19)
points(length(data_test$tr_inc) + 1, xgb_predictionsLog23, col = "purple", pch = 19)
legend("topleft", legend = c("Actual", "LM Log", "LM Log + 0.5", "XGB Log"), col = c("blue", "red", "green", "purple"), lwd = 2)

# Add custom x-axis labels
axis(1, at = 1:length(x_labels), labels = x_labels, cex.axis = 0.8, xaxt = "s")