# Load necessary libraries
library(sparklyr)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(pROC)
library(gridExtra)

# Connect to Spark
sc <- spark_connect(master = "local")

# Đọc dữ liệu từ CSV
path <- 'D:/BigData/heart_disease.csv'
if (!file.exists(path)) {
  write.csv(read.csv(text = 'D:/BigData/heart_disease.csv'), path, row.names = FALSE)
}
# Read the CSV file into a Spark DataFrame
df <- spark_read_csv(sc, name = "heart_disease", path = path, header = TRUE, infer_schema = TRUE)
df_pd <- collect(df)

# Loại bỏ các cột không cần thiết
df_pd <- df_pd %>%
  select(-High_Blood_Pressure, -Sugar_Consumption, -Sleep_Hours, -Stress_Level)

# Kiểm tra tên cột và tỷ lệ lớp
print("Column names:")
print(colnames(df_pd))
print("Class distribution:")
print(prop.table(table(df_pd$Heart_Disease_Status)))

# thay đổi nhãn cho biến phân loại
df_pd <- df_pd %>%
  mutate(
    Heart_Disease_Status = factor(ifelse(Heart_Disease_Status == 1, "Heart_disease", "No_heart_disease"),
                                  levels = c("No_heart_disease", "Heart_disease")),
    Gender = factor(ifelse(Gender == 1, "Male", "Female"),
                    levels = c("Female", "Male")),
    Diabetes = factor(ifelse(Diabetes == 1, "Diabetes", "No_diabetes"),
                      levels = c("No_diabetes", "Diabetes")),
    Low_HDL_Cholesterol = factor(ifelse(Low_HDL_Cholesterol == 1, "Low_HDL_cholesterol", "No_low_HDL_cholesterol"),
                                 levels = c("No_low_HDL_cholesterol", "Low_HDL_cholesterol")),
    High_LDL_Cholesterol = factor(ifelse(High_LDL_Cholesterol == 1, "High_LDL_cholesterol", "No_high_LDL_cholesterol"),
                                  levels = c("No_high_LDL_cholesterol", "High_LDL_cholesterol")),
    Smoking = factor(ifelse(Smoking == 1, "Smoker", "Non_smoker"),
                     levels = c("Non_smoker", "Smoker")),
    Family_Heart_Disease = factor(ifelse(Family_Heart_Disease == 1, "Yes", "No"),
                                  levels = c("No", "Yes"))
  )

# Visualizations
# Biểu đồ 1: Đếm số ca bệnh tim
p1 <- ggplot(df_pd, aes(x = Heart_Disease_Status)) +
  geom_bar(fill = "lightblue") +
  labs(title = "Count of Heart Disease Cases", x = "Heart Disease Status", y = "Count") +
  theme_minimal()
print(p1)

# Biểu đồ 2: Phân bố tuổi theo trạng thái bệnh tim
p2 <- ggplot(df_pd, aes(x = Age, fill = Heart_Disease_Status)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Age by Heart Disease Status", x = "Age", fill = "Heart Disease Status") +
  theme_minimal()
print(p2)

# Biểu đồ 3: Phân bố đường huyết lúc đói theo trạng thái bệnh tim
p3 <- ggplot(df_pd, aes(x = Fasting_Blood_Sugar, fill = Heart_Disease_Status)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Fasting Blood Sugar by Heart Disease Status",
       x = "Fasting Blood Sugar", fill = "Heart Disease Status") +
  theme_minimal()
print(p3)

# Biểu đồ 4: Boxplot của huyết áp theo trạng thái bệnh tim
p4 <- ggplot(df_pd, aes(x = Heart_Disease_Status, y = Blood_Pressure, fill = Heart_Disease_Status)) +
  geom_boxplot() +
  labs(title = "Boxplot of Blood Pressure by Heart Disease Status", x = "Heart Disease Status", y = "Blood Pressure") +
  theme_minimal()
print(p4)

# Biểu đồ 5: Tình trạng hút thuốc theo trạng thái bệnh tim
p5 <- ggplot(df_pd, aes(x = Smoking, fill = Heart_Disease_Status)) +
  geom_bar(position = "dodge") +
  labs(title = "Smoking Status by Heart Disease Status", x = "Smoking Status", y = "Count", fill = "Heart Disease Status") +
  theme_minimal()
print(p5)

# Biểu đồ 6: phân tách BMI và Cholesterol
p6 <- ggplot(df_pd, aes(x = BMI, y = Cholesterol_Level, color = Heart_Disease_Status)) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatterplot of BMI vs Cholesterol Level", x = "BMI", y = "Cholesterol Level", color = "Heart Disease Status") +
  theme_minimal()
print(p6)

# Prepare the data with high-risk group
df_pd <- df_pd %>%
  mutate(high_risk = factor(ifelse(Age >= 40 & Fasting_Blood_Sugar <= 125, "High_risk", "Low_risk"),
                            levels = c("Low_risk", "High_risk")))

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(df_pd$Heart_Disease_Status, p = 0.8, list = FALSE)
train <- df_pd[train_index, ]
test <- df_pd[-train_index, ]

# Train a Random Forest model
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Xác định tham số
param_grid <- expand.grid(mtry = c(2, 5, 10, 15))

rf_model <- train(
  Heart_Disease_Status ~ Age + Gender + Blood_Pressure + Cholesterol_Level +
    Exercise_Habits + Smoking + Family_Heart_Disease + Diabetes + BMI +
    Low_HDL_Cholesterol + High_LDL_Cholesterol + Triglyceride_Level +
    Fasting_Blood_Sugar + CRP_Level + Homocysteine_Level + high_risk,
  data = train,
  method = "rf",
  trControl = train_control,
  tuneGrid = param_grid,
  ntree = 500,
  metric = "ROC"
)

# Print the best model
print("Best Random Forest Model:")
print(rf_model)

# Predictions
rf_model_pred <- predict(rf_model, newdata = test)
test$predicted <- rf_model_pred
test$probability <- predict(rf_model, newdata = test, type = "prob")[, "Heart_disease"]

# Evaluation
confusion_matrix <- confusionMatrix(test$predicted, test$Heart_Disease_Status)
print("Confusion Matrix:")
print(confusion_matrix)

rf_acc <- confusion_matrix$overall['Accuracy']
rf_prec <- posPredValue(test$predicted, test$Heart_Disease_Status, positive = "Heart_disease")
roc_obj <- roc(as.numeric(test$Heart_Disease_Status == "Heart_disease"), test$probability)
rf_roc <- auc(roc_obj)

rf_dict <- list(Accuracy = round(rf_acc, 4),
                `ROC Score` = round(rf_roc, 4))
print("Model Performance Metrics:")
print(rf_dict)

# Plot ROC curve
plot(roc_obj, main = "ROC Curve for Random Forest Model", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

# Tính đặc trưng
rf_importance <- varImp(rf_model)
print("Feature Importance:")
print(rf_importance)
plot(rf_importance, top = 10, main = "Top 10 Important Features in Random Forest")

# Disconnect from Spark
spark_disconnect(sc)