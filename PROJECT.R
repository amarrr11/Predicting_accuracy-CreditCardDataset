
df <- read.csv("C://Users//Amars//OneDrive//Desktop//R_project1_400.csv", stringsAsFactors = FALSE)

View(df)


df <- df[, !names(df) %in% c("ID", "CODE_GENDER", "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", 
                             "OCCUPATION_TYPE")]


df$DAYS_BIRTH <- abs(df$DAYS_BIRTH) / 365  
df$DAYS_EMPLOYED <- abs(df$DAYS_EMPLOYED) / 365  

# Convert specific categorical columns to factors
categorical_columns <- c("FLAG_OWN_CAR", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", 
                         "NAME_HOUSING_TYPE")
df[categorical_columns] <- lapply(df[categorical_columns], as.factor)


df$STATUS <- as.factor(df$STATUS)


normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization only on numeric columns
data_normalized <- as.data.frame(lapply(df, function(x) if(is.numeric(x)) normalize(x) else x))

# Encode categorical columns as numeric
data_normalized[categorical_columns] <- lapply(data_normalized[categorical_columns], function(x) as.numeric(as.factor(x)))


set.seed(123)

trainingIdx <- sample(1:nrow(data_normalized), 0.7 * nrow(data_normalized))

train <- data_normalized[trainingIdx, ]
test <- data_normalized[-trainingIdx, ]

#KNN Model
library(class)
KNN <- knn(train = train[, -which(names(train) == "STATUS")], 
           test = test[, -which(names(test) == "STATUS")], 
           cl = train$STATUS, k = 10)

#accuracy
confusion_matrix_knn <- table(Prediction = KNN, Actual = test$STATUS)
knn_accuracy <- sum(diag(confusion_matrix_knn)) / nrow(test)
print(confusion_matrix_knn)
cat("KNN Accuracy:", knn_accuracy, "\n")

#Naive Bayes Model
library(e1071)
nb_model <- naiveBayes(train[, -which(names(train) == "STATUS")], train$STATUS)
nb_pred <- predict(nb_model, test[, -which(names(test) == "STATUS")])

#accuracy
confusion_matrix_nb <- table(Prediction = nb_pred, Actual = test$STATUS)
nb_accuracy <- sum(diag(confusion_matrix_nb)) / nrow(test)
print(confusion_matrix_nb)
cat("Naive Bayes Accuracy:", nb_accuracy, "\n")

#Decision Tree Model
library(rpart)
dt_model <- rpart(STATUS ~ ., data = train, method = 'class')
dt_pred <- predict(dt_model, test, type = 'class')

#accuracy
confusion_matrix_dt <- table(Prediction = dt_pred, Actual = test$STATUS)
dt_accuracy <- sum(diag(confusion_matrix_dt)) / nrow(test)
print(confusion_matrix_dt)
cat("Decision Tree Accuracy:", dt_accuracy, "\n")

#Visualization
library(rpart.plot)

rpart.plot(dt_model, type = 4, extra = 101, fallen.leaves = TRUE, 
           main = "Decision Tree Visualization", cex = 0.7)


#k-means clustering
kmeans_model <- kmeans(data_normalized, centers = 2)

print("Cluster Centers:")
print(kmeans_model$centers)

print(kmeans_model$cluster)

#Plot the clusters
plot(data_normalized[, 1:2], col = kmeans_model$cluster, 
     main = "K-means Clustering of Dataset")

points(kmeans_model$centers[, 1:2], col = 1:2, pch = 8, cex = 2)


#Compare ALL the ACCURACIES >>>

# Load the necessary library
library(ggplot2)

# Define the models and their accuracies
model_names <- c("KNN", "Naive Bayes", "Decision Tree")
accuracies <- c(knn_accuracy, nb_accuracy, dt_accuracy)

# Create a data frame to hold the model names and accuracies
accuracy_df <- data.frame(Model = model_names, Accuracy = accuracies)

# Plot the bar chart
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +
  scale_y_continuous(labels = scales::percent_format()) +
  geom_text(aes(label = round(Accuracy * 100, 1)), vjust = -0.5)
