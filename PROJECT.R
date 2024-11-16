# Load the dataset
df <- read.csv("C://Users//Amars//OneDrive//Desktop//R_project1_400.csv", stringsAsFactors = FALSE)

# View the dataset
View(df)

# Remove irrelevant columns
df <- df[, !names(df) %in% c("ID", "CODE_GENDER", "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", 
                             "OCCUPATION_TYPE")]

# Convert days columns to years and take absolute values for better readability
df$DAYS_BIRTH <- abs(df$DAYS_BIRTH) / 365  
df$DAYS_EMPLOYED <- abs(df$DAYS_EMPLOYED) / 365  

# Convert specific categorical columns to factors for model compatibility
categorical_columns <- c("FLAG_OWN_CAR", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", 
                         "NAME_HOUSING_TYPE")
df[categorical_columns] <- lapply(df[categorical_columns], as.factor)

# Convert target variable to a factor
df$STATUS <- as.factor(df$STATUS)

# Define normalization function for numeric data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization only to numeric columns
data_normalized <- as.data.frame(lapply(df, function(x) if(is.numeric(x)) normalize(x) else x))

# Encode categorical columns as numeric
data_normalized[categorical_columns] <- lapply(data_normalized[categorical_columns], function(x) as.numeric(as.factor(x)))

# Set seed for reproducibility and create training and testing sets
set.seed(123)
trainingIdx <- sample(1:nrow(data_normalized), 0.7 * nrow(data_normalized))
train <- data_normalized[trainingIdx, ]
test <- data_normalized[-trainingIdx, ]

# ---- K-Nearest Neighbors (KNN) Model ----

# Load necessary library for KNN
library(class)

# Apply KNN with k = 10
KNN <- knn(train = train[, -which(names(train) == "STATUS")], 
           test = test[, -which(names(test) == "STATUS")], 
           cl = train$STATUS, k = 10)

# Calculate KNN accuracy
confusion_matrix_knn <- table(Prediction = KNN, Actual = test$STATUS)
knn_accuracy <- sum(diag(confusion_matrix_knn)) / nrow(test)
print(confusion_matrix_knn)
cat("KNN Accuracy:", knn_accuracy, "\n")

# ---- Naive Bayes Model ----

# Load necessary library for Naive Bayes
library(e1071)

# Train Naive Bayes model
nb_model <- naiveBayes(train[, -which(names(train) == "STATUS")], train$STATUS)

# Predict with Naive Bayes
nb_pred <- predict(nb_model, test[, -which(names(test) == "STATUS")])

# Calculate Naive Bayes accuracy
confusion_matrix_nb <- table(Prediction = nb_pred, Actual = test$STATUS)
nb_accuracy <- sum(diag(confusion_matrix_nb)) / nrow(test)
print(confusion_matrix_nb)
cat("Naive Bayes Accuracy:", nb_accuracy, "\n")

# ---- Decision Tree Model ----

# Load necessary library for Decision Tree
library(rpart)

# Train Decision Tree model
dt_model <- rpart(STATUS ~ ., data = train, method = 'class')

# Predict with Decision Tree
dt_pred <- predict(dt_model, test, type = 'class')

# Calculate Decision Tree accuracy
confusion_matrix_dt <- table(Prediction = dt_pred, Actual = test$STATUS)
dt_accuracy <- sum(diag(confusion_matrix_dt)) / nrow(test)
print(confusion_matrix_dt)
cat("Decision Tree Accuracy:", dt_accuracy, "\n")

# Visualize the Decision Tree
library(rpart.plot)
rpart.plot(dt_model, type = 4, extra = 101, fallen.leaves = TRUE, 
           main = "Decision Tree Visualization", cex = 0.7)

# ---- K-Means Clustering ----

# Apply K-means clustering with 2 clusters
kmeans_model <- kmeans(data_normalized, centers = 2)

# Display cluster centers
print("Cluster Centers:")
print(kmeans_model$centers)

# Print cluster assignments
print(kmeans_model$cluster)

# Plot clusters
plot(data_normalized[, 1:2], col = kmeans_model$cluster, 
     main = "K-means Clustering of Dataset")
points(kmeans_model$centers[, 1:2], col = 1:2, pch = 8, cex = 2)

# ---- Accuracy Comparison Plot ----

# Load library for visualization
library(ggplot2)

# Define models and their accuracies
model_names <- c("KNN", "Naive Bayes", "Decision Tree")
accuracies <- c(knn_accuracy, nb_accuracy, dt_accuracy)

# Create a data frame for model accuracies
accuracy_df <- data.frame(Model = model_names, Accuracy = accuracies)

# Plot model accuracy comparison
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  theme(text = element_text(size = 12)) +
  scale_y_continuous(labels = scales::percent_format()) +
  geom_text(aes(label = round(Accuracy * 100, 1)), vjust = -0.5)
