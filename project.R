install.packages("class")
install.packages("dplyr")
install.packages("e1071")
install.packages("rpart")
install.packages("ggplot2")

df <- read.csv("C://Users//Amars//OneDrive//Desktop//R_project1_400.csv", stringsAsFactors = FALSE)
df
View(df)

library(dplyr)
df <- df %>% select(-ID,-CODE_GENDER,-FLAG_MOBIL,-FLAG_WORK_PHONE,-FLAG_PHONE,-FLAG_EMAIL,
                    -OCCUPATION_TYPE)

df$DAYS_BIRTH <- abs(df$DAYS_BIRTH) / 365  
df$DAYS_EMPLOYED <- abs(df$DAYS_EMPLOYED) / 365  

# Convert specific categorical columns to factors
categorical_columns <- c( "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", 
                         "NAME_HOUSING_TYPE")
df[categorical_columns] <- lapply(df[categorical_columns], as.factor)


normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization only on numeric columns
data_normalized <- as.data.frame(lapply(df, function(x) if(is.numeric(x)) normalize(x) else x))

# Encode categorical columns as numeric
data_normalized[categorical_columns] <- lapply(data_normalized[categorical_columns], function(x) as.numeric(as.factor(x)))
str(data_normalized[categorical_columns])

set.seed(123)

trainingIdx <- sample(1:nrow(data_normalized), 0.7 * nrow(data_normalized))

train <- data_normalized[trainingIdx, ]
test <- data_normalized[-trainingIdx, ]

#KNN Model
library(class)

KNN <- knn(train = select(train,-STATUS), 
           test = select(test,-STATUS), 
           cl = train$STATUS, k = 70)

#accuracy
confusion_matrix_knn <- table(Prediction = KNN, Actual = test$STATUS)
knn_accuracy <- sum(diag(confusion_matrix_knn)) / nrow(test)
print(confusion_matrix_knn)
cat("KNN Accuracy:", knn_accuracy * 100, "%\n")

#Naive Bayes Model
library(e1071)
nb_model <- naiveBayes(select(train, -STATUS), train$STATUS)
nb_pred <- predict(nb_model, select(test, -STATUS))

#accuracy
confusion_matrix_nb <- table(Prediction = nb_pred, Actual = test$STATUS)
nb_accuracy <- sum(diag(confusion_matrix_nb)) / nrow(test)
print(confusion_matrix_nb)
cat("Naive Bayes Accuracy:", nb_accuracy * 100, "%\n")

#Decision Tree Model
library(rpart)
dt_model <- rpart(STATUS ~ DAYS_BIRTH + AMT_INCOME_TOTAL + NAME_INCOME_TYPE + DAYS_EMPLOYED, 
                  data = train, method = 'class')
dt_pred <- predict(dt_model, test, type = 'class')

#accuracy
confusion_matrix_dt <- table(Prediction = dt_pred, Actual = test$STATUS)
dt_accuracy <- sum(diag(confusion_matrix_dt)) / nrow(test)
print(confusion_matrix_dt)
cat("Decision Tree Accuracy:", dt_accuracy* 100, "%\n")

#Visualization
library(rpart.plot)

rpart.plot(dt_model, type = 5, extra = 103, fallen.leaves = TRUE, 
           main = "Decision Tree Visualization", cex = 0.7)


# K-means clustering
kmeans_model <- kmeans(data_normalized, centers = 2)

print(kmeans_model$cluster)
plot(data_normalized[, 1:2], col = kmeans_model$cluster, 
     main = "K-means Clustering of Dataset")

points(kmeans_model$centers[, 1:2], col = 1:2, pch = 8, cex = 2)

true_labels <- df$STATUS

cluster_mapping <- as.factor(kmeans_model$cluster)
mapped_labels <- as.factor(ifelse(cluster_mapping == 1, 0, 1))


confusion_matrix_kmeans <- table(Predicted = mapped_labels, Actual = true_labels)
print(confusion_matrix_kmeans)

kmeans_accuracy <- sum(diag(confusion_matrix_kmeans)) / sum(confusion_matrix_kmeans)
cat("K-means Accuracy:", kmeans_accuracy * 100, ---"%\n")



# Compare ALL the ACCURACIES -------------------------------------------------------
model_names <- c("KNN", "Naive Bayes", "Decision Tree", "K-means Clustering")
accuracies <- c(knn_accuracy, nb_accuracy, dt_a=-+--+--ccuracy, kmeans_accuracy)

accuracy_df <- data.frame(Model = model_names, Accuracy = accuracies)

#bar chart
library(ggplot2)

ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  theme(text = element_text(size = 11)) +
  scale_y_continuous(labels = scales::percent_format()) +
  geom_text(aes(label = round(Accuracy * 100, 1)), vjust = -0.5)


