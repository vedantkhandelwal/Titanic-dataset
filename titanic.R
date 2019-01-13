#Load the dataset
dataset <- read.csv("C:\\Users\\vedant khandelwal\\Downloads\\kaggle\\titanic\\train.csv")
View(dataset)
str(dataset)
summary(dataset)
#Check the number of rows
dim(dataset)#891rows and 12 attributes

#Check the number of missing values in each attribute
sum(is.na(dataset$PassengerId))
sum(is.na(dataset$Survived))
sum(is.na(dataset$Pclass))
sum(is.na(dataset$Name))
sum(is.na(dataset$Sex))
sum(is.na(dataset$Age))#the age attribute has 177 missing values
sum(is.na(dataset$SibSp))
sum(is.na(dataset$Parch))

#Removal of NULL OR Missing values
dataset <- na.omit(dataset)

#Check the new dataset
sum(is.na(dataset$PassengerId))
sum(is.na(dataset$Survived))
sum(is.na(dataset$Pclass))
sum(is.na(dataset$Name))
sum(is.na(dataset$Sex))
sum(is.na(dataset$Age))
sum(is.na(dataset$SibSp))
sum(is.na(dataset$Parch))
dim(dataset)#now dataset has 714 rows and 12 attributes
#Removal of useless attributes
dataset=dataset[,c(2,3,5,6,7,8,10)]
#Correlation between variables
library(GGally)
ggcorr(dataset,label = TRUE)

#Make Sex,Pclass and Survived as factors
dataset$Sex <-  factor (dataset$Sex,
                        levels = c ('male', 'female'),
                        labels = c (0, 1))
dataset$Pclass <- factor(dataset$Pclass, levels = c(1,2,3))
dataset$Survived <- factor(dataset$Survived, levels = c(0,1))
str(dataset)
summary(dataset)

#####data visualization
library(mosaic)
library(manipulate)
mplot(dataset)
1
#Scaling of Age and Fare attribute
dataset[, c(4,7)] <- scale(dataset[, c(4,7)])

#Split the dataset into the training set and the  test set
library(caTools)
set.seed(123)
split <- sample.split(dataset$Survived, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset (dataset, split == FALSE)
View(training_set)
View(test_set)

#Various Model Fitting over Survived
#1. Logistic Regression Model
logisticmodel <- glm(Survived ~ ., family = 'binomial', data = training_set)

summary(logisticmodel)
#Predict the model
prediction <- predict (logisticmodel, type = 'response', newdata = test_set[-1]) 
prediction

#Make Binary Predictions
y_pred <- ifelse(prediction > 0.5, 1, 0)
y_pred
y_pred=as.factor(y_pred)
summary(y_pred)

#Make the Confusion Matrix
library(caret)
confusionMatrix(data = y_pred,reference = test_set$Survived)
#Accuracy = 82.02%

#Plotting the ROC CUrve and AUC
library(pROC)

roc_obj <- roc(as.numeric(test_set[, 1]),as.numeric (y_pred))
auc(roc_obj) 
#AUC = 0.8112

#2. k-NN model
library(class)
y_pred <- knn(train = training_set [, -1], 
              test = test_set[, -1],
              cl = training_set[, 1],
              k = 3)
y_pred
#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Survived)


#Accuracy = 83.15%

#ROC Curve
#Plotting the ROC CUrve and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.8184

#3. SVM Model kernel=linear
library(e1071)
classifier <- svm(formula = Survived ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'linear')
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-1]) 
y_pred

#Confusion Matrix 
confusionMatrix(data = y_pred,reference = test_set$Survived)

#Accuracy = 78.65%

#Find the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.774

#4. SVM kernel=radial
library(e1071)
classifier <- svm(formula = Survived ~ .,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'radial')
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-1]) 
y_pred

#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Survived)
#Accuracy = 82.58%

#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.8115

#5. CART Model
library(rpart)
classifier <- rpart(formula = Survived ~.,
                    data = training_set)

classifier

#Plotting the Decision Tree
plot(classifier)
text(classifier)

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-1], type = 'class') 
y_pred 

#Confusion Matrix 
confusionMatrix(data = y_pred,reference = test_set$Survived)

#Accuracy = 82.02%

#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7978

#6. Random Forest Model
library(randomForest)
classifier <- randomForest(x = training_set[-1],
                           y = training_set$Survived,
                           ntree = 1000)
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-1]) 
y_pred

#Confusion Matrix
confusionMatrix(data = y_pred,reference = test_set$Survived)

#Accuracy = 87.08%

#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]), as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.8581

#7. Naive Bayes 
#Fitting the Naive Bayes Model
library(e1071)
classifier <- naiveBayes(x = training_set[-1],
                         y = training_set$Survived)
classifier

#Predicting the Test Set Result
y_pred <- predict (classifier, newdata = test_set[-1]) 
y_pred

#Making the Confusion Matrix. 
confusionMatrix(data = y_pred,reference = test_set$Survived)

#Accuracy = 80.34%


#Fitting the ROC and AUC
library(pROC)
roc_obj <- roc(as.numeric(test_set[, 1]),as.numeric(y_pred))
auc(roc_obj)
#AUC = 0.7814


#Conclusion:We came to a conclusion that random forest is the most suitable model having 87.08% accuracy and 0.8581 area under ROC curve  
