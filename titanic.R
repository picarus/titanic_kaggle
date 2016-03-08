library(data.table)
library(ggplot2)
library(GGally)
library(randomForest)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

source("multiplot.R")

preprocess <- function(dt){
  dt[, Pclass := factor(Pclass, ordered = T, levels = c(1,2,3))]
  
  meanFares <- dt[, .(meanFare = mean(Fare, na.rm = T)), by = Pclass]
  
  dt[is.na(Fare), Fare:= meanFares[Pclass==dt[is.na(Fare),Pclass], meanFare] ]
  
  spltNames <- strsplit(dt$Name, c(","), fixed=T)
  dt[, LastName := sapply(spltNames, function(x) x[1])]
  restName <- sapply(spltNames, function(x) x[2])
  
  spltNames <- strsplit(restName, c("."), fixed=T)
  dt[, Title := sapply(spltNames, function(x) trimws( x[1]))]
  dt <- dt[Title == "Don", Title := "Mr"]
  dt <- dt[Title %in% c("Mme","Dona"), Title := "Mrs"]
  dt <- dt[Title == "Mlle", Title := "Miss"]
  dt <- dt[Title %in% c("Capt","Major"), Title := "Col"]
  dt[, Title :=factor(Title)]
  
  dt[ Cabin != "", CabinType := substr(Cabin, 1, 1)]
  dt[ Cabin == "", CabinType := "X"]
  dt[, CabinType := factor(CabinType)]
  
  meanAges <- dt[, .(meanAge=mean(Age, na.rm=T)), by = Title]
  #meanAgesPclass <- dt[, .(meanAge=mean(Age, na.rm=T)), by = c("Title","Pclass")]
  
  replAges <- sapply(dt[is.na(Age)]$Title, function(x){meanAges[Title==x]$meanAge})
  dt[is.na(Age), Age:= replAges]
  
  dt
} 

featureEngineering <- function(dt){
  dt[, Child := ( Age<18 )]
  dt$FamilySize <- dt$SibSp + dt$Parch + 1
  dt
}

trainDataset = data.table(read.csv(file = "train.csv", 
                                   header = T, 
                                   sep = ",", 
                                   colClasses = c('integer', 'integer', 'integer', 
                                                  'character', 'factor', 'numeric', 'numeric',
                                                  'numeric', 'character', 'numeric', 'character',
                                                  'factor') ) )




testDataset = data.table(read.csv(file = "test.csv", 
                                  header = T, 
                                  sep = ",", 
                                  colClasses = c('integer', 'integer', 
                                                 'character', 'factor', 'numeric', 'numeric',
                                                 'numeric', 'character', 'numeric', 'character',
                                                 'factor') ) )

trainDataset[, Survived := factor(Survived, levels=c(0,1))]
trainDatasetY <- trainDataset[,  c("PassengerId","Survived"), with=F] 
trainDatasetX <- trainDataset[, !c("Survived"), with=F]
trainDatasetX[, TrOrTe := "Tr"]

testDataset[, TrOrTe := "Te"]

dataset <- rbind(trainDatasetX, testDataset)

dataset <- preprocess(dataset)
dataset <- featureEngineering(dataset)

dataset <- dataset[ , !c("Name","Ticket","Cabin","LastName") ,with = F]

trainDatasetX <- dataset[TrOrTe == "Tr", !c("TrOrTe"), with = F]
testDataset <- dataset[TrOrTe == "Te", !c("TrOrTe"), with = F]

# NA´s in  Age
trainNP<-trainDatasetX[,!c("PassengerId"),with=F]
trainFull <- cbind(Survived=trainDatasetY$Survived, trainNP)

if (T) {
  ggp <- ggpairs(trainFull, aes(color=Survived))
  multiplot( ggp[1,2], ggp[1,3], ggp[1,4], ggp[1,5], ggp[1,6], ggp[1,7], cols = 2)
  multiplot( ggp[1,8], ggp[1,9], ggp[1,10], ggp[1,11], ggp[1,12], ggp[1,13], cols = 2)
}

#######################################
## Random Forest 

rf <- randomForest(trainNP, trainDatasetY$Survived, na.action=na.omit)

varImpPlot(rf)

testDatasetY <- predict(rf, testDataset, type="prob")
testDatasetY <- predict(rf, testDataset, type="response")
outRF <- data.frame(PassengerId=as.integer(testDataset$PassengerId),
                    Survived=as.integer(levels(testDatasetY))[testDatasetY])
write.csv(outRF, file = "submissionRF.csv", row.names = F, quote = F)

#######################################
## Decision Trees

fitDT <- rpart(Survived ~ ., data=trainFull, method="class")
Prediction <- predict(fitDT, testDataset, type = "class")
outDT <- data.frame(PassengerId = testDataset$PassengerId, Survived = Prediction)
write.csv(outDT, file = "submissionDT.csv", row.names = FALSE, quote =F)

######################################
## GLM
#library(glm)
#fitGLM <- glm(Survived ~ ., data = trainFull, family = binomial)

library(gbm)
trainDatasetGBMX <- trainDatasetX[,Child:=factor(Child)]

testGBM <-testDataset[,Child:=factor(Child)]
fitGBM <- gbm.fit(y = as.logical(levels(trainDatasetY$Survived)[trainDatasetY$Survived]), 
                  x = trainDatasetGBMX, 
                  n.trees=5000,
                  shrinkage = 0.01, 
                  n.minobsinnode = 5,
                  bag.fraction = 0.6,
                  interaction.depth = 3,
                  nTrain = 891*0.8,
                  distribution = "bernoulli",
                  verbose = T)
PredictionGBM <- predict(fitGBM, testDataset, 
                         n.trees=5000, type = "response")
outGBM <- data.frame(PassengerId = testDataset$PassengerId, Survived = PredictionGBM)
write.csv(outGBM, file = "submissionGBM.csv", row.names = FALSE, quote =F)

######################################
## XGBOOST




