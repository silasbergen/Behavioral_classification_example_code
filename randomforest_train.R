library(dplyr)
library(randomForest)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
eagle_subset <- read.csv('eagle_subset.csv')

df <- eagle_subset %>% 
  dplyr::select(risk_class, DEM_IA, TPI_IA,
         Slope_IA, d2water_IA,d2edge_IA, d2Landfills_IA,
         d2Feedlots_IA, d2streets, d2nestsV2_IA, northness,month_factor, Season_longshort, validation_set) %>%
  mutate(risk_class = factor(risk_class)) %>% 
  mutate(Season_longshort = factor(Season_longshort))

train <- df %>% filter(validation_set=='Train') %>% dplyr::select(-validation_set)
test <- df %>% filter(validation_set=='Test') %>% dplyr::select(-validation_set)


#Return log-loss and misclassification error on test set:

calc.metrics <- function(truth, phats) {
  #truth should be a factor of the true test set levels
  #phats should be a nt x 3 matrix of the predicted probabilities
  ntest <- length(truth)
  truemat <- model.matrix(~truth-1)
  logloss <- -sum(truemat*log(phats))
  predclass <- factor(apply(phats, 1, function(r) which.max(r)-1), levels = c('0','1','2'))
  confmat <- xtabs(~predclass + truth)
  ncorrect <- sum(diag(confmat))
  nmissed <- ntest-ncorrect
  misclassification_rate <- nmissed/ntest
  return(c('logloss'=logloss, 'merror'=misclassification_rate))
}


#################################
###Search a parameter grid:
#################################


rf.grid <- expand.grid(ntree = seq(50, 300, by = 50),
                       colsamplepct = seq(1,.6, by = -.1),
                       rowsamplepct = seq(1,.6, by = -.1))

rf.train.results <- data.frame(rf.grid, logloss = NA, merror = NA, train.time = NA, avg.nodes.pertree = NA)
rf.testprobs <- list()

p <- ncol(train)-1
n <- nrow(train)


for(i in 1:10) {
  print(i)
  ntree <- rf.grid$ntree[i]
  trainwithcols <- round((rf.grid$colsamplepct[i])*p)
  sampsize <- round((rf.grid$rowsamplepct[i])*n)
  t1 <- Sys.time()
  train.rf <- randomForest(risk_class~., data = train, 
                             ntree = ntree, mtry = trainwithcols, 
                             sampsize = sampsize)
  t2 <- Sys.time()  

  phat.rf <- predict(train.rf, newdata = test, type = 'prob')
  metrics <- calc.metrics(test$risk_class, phat.rf)
  train.time <- as.numeric(difftime(t2, t1, units='mins'))
  rf.train.results[i,4:5] <- metrics
  rf.train.results[i,6] <- train.time
  rf.train.results[i,7] <- mean(treesize(train.rf))
  rf.testprobs[[i]] <- phat.rf
  #save(rf.train.results, rf.testprobs, file = 'rf.train.results.RData')
}

rf.train.results %>% head(10)




######################################
###5-fold CV and model assessment
######################################

set.seed(282828)
cv5_id <- sample(1:5, nrow(eagle_subset),replace=TRUE)


do.one.rf.cv <- function(i)  {
  library(dplyr)
  library(randomForest)
  print(i)
  trainids <- which(cv5_id!=i)
  testids <- which(cv5_id==i)
  traindf <- df %>% dplyr::slice(trainids) %>% 
    select(risk_class:Season_longshort)%>% 
    mutate(risk_class = factor(risk_class))
  testdf <- df %>% dplyr::slice(testids)
  train.rf <- randomForest(risk_class~., data = traindf, 
                           ntree = 300, mtry = 9, 
                           sampsize = nrow(traindf))
  phat.rf <- predict(train.rf, newdata = testdf, type = 'prob')
  out <- data.frame(testids, phat.rf)
  names(out) <- c('testids',c('p0','p1','p2'))
  return(out)
}

rf_cv_res <- sapply(1:5, do.one.rf.cv, simplify=FALSE)
rf_cv_results <- do.call(rbind, rf_cv_res) %>% arrange(testids)
rf_cv_pred <- apply(rf_cv_results[,2:4], 1, function(x) which.max(x)-1)


##Function to compute pairwise and averaged pairwise area under ROC curve
multiroc <- function(y, pmat) {
  library(pROC)
  df <- data.frame(y,pmat) 
  names(df) <- c('y','c0','c1','c2')
  d01 <- df %>% mutate(p = c0/(c0+c1)) %>%   filter(y!=2)
  d02 <- df %>% mutate(p = c0/(c0+c2)) %>% filter(y!=1)
  d12 <- df %>% mutate(p = c1/(c1+c2)) %>% filter(y!=0)
  roc01 <- roc(factor(y)~p, data = d01)$auc %>% as.numeric
  roc02 <- roc(factor(y)~p, data = d02)$auc %>% as.numeric
  roc12 <- roc(factor(y)~p, data = d12)$auc %>% as.numeric
  overall_roc <- 2*(roc01 + roc02 + roc12)/6
  pairwise_auc <- c(roc01, roc02, roc12)
  return(list(pairwise_auc, overall_roc))
}


#Pairwise auROC:
(rf.mroc <- multiroc(eagle_subset$risk_class, rf_cv_results[,2:4]))
#Confusion matrix and by-class accuracy metrics:
(cmat_rf <- caret::confusionMatrix(reference = factor(eagle_subset$risk_class), data = factor(rf_cv_pred)))

