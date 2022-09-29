library(dplyr)
library(kknn)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
eagle_subset <- read.csv('eagle_subset.csv')

df <- eagle_subset %>% 
  dplyr::select(risk_class, DEM_IA, TPI_IA,
         Slope_IA, d2water_IA,d2edge_IA, d2Landfills_IA,
         d2Feedlots_IA, d2streets, d2nestsV2_IA, northness,month_factor, Season_longshort, validation_set) %>%
  mutate(risk_class = factor(risk_class)) %>% 
  mutate(Season_longshort = factor(Season_longshort)) %>% 
  mutate(across(.cols = DEM_IA:northness, scale))


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

param.grid <- expand.grid(k = 5:20, kernel = c('rectangular','triangular','epanechnikov','gaussian'))
knn.train.results <- data.frame(param.grid, logloss = NA, merror = NA, train.time = NA)
knn.testprobs <- list()

##Search first 10 combos:
for(i in 1:10) {
  print(i)
  t1 <- Sys.time()
  kuse <- param.grid[i,1]
  kerneluse <- as.character(param.grid[i,2])
  trainknn <- kknn(risk_class~., train= train, test = test, k = kuse, kernel = kerneluse)
  t2 <- Sys.time()
  knn.testprobs[[i]] <- trainknn$prob
  metrics <- calc.metrics(test$risk_class, trainknn$prob)
  train.time <- as.numeric(difftime(t2, t1, units='mins'))
  knn.train.results[i,3:4] <- metrics
  knn.train.results[i,5] <- train.time
  #save(knn.train.results, knn.testprobs, file = 'knn.train.results.RData')
}

knn.train.results %>% head(10)

######################################
###5-fold CV and model assessment
######################################

set.seed(282828)
cv5_id <- sample(1:5, nrow(eagle_subset),replace=TRUE)


do.one.cv <- function(i) {
  library(kknn)
  library(dplyr)
  test_ids <- which(cv5_id==i)
  train_ids <- which(cv5_id!=i)
  trainX <- df %>% 
    dplyr::slice(train_ids) 
  testX <- df%>% 
    dplyr::slice(test_ids)
  knntrain <- kknn(risk_class~ ., train = trainX, test = testX, k = 10, kernel = 'triangular')
  out <- data.frame(test_ids, knntrain$prob, knntrain$fitted.values)
  names(out) <- c('testids','phat0','phat1','phat2','predclass')
  return(out)
}



knn_cv_res <- sapply(1:5, do.one.cv, simplify=FALSE)
knn_cv_results <- do.call(rbind, knn_cv_res) %>% arrange(testids)

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
(knn.mroc <- multiroc(eagle_subset$risk_class, knn_cv_results[,2:4]))
#Confusion matrix and by-class accuracy metrics:
(cmat_knn <- caret::confusionMatrix(reference = factor(eagle_subset$risk_class), data = factor(knn_cv_results$predclass)))

