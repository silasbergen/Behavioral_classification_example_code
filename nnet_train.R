library(dplyr)
library(nnet)

eagle_subset <- read.csv('https://github.com/silasbergen/XGBoost_example_code/raw/main/eagle_subset.csv')



df <- eagle_subset %>% 
  dplyr::select(risk_class, DEM_IA, TPI_IA,
                Slope_IA, d2water_IA,d2edge_IA, d2Landfills_IA,
                d2Feedlots_IA, d2streets, d2nestsV2_IA, northness,month_factor, Season_longshort, validation_set) %>%
  mutate(risk_class = factor(risk_class)) %>% 
  mutate(Season_longshort = factor(Season_longshort)) %>% 
  mutate(across(DEM_IA:d2nestsV2_IA, scale))

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

param.grid <- expand.grid(size =seq(2,20,by=2), decay = c(0, 0.001, 0.01, 0.1,0.5),skip = c(TRUE,FALSE))
nnet.train.results <- data.frame(param.grid, logloss = NA, merror = NA, train.time = NA)
nnet.testprobs <- list()

#Train first 10 combos:
for(i in 1:10) {
  print(i)
  size <- param.grid$size[i]
  decay <- param.grid$decay[i]
  skip <- param.grid$skip[i]
  t1 <- Sys.time()
  trainnet <- nnet(risk_class~., data = train, size = size,decay = decay,skip = skip,maxit = 5000)
  t2 <- Sys.time()
  predit <- predict(trainnet, test)
  metrics <- calc.metrics(test$risk_class, predit)
  train.time <- as.numeric(difftime(t2, t1, units='mins'))
  nnet.train.results[i,4:5] <- metrics
  nnet.train.results[i,6] <- train.time
  nnet.testprobs[[i]] <- predit
 # save(nnet.train.results, nnet.testprobs, file = 'nnet.train.results.RData')
}

nnet.train.results %>% head(10)



######################################
###5-fold CV and model assessment
######################################

set.seed(282828)
cv5_id <- sample(1:5, nrow(eagle_subset),replace=TRUE)



do.one.cv <- function(i)  {
  library(dplyr)
  library(nnet)
  trainids <- which(cv5_id!=i)
  testids <- which(cv5_id==i)
  traindf <- df %>% 
    dplyr::slice(trainids) %>% 
    select(risk_class:Season_longshort)%>% 
    mutate(risk_class = factor(risk_class))
  testdf <-  df %>% 
    dplyr::slice(test) %>% 
    select(risk_class:Season_longshort)%>% 
    mutate(risk_class = factor(risk_class))
  train.nnet <- nnet(risk_class~., data = traindf, size = 20,decay = 0.01,skip=TRUE,maxit = 5000)
  phat_nnet <- predict(train.nnet, newdata = testdf)
  out <- data.frame(testids, phat_nnet)
  names(out) <- c('testids','phat0','phat1','phat2')
  return(out)
}


nnet_cv_res <- sapply(1:5, do.one.cv, simplify=FALSE)
nnet_cv_results <- do.call(rbind, nnet_cv_res) %>% arrange(testids)
nnet_cv_pred <- apply(nnet_cv_results[,2:4], 1, function(x) which.max(x)-1)


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
(nnet.mroc <- multiroc(eagle_subset$risk_class, nnet_cv_results[,2:4]))
#Confusion matrix and by-class accuracy metrics:
(cmat_nnet <- caret::confusionMatrix(reference = factor(eagle_subset$risk_class), data = factor(nnet_cv_pred)))

