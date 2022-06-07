library(dplyr)
library(xgboost)


#Read in data file, 10,000 observations:
eagle_subset <- read.csv('https://github.com/silasbergen/XGBoost_example_code/raw/main/eagle_subset.csv')


#Set up data structures for boosting; use sample of 10,000:
df <- eagle_subset %>% 
  dplyr::select(risk_class, DEM_IA, TPI_IA,
         Slope_IA, d2water_IA,d2edge_IA, d2Landfills_IA,
         d2Feedlots_IA, d2streets, d2nestsV2_IA, northness,month_factor, Season_longshort,validation_set)  %>% 
  mutate(month_factor = factor(month_factor, levels = month.abb)) 
train <- df %>% filter(validation_set=='Train') %>% dplyr::select(-validation_set) 
test <- df %>% filter(validation_set=='Test') %>% dplyr::select(-validation_set) 
trainX <- model.matrix(~.-1, data = train %>% dplyr::select(-risk_class))
testX <- model.matrix(~.-1, data = test%>% dplyr::select(-risk_class))
dtrain <- xgb.DMatrix(trainX, label = train$risk_class)
dtest <- xgb.DMatrix(testX, label = test$risk_class)

###################################
#Set up the parameter grid
####################################


#A reduced parameter grid just to demonstrate grid search:

grid1 <- expand.grid(M = seq(16,20,by=2),
                       csp = seq(.6,1,by=.2),
                       rsp=seq(.6,1,by=.2),
                       eta = c(0.1,0.2,0.3), lambda = c(0,1,2), gamma = c(0,1,2))

dim(grid1)


grid1_results <- list()


##Function to get the misclassification error from the XGBoost messages

getmerror <- function(msg) {
  splits <- strsplit(msg,":|\\\ttest", fixed=FALSE)
  want <- as.numeric(do.call(rbind,splits)[,3])
  return(want)
}



#Train the first 10 parameter combos:

for(i in 1:10) {
    paramvec <- grid1[i,]
    p <- with(paramvec, list(max_depth = M, 
                             colsample_bytree = csp,
                             subsample = rsp,
                             eta = eta, gamma = gamma, lambda = lambda,
                             num_class = 3))
    cat("################################\n")
    print(paste('Training ',i,'of',nrow(grid1)))
    print(Sys.time())
    print(unlist(p))
    set.seed(1122021)
    time.start <- Sys.time()
    boostit <- xgb.train(params = p, data=dtrain, nrounds = 20000,
                         objective = "multi:softprob", 
                         eval_metric="merror", 
                         eval_metric="mlogloss", 
                         watchlist = list(test = dtest), 
                         early_stopping_rounds = 10,
                         print_every_n = 50)
    time.end <- Sys.time()
    train.time <- as.numeric(difftime(time.end, time.start, units='mins'))
    print(paste('Train time = ', train.time))
    grid1_results[[i]] <- data.frame(boostit$params, bestiter = boostit$best_iteration,
                           bestscore = boostit$best_score, bestmsg = boostit$best_msg,
                           runtime = train.time,
                           merror = getmerror(boostit$best_msg))
    #save(grid1_results, file = 'grid1_results_refinedsearch.Rdata')
}

grid1_res <- do.call(rbind, grid1_results)

grid1_res %>% 
  arrange(merror) 


#############################################
##FIT XGBOOST MODEL TO ENTIRE DATA:
#############################################


X <- model.matrix(risk_class~.-1, data = df %>% dplyr::select(-validation_set))
dX <- xgb.DMatrix(X, label = df$risk_class)

best_xgmod <- xgb.train(data=dX, params = list(max_depth = 26, 
                                               eta = 0.1, 
                                               colsample_bytree = 0.9,
                                               subsample = 1,
                                               num_class = 3,
                                               gamma = 0,
                                               lambda = 2),
                        nrounds = 125,
                        objective = "multi:softprob", 
                        eval_metric = 'mlogloss')






##############################
### SHAP plotting
##############################

library(tidyr)
library(ggplot2)

shapvals <- predict(best_xgmod, newdata = X, predcontrib  = TRUE, approxcontrib = TRUE)

# Output is list length 3 (num_classes), each list containing a matrix of shap values for each obs


##Overall variable importance plot:
n <- nrow(df)

meanabs <- function(x) mean(abs(x))

shap_summary_df <- do.call(rbind, shapvals) %>% 
  data.frame() %>% 
  dplyr::select(-BIAS) %>% 
  mutate(class = rep(0:2, each = n)) %>% 
  mutate(rowid = rep(1:n, 3)) %>% 
  gather(key = variable, value = value, DEM_IA:Season_longshortLocal.movements) %>% 
  group_by(variable, class) %>% 
  summarize(meanshap =  meanabs(value)) %>% 
  ungroup(variable) %>% 
  mutate(variable = reorder(variable, meanshap, FUN = mean))
  

ggplot(data = shap_summary_df) + 
    geom_bar(aes(x = meanshap, y= variable), stat = 'identity') + 
  facet_wrap(~class) 


### Re-create d2landfill plot for high-risk (3rd array) flight:

df$landfill_shap <-  shapvals[[3]][,'d2Landfills_IA']


ggplot(data = df,aes(x = d2Landfills_IA, y = landfill_shap)) + 
  geom_point( shape='.',alpha= 0.6) + 
  geom_smooth()


### Faceted by month:

ggplot(data = df,aes(x = d2Landfills_IA, y = landfill_shap)) + 
  geom_point( shape='.',alpha= 0.6) + 
  geom_smooth() + 
  facet_wrap(~month_factor)



##############################
##PARTIAL DEPENDENCE PLOTS
##############################

#I created the ICE's manually, 
#creating a sequence along each predictor and predicting for the sequence.
#All other variables are held constant.




landvars <- df %>% 
  dplyr::select(DEM_IA, TPI_IA,
                Slope_IA, d2water_IA,d2edge_IA, d2Landfills_IA,
                d2Feedlots_IA, d2streets, d2nestsV2_IA, northness,month_factor,Season_longshort) %>% 
  data.frame()

allvars <- names(landvars)


#Using SHAP plot to order:
varnames <- c('d2water_IA','d2Landfills_IA','d2nestsV2_IA',
                            'DEM_IA','d2Feedlots_IA','d2edge_IA','Slope_IA',
                            'd2streets','TPI_IA','northness','Season_longshort','month_factor')
pdp_df_list_multiclass <- list()



##Sampling for pdps
for(i in 1:length(varnames)) {
  var <- varnames[i]
  x <- landvars[,var]
  deciles <- NULL
  if(!var %in% c('month_factor','Season_longshort')) {
    gridbounds <- quantile(x, c(.05, .95))
    grid <- seq(gridbounds[1],gridbounds[2],l=50)
    deciles <- quantile(x, seq(.1, .9, by = .1))
  }
  if(var =='month_factor') grid <- factor(month.abb)
  if(var =='Season_longshort') grid <- factor(c('Dispersal/migration','Fledgling period','Local movements'))
  set.seed(1234)
  sampind <- sample(1:nrow(landvars),5000,replace=FALSE)
  Xpred <- landvars %>% 
    dplyr::slice(sampind)%>%
    dplyr::slice(rep(1:n(),each=length(grid)))
  Xpred[,var] <- rep(grid,length(sampind))
  if(var=='month_factor') Xpred[,var] <- factor(Xpred[,var],levels=month.abb)
  Xpredmat <- model.matrix(~.-1, data = Xpred)
  yhat <- predict(best_xgmod, newdata = Xpredmat)
  yhat_mat <- matrix(yhat, nrow(Xpred),3,byrow = TRUE)
  ice <- Xpred %>% 
    mutate(p0 = yhat_mat[,1],
           p1=yhat_mat[,2],
           p2 = yhat_mat[,3]) %>% 
    dplyr::select(var, p0:p2) %>% 
    mutate(rowid = rep(1:5000,each=length(grid)))
  ice_sample <- ice %>% filter(rowid %in% sample(1:5000,100))
  names(ice)[1] <- 'ventile'
  pdp <- ice %>%
    group_by(ventile) %>% 
    summarize(across(.cols=p0:p2,.fns=list(mean = mean,sd = sd)))
  names(pdp)[1] <- var
  print(paste('done with ',var))
  pdp_df_list_multiclass[[i]] <- list('var' = var, 'pdp' = pdp, 'deciles'=deciles,'ice_sample'=ice_sample)
  #save(pdp_df_list_multiclass, file = 'pdp_df_list_multiclass_refinedsearch.Rdata')
}




###############################
##ICE/PDP plot for d2water
###############################

waterdf <- pdp_df_list_multiclass[[1]]$ice_sample %>% 
  gather(key = p, value = ice, p0:p2)

waterdeciles <- data.frame(deciles = pdp_df_list_multiclass[[1]]$deciles)

(watericeplot <- ggplot(data = waterdf) + 
    geom_line(aes(x = d2water_IA, y = ice, group=rowid),alpha = .4, size = .1) +
    stat_summary(aes(x = d2water_IA, y = ice),fun='mean',col='goldenrod',geom='line',size = 1.5) + 
    facet_wrap(~p, labeller = labeller(p = c('p0' = 'Low', 'p1' = 'Moderate','p2' = 'High'))) + 
    ylab('Probability') + xlab('D2 water (m)') + 
    geom_point(aes(x = deciles, y = -0.01), shape='|', data = waterdeciles,col='black',size=1) + 
    theme_classic() + 
    theme(axis.text = element_text(color='black', size = 8)) + 
    xlim(c(0,6203))+
    ggtitle('(A) ICE/PDP plot')) 



######################################
###PDP plots of multiple variables:
######################################

library(gridExtra)
#Build a template:

riskcols <- c('#0571b0','#bcbddc','#d95f02')
labs <- c('D2 water (m)','D2 landfill (m)','D2 nest (m)','Elevation (m)','Stage','Month')


gglist <- list() 
j <- 1

for(i in c(1:4, 11, 12)) {
  var <- pdp_df_list_multiclass[[i]]$var
  df <- pdp_df_list_multiclass[[i]]$pdp %>% 
    gather(key = class, value = pdp, p0_mean,p1_mean,p2_mean) %>% 
    mutate(class = gsub('_mean','',class)) %>% 
    mutate(feature = var)
  names(df)[1] <- 'var'
  
  
  if(!var%in% c('month_factor','Season_longshort')){
    deciles <- data.frame(deciles = pdp_df_list_multiclass[[i]]$deciles,
                          y = min(df$pdp))
    gplt <- ggplot(data = df) + 
      geom_line(aes(x = var, y = pdp, col=class,linetype=class),size=.5) + 
      scale_color_manual(values = riskcols) + 
      geom_point(aes(x = deciles, y = 0), shape='|', data = deciles,col='black',size=2) + 
      theme_classic() + 
      theme(axis.text = element_text(color='black', size = 8)) +
      ylab('Avg predicted p') + 
      guides(color='none',linetype='none') + 
      scale_y_continuous(limits = c(0, .7), name = 'Avg predicted p') + 
      xlab(labs[j])
  }
  
  if(i==1) {
    txtdf <- data.frame(x = seq(500,6000, l=3), y = rep(.7, 3), label = c('Low','Mod','High'))
    gplt <- gplt + 
      geom_text(aes(x = x, y = y, label = label),
                data = txtdf,
                col = riskcols)
  }
  if(var%in% c('month_factor','Season_longshort')){
    gplt <- ggplot(data = df, aes(x = var, y = pdp, fill=class)) + 
      geom_bar(position='stack',stat='identity') + 
      scale_fill_manual(values = riskcols) + 
      theme_classic() + 
      theme(axis.text = element_text(color='black', size = 8)) +
      ylab('Avg predicted p') + 
      guides(fill='none') + 
      xlab(labs[j])
    if(var=='Season_longshort') gplt <- gplt + scale_x_discrete(labels = c('Disp/Mig','Fledgling','Local'))
    if(var=='month_factor') gplt <- gplt + scale_x_discrete(labels = 1:12)
    
  }
  gglist[[j]] <- gplt
  j <- j + 1
}


layout <- matrix(1:6, 2,3,byrow=TRUE)
marrangeGrob(gglist,nrow=2,ncol=3,top="",layout_matrix = layout)





