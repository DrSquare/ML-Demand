# 0.1  Information #############################################################
#                                              					
#    Machine Learning Methods for Demand Estimation
#             
#    Patrick Bajari, Denis Nekipelov, Stephen Ryan and Miaoyu Yang
#              
#    Updated: 2015-1-20    
#                                              					
################################################################################

# 0.2  Load Required Packages and Functions ####################################

# Load the script which includes all the required packages and functions
setwd("../")
source("scripts/Initialize.R")

# Set Run Parameters
# If test is TRUE, this will subset the data by 10% and run the code
test <- FALSE
# If with_competition is TRUE, we include the competitor prices in our model
with_competition <- FALSE
# If vif comparison is needed
vif_comp <- FALSE

# 0.3 Read and Prepare Data ####################################################
                              
# Load the Stata 12 format data
SS <- read.dta("data/saltysnack_temp_0.05.dta")
#SS <- read.dta("data/salty_snack_temp_0.05_0.05.dta")

if (test == TRUE) {
  #subset the data frame to test the code
  set.seed(1)  
  SS$sub <- runif(dim(SS)[1],0,1)
  SS <- subset(SS,sub<=0.1)
  SS <- subset(SS,select=-sub)
}


today <- Sys.Date()   

# Define Categorical Variables
SS$p_id <- as.factor(SS$upc)
SS$iri_key <- as.factor(SS$iri_key)
SS$week <- as.factor(SS$week)
SS$producttype <- factor(SS$producttype)
SS$package <- factor(SS$package)
SS$flavorscent <- factor(SS$flavorscent)
SS$fatcontent <- factor(SS$fatcontent)
SS$cookingmethod <- factor(SS$cookingmethod)
SS$saltsodiumcontent <- factor(SS$saltsodiumcontent)
SS$typeofcut <- factor(SS$typeofcut)
SS$pr <- factor(SS$pr)
SS$f <- factor(SS$f)
SS$d <- factor(SS$d)
SS$brand <- factor(SS$l5)

# Spilt the data set into training(j=1), test(j=0) set and validation (j=2) set
set.seed(1)
SS$j <- runif(dim(SS)[1],0,1)
SS$j[SS$j>=0.4] <- 1
SS$j[SS$j<0.4 & SS$j >=0.25] <- 2
SS$j[SS$j<0.25] <- 0 

# Drop some observations where the brand level only exists in one set
k<- c("brand")
SS1 <- SS[SS$j==1,]
SS2 <- SS[SS$j==2,]
SS0 <- SS[SS$j==0,]
for (kk in k) {
  SS1[[kk]] <- factor(SS1[[kk]])
  # levels() does not drop values after subsetting, need to reapply factor()
  
  ss2_index <- SS[SS$j==2,][[kk]] %in% levels( SS1[[kk]] )
  ss0_index <- SS[SS$j==0,][[kk]] %in% levels( SS1[[kk]] )
  
  if (!all(ss2_index)) {
    print(kk)
    SS2 <- SS2[ss2_index,]
  } 
  SS2 <- droplevels(SS2)
  
  if (!all(ss0_index)) {
    print(kk)
    SS0 <- SS0[ss0_index,]
  }
  SS0 <- droplevels(SS0)
}

# Backup the data frame and check data dimentions are consistent
SS_BACK <- SS
SS <- rbind(SS0,SS1,SS2)
mapply(dim,list(SS,SS0,SS1,SS2))


# 0.4 Create Formulae ##########################################################
 
# Define X Matrix for both cases: with/without competitor prices
if (with_competition == TRUE) {
  # Grep the indices of all the competitor price variables and feed them into a formula
  # note the missing values in logprice* variables are encoded as -9999. If missings are not
  # encoded, may cause a problem in model.matrix
  comp_ind <- grep("logprice",names(SS))
  comp_price_name <- paste(names(SS)[comp_ind[-1]],sep="",collapse="+")
  
  # Version 1: with competitor price
  x_formula <- formula(paste("~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                       fatcontent+cookingmethod+saltsodiumcontent+typeofcut+iri_key+week+",
                       comp_price_name,")",sep=""))
  x_mat <- model.matrix(x_formula,data=SS)
  
  
  # Define a Covariate Matrix for Logit Share Model, Excluding All the Fixed Effects
  x_formula_no_fe <- formula(paste("~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                             fatcontent+cookingmethod+saltsodiumcontent+typeofcut+",
                             comp_price_name,")",sep=""))
  x_mat_no_fe <- model.matrix(x_formula_no_fe,data=SS)
  
  # Define Share forluma
  share_formula <- formula(paste("share~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                           fatcontent+cookingmethod+saltsodiumcontent+typeofcut+",
                           comp_price_name,"-1)",sep=""))
  
} else {
  # Version 2: without competitor price
  
  x_formula <- formula(~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+fatcontent+
                           cookingmethod+saltsodiumcontent+typeofcut+iri_key+week))
  x_mat <- model.matrix(x_formula,data=SS)
  
  y_formula <- formula(logunits~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+fatcontent+
                           cookingmethod+saltsodiumcontent+typeofcut+iri_key+week))
  
  # Define a Covariate Matrix for Logit Share Model, Excluding All the Fixed Effects
  x_formula_no_fe <- formula(~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                                 fatcontent+cookingmethod+saltsodiumcontent+typeofcut))
  x_mat_no_fe <- model.matrix(x_formula_no_fe,data=SS)
  
  # Define Share forluma
  share_formula <- formula(share~logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                           fatcontent+cookingmethod+saltsodiumcontent+typeofcut-1)
  share_formula2 <- formula(share2~logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent+
                            fatcontent+cookingmethod+saltsodiumcontent+typeofcut-1)
}

# Define Covariates Matrix x and Response Variable y by Train(train*), Validation(val*) and Test(test*)
# Rename trainx and trainy to x and y for simpilicity
testx <- x_mat[SS$j==0,]
x <- x_mat[SS$j==1,]
valx <- x_mat[SS$j==2,]

testy <- SS$logunits[SS$j==0]
y <- SS$logunits[SS$j==1]
valy <- SS$logunits[SS$j==2]

testx_no_fe <- x_mat_no_fe[SS$j==0,]
x_no_fe <- x_mat_no_fe[SS$j==1,]
valx_no_fe <- x_mat_no_fe[SS$j==2,]

# 1.0 Main Models ##############################################################
#                                              
#                                              
#                 9 Main Models and the Combined Model             
#                                              
#                                              
################################################################################

# Run 9 Models on Train Set
# 1.1 Model 1: Linear Regression ###############################################
f1 <- lm(y~x-1)

# Predict in Validation Set
f1.coef <- f1$coefficients
f1.coef[is.na(f1.coef)] <- 0 
p1 <- f1.coef %*% t(valx)

# Predict in Test Set
p1t <- f1.coef %*% t(testx)

# 1.2 Model 2: Forward Stepwise ################################################
f2 <- lars_type(t="step",x=x,y=y,step=100)

# Predict in Validation Set and Test Set
step <- length(f2$actions)
f2.coef <- predict(f2,s=step,type="coef")$coef
p2<-predict(f2,s=step,valx,type="fit",mode="step")$fit
p2t <-predict(f2,s=step,testx,type="fit",mode="step")$fit

# 1.3 Model 3: Forward Stagewise ###############################################
f3 <- lars_type(t="forward.stagewise",x=x,y=y,step=100)

# Predict in Validation Set and Test Set
step <- length(f3$actions)
f3.coef <- predict(f3,s=step,type="coef")$coef
p3<-predict(f3,s=step,valx,type="fit",mode="step")$fit
p3t <-predict(f3,s=step,testx,type="fit",mode="step")$fit

# 1.4 Model 4: LASSO ###########################################################
f4 <- lars_type(t="lasso",x=x,y=y,step=100)

# Predict in Validation Set and Test Set
step <- length(f4$actions)
f4.coef <- predict(f4,s=step,type="coef")$coef
p4<-predict(f4,s=step,valx,type="fit",mode="lambda")$fit
p4t <-predict(f4,s=step,testx,type="fit",mode="lambda")$fit

# VIF comparison between linear and LASSO selection
if (vif_comp == TRUE) {
  # see VIF after LASSO selection
  f4_index <- dimnames(x_mat)[[2]] %in% names(unlist(f4$actions))
  f4_col <- x_mat[,f4_index]

  # VIF
  vif1 <- vif(lm(logunits~(logprice+pr+f+d+brand+vol_eq+producttype+package+flavorscent
                           +fatcontent+cookingmethod+saltsodiumcontent+typeofcut),data=SS))
  vif4 <- vif(lm(SS$logunits~f4_col))
  
  vif1<- sort(vif1,decreasing=TRUE)
  vif4<- sort(vif4,decreasing=TRUE)
  unname(vif1)
  names(vif1)
  unname(vif4)
  names(vif4)
}

# 1.5 Model 5: Random Forest ###################################################
#f5x <-tuneRF(x,y,mtryStart=60,ntreeTry=50,doBest=T)
#set nodesize bigger/maxnodes smaller
f5x <-tuneRF(x,y,mtryStart=500,ntreeTry=50,nodesize=100,maxnodes=1000,doBest=T)
f5 <- randomForest(x,y,mtry=f5x$mtry,importance=T)

# Predict in Validation Set and Test Set
p5<-predict(f5,valx)
p5t <-predict(f5,testx)

# 1.6 Model 6: Support Vector Machine ##########################################
# Remove Intercept from X Matrix
x2<-x[,-grep("Intercept",names(as.data.frame(x)))]
f6 <- svm(x2,y,type="eps",kernel="linear")

# Predict in Validation Set and Test Set
valx2<-valx[,-grep("Intercept",names(as.data.frame(valx)))]
p6 <- predict(f6,newdata=as.data.frame(valx2))
testx2<-testx[,-grep("Intercept",names(as.data.frame(testx)))]
p6t <- predict(f6,testx2)

# 1.7 Model 7_new: Bagging #####################################################

f7_new <- bagging(y_formula,SS1)
p7_train <- predict(y_formula,SS1)

# Predict in Validation Set and Test Set
p7_new <- predict(f7_new,SS2)
p7t_new <- predict(f7_new,SS0)


# 1.8 Model 8: Logistic Model ##################################################
# Predict Market Size Using a Gradient Boosting Model
#f8x <- lm(logunits~iri_key+week,data=SS)
f8x <- gbm.fixed(data=as.data.frame(cbind(x_mat,SS$logunits)),1:dim(x_mat)[2],
		dim(x_mat)[2]+1,n.trees=500,family="gaussian",keep.data=TRUE)
SS$units_hat <- round(exp(f8x$fit))

# Aggregate Weekly Store Total Predicted Units and Merge Total Units (Market Size) into Data
total <- aggregate(units_hat~iri_key+week,data=SS,sum,na.rm=T)
names(total)[3] <- "total_units"
SS <- merge(SS,total,by=c("iri_key","week"))

# Calculate Predicted Market Share (A Market is a store during a week.)
SS$share <- SS$units_hat/SS$total_units

f8 <- glm(share_formula,data=SS[SS$j==1,],family=binomial)

# Predict in Validation Set
p8share <- predict(f8,newdata=SS2,type="response")
p8 <- log(p8share*SS$total_units[SS$j==2])

# Predict in Test Set
p8share.t <- predict(f8,newdata=SS0,type="response")
p8t <- log(p8share.t*SS$total_units[SS$j==0])

# 1.9 Model 9: Logistic Model (Traditional) ####################################
# Predict Market Size Using a Linear Model
f9x <- lm(logunits~iri_key+week,data=SS)
# f8x <- gbm.fixed(data=as.data.frame(cbind(x_mat,SS$logunits)),1:dim(x)[2],dim(x)[2]+1,n.trees=500,
#                 family="gaussian",keep.data=TRUE)
SS$units_hat2 <- round(exp(f9x$fit))

# Aggregate Weekly Store Total Predicted Units and Merge Total Units (Market Size) into Data
total2 <- aggregate(units_hat2~iri_key+week,data=SS,sum,na.rm=T)
names(total2)[3] <- "total_units2"
SS <- merge(SS,total2,by=c("iri_key","week"))

# Calculate Predicted Market Share (A Market is a store during a week.)
SS$share2 <- SS$units_hat2/SS$total_units2

f9 <- glm(share_formula2,data=SS[SS$j==1,],family=binomial)

# Predict in Validation Set
p9share <- predict(f9,newdata=SS2,type="response")
p9 <- log(p9share*SS$total_units[SS$j==2])

# Predict in Test Set
p9share.t <- predict(f9,newdata=SS0,type="response")
p9t <- log(p9share.t*SS$total_units[SS$j==0])

# 1.10 Combine Models ##########################################################
# Regress Actual Values on Eight Predict Series
# Constraints: Coefficients are positive, sum to 1 and no intercept
X <- t(as.matrix(rbind(p1,p2,p3,p4,p5,p6,p7_new,p8,p9)))
Y <- valy
Rinv <- solve(chol(t(X) %*% X))
A <- cbind(rep(1,dim(X)[2]),diag(dim(X)[2]))
B <- c(1,rep(0,dim(X)[2]))
D <- t(Y) %*% X
# Calculate the coefficients, aka weights of models
sol <- solve.QP(Dmat=Rinv,dvec=D,Amat=A,bvec=B,meq=1,factorized=TRUE)
f_combine <- sol$solution
p_combine <- f_combine %*% t(X)

# Apply the Weights in Test Set and Create a New Prediction
X.t <- t(as.matrix(rbind(p1t,p2t,p3t,p4t,p5t,p6t,p7t_new,p8t,p9t)))
p_combine_t <- f_combine %*% t(X.t)
Y.t <- testy

# Calculate all the root mean square errors (validation/test)
rmse_all <-  mapply(rmse,list(p1,p2,p3,p4,p5,p6,p7_new,p8,p9,p_combine),MoreArgs=list(Y)) 
rmse_all.t <-  mapply(rmse,list(p1t,p2t,p3t,p4t,p5t,p6t,p7t_new,p8t,p9t,p_combine_t),MoreArgs=list(Y.t)) 

# 2. Calculating Standard Errors #############################################
# 
# Hold Variable Selection Fixed and Subsample to make the Standard Errors Consistent
#
#
################################################################################

# 2.1 define subsample (withouth replacement) function, holding variable selection fixed
subsampleFun <- function(f=obj,model = "lars", type = "lasso",x.data,y.data,n) {

  smpl <- sample(seq(n),replace=FALSE)
  smpl.x <- x.data[smpl,]
  smpl.y <- y.data[smpl]

  testx <- smpl.x[SS$j==0,]
  valx <- smpl.x[SS$j==2,]
  x <- smpl.x[SS$j==1,]
  testy <- smpl.y[SS$j==0]
  valy <- smpl.y[SS$j==2]
  y <- smpl.y[SS$j==1]
  
  # Model 1
  if (model == "lm") {
    coef <- f$coefficients
    coef[is.na(coef)] <- 0 
    
    pred <- coef %*% t(x)
    pred.v <- coef %*% t(valx)
    pred.t <- coef %*% t(testx)
  }
  
  # Model 2-4
  if (model == "lars") {
    if(missing(type)) stop("Specify type for Lars function")
    else {
      if (type == "lasso") {
        mode = "lambda" 
      } else {
        mode = "step"
      }
      
      step <- length(f$actions)
      pred <- predict(f,x,s=step,type="coef")$fit
      pred.v <-predict(f,s=step,valx,type="fit",mode=mode)$fit
      pred.t <-predict(f,s=step,testx,type="fit",mode=mode)$fit
    }
  }
      
  # Model 5
  if (model == "randomForest" ) {
    pred <- f$predicted
    pred.v <- predict(f,valx)
    pred.t <- predict(f,testx)
  }
  
  # Model 6
  if (model == "svm" ) {
    x2<-x[,-grep("Intercept",names(as.data.frame(x)))]
    valx2<-valx[,-grep("Intercept",names(as.data.frame(valx)))]
    testx2<-testx[,-grep("Intercept",names(as.data.frame(testx)))]
    
    pred <- predict(f,x2)
    pred.v <- predict(f,valx2)
    pred.t <- predict(f,testx2)
  }
  
  # Model 7
  if (model == "boosting") {
    pred <- predict(f,n.trees=500,type="response")
    pred.v <- predict(f,newdata=as.data.frame(cbind(valx,valy)),n.trees=500,type="response")
    pred.t <- predict(f,newdata=as.data.frame(cbind(testx,testy)),n.trees=500,type="response")
  }

  # Model 8-9
  if (model == "logit") {
    share.v <- predict(f, newdata=valx,type="response")
    pred.v <- log(share.v*SS$total_units[SS$j==2])

    share.t <- predict(f, newdata=testx,type="response")
    pred.t <- log(share.t*SS$total_units[SS$j==0])
  }
  
  predList <- list("train"=pred,"validate"=pred.v,"test"=pred.t)
  return(predList)
}

# 2.2 Calculating standard errors ##############################################
# define subsample parameters
n.bs = 1000

# define matrices that stores the predicted values for subsample 
train_mat_list <- list()
validate_mat_list <- list()
test_mat_list <- list()

# define function list and model names
fun_list <- list(f1,f2,f3,f4,f5,f6,f7,f8,f9)
model_list <- c("lm","lars","lars","lars","randomForest","svm","boosting","logit","logit")
num_fun <- length(fun_list)
type_list <- c("step","forward.stagewise","lasso")

num_fun =9

# subsample the Main models
for (i.fun in seq(1,num_fun)) {
  
    tic()
    error <- replicate( n.bs, subsampleFun(f=fun_list[[i.fun]],model=model_list[i.fun],type=type_list[i.fun%%3+1],x.data=x_mat,y.data=SS$logunits,n=dim(x_mat)[1]/2) )
    toc()
    
    train_mat_list[[i.fun]] <-  error["train",]
    validate_mat_list[[i.fun]] <- error["validate",]
    test_mat_list[[i.fun]] <- error["test",]

}

# calculate the standard deviation of rmse for each model (validation/test)
m = 9
sd_val = numeric()
sd_test = numeric()

for ( i in seq(1,m)) {
  output.v <- matrix(unlist(validate_mat_list[[i]]), ncol = n.bs, byrow = TRUE)
  output.t <- matrix(unlist(test_mat_list[[i]]), ncol = n.bs, byrow = TRUE)
  
  rmse.v <-mapply(rmse,as.list(as.data.frame(output.v)),MoreArgs=list(valy))
  sd.v <- sd(rmse.v)
  
  rmse.t <-mapply(rmse,as.list(as.data.frame(output.t)),MoreArgs=list(testy))
  sd.t <- sd(rmse.t)
  
  sd_val[i] = sd.v
  sd_test[i] = sd.t
  
}

