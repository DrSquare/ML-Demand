
# Remove everything from the environment and clear out memories
rm(list = ls())
gc()

# Statistic Packages
require(foreign)
require(stats)
require(lattice)
require(Matrix)
require(HH) #vif
require(quadprog) #Qudratic Programming: Constrained Regression
require(pastecs) #Summary Statistics
# Machine Learning Pacakges
require(lars)
require(e1071) #SVM
require(randomForest)
require(gbm) 
require(dismo) #Gradient Boosting Model
require(ipred) #Bagging

# Useful Functions
# Function 1: RMSE
rmse <- function(obs, pred) sqrt(mean((obs-pred)^2))

# Function 2: remove outliers
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- qnt[1]
  y[x > (qnt[2] + H)] <- qnt[2]
  y
}

# Function 3: Try and Error Lars
lars_type <- function(t="step",x,y,step=100) {
  for (i in seq(1,30)) {
    f <- tryCatch(lars(x,y,type=t,eps=0.0001,trace=F,max.steps=step),
                  warning=function(w) TRUE, error=function(e) TRUE)
    # if the lars function does not converge in # of steps, move on to the next iteration
    if(is.logical(f)){
      print(paste(t,"not working",i))
      next
    }
    # if the lars function converges, return the lars object
    if(!is.logical(f)){
      print(paste(t,"is working",i))
      return(f) 
      break
    }
  }
}

# Function 4: Regress Predicted Error on Constant
regconst <- function(x,y){
  z <- x - y
  mean(lm(z~1)$coef)
}

# Function 5: Create Positive Definite Matrix
posdef <- function (n, ev = runif(n, 0, 10)) {
  Z <- matrix(ncol=n, rnorm(n^2))
  decomp <- qr(Z)
  Q <- qr.Q(decomp) 
  R <- qr.R(decomp)
  d <- diag(R)
  ph <- d / abs(d)
  O <- Q %*% diag(ph)
  Z <- t(O) %*% diag(ev) %*% O
  return(Z)
}

# Function 5: tic/toc function to time the execution
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self"))
{
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function()
{
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}