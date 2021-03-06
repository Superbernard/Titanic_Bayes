## A case study on survivial data of passengers on Titanic
#
# The aim of this piese of script is to analyze how each variable ( passenger class, gender etc. ) affects the survival rate 
# of passengers onboard in the case of Titanic. Bayesian and classic logistic regression models as well as artifical neural network
# are established to fit the training set. The performance of these models are then compared on a test set.
#
# In the  Bayesian model, Markov Chain Monte Carlo (MCMC) is used to approximate the posterior distribution as calculated by 
# Bayes' Theorem.
#



data<-read.csv("Titanic.csv",header= T, sep = ",")   #load dataset 
data<-data[complete.cases(data),]   #get rid of the rows with missing values

Pclass.f = factor(data[,3])    #define a factor variable to hold the dummy variable for Pclass        
dummies_Pclass = model.matrix(~Pclass.f)   #create dummy variable for Pclass (passenger class) 
dummies_Pclass = dummies_Pclass[,-1]   #drop the first column (intercept)

Sex.f = factor(data[,5])    #define a factor variable to hold the dummy variable for passenger gender             
dummies_Sex = model.matrix(~Sex.f)    #create dummy variable for Sex, male 1; female 0
dummies_Sex = dummies_Sex[,-1]   #drop the first column (intercept)



##split the dataset into two subsets (training and testing)
n_obser=dim(data)[1]   #number of observations 

number_train<-round(n_obser*0.8)    #80% of the data set to training set (an arbitrary split)
number_test<-n_obser - number_train   #the rest 20% of the data set to test set

#since the observations in the data set are already in a random order, we can simply pick the first 80% 
#as our training set and the last 20% as our test set. Otherwise, we need to randomly pick 80% and 20% 
#observations from the whole set to form our training and test set.

index_train<-sample(number_train,number_train,replace = F)    #shuffle observations in the training set again 
train<-data[index_train,]    #training set

x<-cbind(dummies_Pclass[index_train,], dummies_Sex[index_train] ,train[,c(6,7,8)])  #select appropriate variables as predictors
y<-train[,2]  #Survival as the binary response

test<-data[-index_train,]    #test set

x_test<-cbind(dummies_Pclass[-index_train,],dummies_Sex[-index_train] ,data[-index_train,c(6,7,8)])  
#select appropriate variables as predictors
y_test<-test[,2]  #Survival as the binary response

data<-rbind(train,test)[,c(3,5,6,7,8,2)]  #full data set used in the analysis
data$Pclass<-as.factor(data$Pclass)    #convert Pclass to factor variable
data$Survived<-as.factor(data$Survived)   #convert Survived to factor
head(data)   #check the structure of data
summary(data)   #summary of data


######Examine marginal effects##########
library(ggplot2)    #load package ggplot2

p1<-ggplot(data, aes(x = data$Pclass, fill = factor(data$Survived))) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'PClass' ,y="Count", title="Pclass vs Survival",fill="Survived")    #effect of Pclass on Survival
plot(p1)

gender_vs_survival = table(data$Sex,data$Survived)       #count of survived individuals on gender
gender_vs_survival =as.data.frame.matrix(gender_vs_survival)     #convert to data frame
gender_vs_survival$Survival_rate <- gender_vs_survival[,2] / (gender_vs_survival[,1]+gender_vs_survival[,2])    #survival rate
gender_vs_survival

p2<-ggplot(data, aes(x = data$Survived, y = data$Age, col=data$Survived))+
  geom_point()+
  labs(title="Age vs Survival",x="Survived",y = "Age",colour = "Survivrd")+
  geom_boxplot(fill=NA, outlier.colour=NA)       #effect of gender on Survival
plot(p2)

sibtab = table(data$SibSp, data$Survived)      #count of survived individuals on number of sibling and spouse
sibtab <- as.data.frame.matrix(sibtab)     #convert to data frame
sibtab$Survival_rate <- sibtab[,2] / (sibtab[,1]+sibtab[,2])     #survival rate
Sibsp_vs_survival=sibtab
Sibsp_vs_survival

parchtab = table(data$Parch, data$Survived)       #count of survived individuals on number of parents and children
parchtab <- as.data.frame.matrix(parchtab)     #convert to data frame
parchtab$Survival_rate <- parchtab[,2] / (parchtab[,1]+parchtab[,2])     #survival rate
parchtab



#################Bayesian logistc model#####################
library(R2jags)    #load package R2jags to interact with JAGS in R

# Specify the data in a list, for later shipment to JAGS:
n = length(y)
Pclass.f2 = x$Pclass.f2
Pclass.f3 = x$Pclass.f3
Sex = x$dummies_Sex
Age = x$Age
Sibsp = x$SibSp
Parch = x$Parch
Survived=y

dataList = list(
  n=n, Pclass.f2=Pclass.f2, Pclass.f3=Pclass.f3,
  Sex=Sex, Age=Age, Sibsp=Sibsp, Parch=Parch,Survived=Survived
)



## DEFINE THE MODEL.
modelString = "
model{

# LOGIT MODEL
for (i in 1:n){ 
Survived[i] ~ dbern( pi[i] )
logit( pi[i] ) <- beta0 + beta1 * Pclass.f2[i] + beta2 * Pclass.f3[i] +
beta3 * Sex[i] + beta4 * Age[i] + beta5 * Sibsp[i] + beta6 * Parch[i]
}

# priors 
beta0~dnorm( 0, 0.00001)     
beta1~dnorm( 0, 0.00001)
beta2~dnorm( 0, 0.00001)
beta3~dnorm( 0, 0.00001)
beta4~dnorm( 0, 0.00001)
beta5~dnorm( 0, 0.00001)
beta6~dnorm( 0, 0.00001)

}
" # close quote for modelString

# Write modelString to a text file
writeLines( modelString , con="titanic.txt" )


initsList =list( beta0=0, beta1=0.01, beta2=0.01, beta3=0.01,
                 beta4=0.01, beta5=0.01, beta6=0.01)    #initial values


jagsModel = jags.model( file="titanic.txt" , data=dataList , inits=initsList , 
                        n.chains=3 , n.adapt=1000 )   #create the jags model object

update( jagsModel , n.iter=1000 )
codaSamples = coda.samples( jagsModel , 
                            variable.names=c("beta0","beta1","beta2","beta3","beta4","beta5","beta6") ,
                            n.iter=10000 )   #generate posterior samples via MCMC

save( codaSamples , file=paste0("titanic","Mcmc.Rdata") )    


#Diagnose MCMC

windows()
plot(codaSamples[,1:4])

windows()
plot(codaSamples[,5:7])

windows()
gelman.plot(codaSamples)


# Posterior descriptives:
result=summary(codaSamples)
result

##point estimates
beta0 = (result$statistics)[1,1]              
beta1 = (result$statistics)[2,1]                            
beta2 = (result$statistics)[3,1]            
beta3 = (result$statistics)[4,1]
beta4 = (result$statistics)[5,1]             
beta5 = (result$statistics)[6,1]
beta6 = (result$statistics)[7,1]              

beta=matrix(c(beta0,beta1,beta2,beta3,beta4,beta5,beta6),1,7) #put them into a column vector

pctg_change_odds<-exp(beta[,-1])-1   #exclude the beta0 for intercept

pctg_change_odds   #check result


library(ggplot2)
library(grid)

vplayout <- function(x, y) viewport(layout.pos.row = x, layout.pos.col = y) 
#function for arranging the plots layout 

denplot<-function(postsamples){  #function to make histogram and density plots for posterior 
  p<-list(7)
  dat<-as.data.frame(as.matrix(postsamples,chains=TRUE))   #convert to data frame
  dat<-dat[,-1]   #exclude the variable chain (first column)
  title<-paste("beta",0:6, sep = " ")    #set titles for plots
  xl<-paste("beta",0:6, sep = " ") 
  
  windows()
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(3, 3)))
  layout_r<-rep(1:3,each=3)   #row index for plots layout
  layout_c<-rep(1:3,3)        #column index for plots layout
  
  for (i in 1:7){
    # Histogram overlaid with kernel density curve
    p[[i]] <- ggplot() + aes(dat[,i], ..density..) + geom_histogram(color="Pink",fill="Red",bins=40)+
      geom_density(colour="black")+
      xlab(xl[i]) +
      ylab("Density") +
      ggtitle(title[i])
    
    
    print(p[[i]], vp = vplayout(layout_r[i], layout_c[i]))
    
  }
  
}

denplot(codaSamples)   #call denplot function

x_test_des<-cbind(rep(1,number_test),x_test)

log_odds = function(x_design) beta%*%t(x_design)   #formula of log odds
p= function(x_design) exp(log_odds(x_design))/(1+exp(log_odds(x_design)))   #function to calculate p



##Make graphs to test model performance on test data

graphics.off()
windows()
layout(matrix(1:2,1,2))
pred_bay= p(x_test_des)  #Predictions on test data via the Bayesian model
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Bayesian model performance test\n before applying decision boundary")
points(1:length(pred_bay),pred_bay,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from Bayesian"),pch = c(16,16))
pred_bay= ifelse(pred_bay > 0.5,1,0)   #Put a decision boundary at 0.5
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Bayesian model performance test\n after applying decision boundary")
points(1:length(pred_bay),pred_bay,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from Bayesian"),pch = c(16,16))



misClasificError1 <- mean(pred_bay != y_test)
print(paste('Accuracy',1-misClasificError1))    #check accuracy 




#################Classic model#####################
model <- glm(Survived ~.,family=binomial(link='logit'),data=train[,c(2,3,5,6,7,8)])

summary(model)


##Make graphs to test model performance on test data
graphics.off()
windows()
layout(matrix(1:2,1,2))
fitted.results <- predict(model,newdata=test,type='response') #Predictions on test data via the classical model
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Classical model performance test\n before applying decision boundary")
points(1:length(fitted.results),fitted.results,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from Classical"),pch = c(16,16))
fitted.results <- ifelse(fitted.results > 0.5,1,0) #Same decision boundary at 0.5
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Classical model performance test\n after applying decision boundary")
points(1:length(fitted.results),fitted.results,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from Classical"),pch = c(16,16))


misClasificError2 <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError2))    #check accuracy 




##############Artificial Neural Network############################
library(neuralnet)
train_net_data<-cbind(x,y)    #prepare training data
colnames(train_net_data)[3]<-"Sex.dummy"     #specify column names
colnames(train_net_data)[7]<-"Survived"
net.titanic <- neuralnet(Survived~Pclass.f2+Pclass.f3+Sex.dummy+Age+SibSp+Parch, 
                         data=train_net_data,hidden=10, threshold=0.01)    #build neural network
print(net.titanic)

plot(net.titanic,rep = "best") #Plot the neural network

test_net_data<-cbind(x_test)   #prepare test data
colnames(test_net_data)[3]<-"Sex.dummy"    #specify column names
net.results <- compute(net.titanic, test_net_data)    #run test data on the trained net

ls(net.results)

##Make graphs to test model performance on test data
graphics.off()
windows()
layout(matrix(1:2,1,2))
net.fitted.results<-net.results$net.result #predictions on test data via trained net
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Net performance test\n before applying decision boundary")
points(1:length(net.fitted.results),net.fitted.results,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from net"),pch = c(16,16))
net.fitted.results <- ifelse(net.fitted.results > 0.5,1,0)  #Same decision boundary at 0.5
plot(y_test,pch=16,col=rgb(1,0,0,0.75),xlab="Index",ylab = "Survived", ylim=c(-0.2,1.2),
     main="Net performance test\n after applying decision boundary")
points(1:length(net.fitted.results),net.fitted.results,pch=16,col=rgb(0,0,1,0.75))
legend("topright",col=c(rgb(1,0,0,0.75),rgb(0,0,1,0.75)),legend =c("Target","Output from net"),pch = c(16,16))

misClasificError3 <- mean(net.fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError3))    #check accuracy 


misClasificError1    #check error from Bayesian model

misClasificError2    #check error from classic model

misClasificError3    #check error from nerual network model
