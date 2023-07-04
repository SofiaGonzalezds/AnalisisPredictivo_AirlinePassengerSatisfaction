library(tidyverse)
library(janitor)
library(skimr)
library(GGally)
library(ggpubr)
library(ggcorrplot)
library(sjstats)
library(ggcorrplot)

#Importo el Dataset
dataset=read.csv("Airline_Passenger_Satisfaction.csv")
view(dataset)
describe(dataset)
summary(dataset)
#Transformo los puntajes en variables categóricas
dataset= dataset %>% mutate(Inflight.wifi.service= as.character(Inflight.wifi.service),Departure.Arrival.time.convenient= as.character(Departure.Arrival.time.convenient),Gate.location= as.character(Gate.location),Food.and.drink= as.character(Food.and.drink),Online.boarding= as.character(Online.boarding),
                            Seat.comfort= as.character(Seat.comfort),Inflight.entertainment= as.character(Inflight.entertainment), On.board.service= as.character(On.board.service),Leg.room.service= as.character(Leg.room.service),Baggage.handling= as.character(Baggage.handling), Checkin.service= as.character(Checkin.service),
                            Inflight.service= as.character(Inflight.service),Cleanliness= as.character(Cleanliness), Ease.of.Online.booking= as.character(Ease.of.Online.booking))
#funciones utilizadas

es.outlier=function(x){
  sup = quantile(x,0.75,na.rm=T)+IQR(x,na.rm=T)*1.5
  inf = quantile(x,0.25,na.rm=T)-IQR(x,na.rm=T)*1.5
  es=!between(x,inf,sup)
  return(es)
}
cant_outliers= function(x){
  cont=0
  for (i in x) {
    if(i==TRUE){
      cont=cont+1
    }
  }
  return(cont)
}




#Limpieza de la base

#Missings

skim(dataset)

#Outliers

outlier_Age= cant_outliers(es.outlier(dataset$Age))
outlier_Age
#2291 outliers en la variable Flight Distance
outlier_Distance= cant_outliers(es.outlier(dataset$Flight.Distance))
outlier_Distance
vecoutlier_Dis=es.outlier(dataset$Flight.Distance)
vecoutlier_Dis
regis_outliers_Distance=subset(dataset, vecoutlier_Dis)
view(regis_outliers_Distance %>% arrange(desc(regis_outliers_Distance$Flight.Distance)))
boxplot(dataset$Flight.Distance)
#14526 outliers en la variable Departure Delay
outlier_DepartureDelay= cant_outliers(es.outlier(dataset$Departure.Delay.in.Minutes))
outlier_DepartureDelay
vecoutlier_DepDel=es.outlier(dataset$Departure.Delay.in.Minutes)
vecoutlier_DepDel
regis_outliers_DepartureDelay=subset(dataset, vecoutlier_DepDel)
view(regis_outliers_DepartureDelay %>% arrange(desc(regis_outliers_DepartureDelay$Departure.Delay.in.Minutes)))

#Quito los missings solamente para el análisis de outliers
dataset2=dataset[complete.cases(dataset$Arrival.Delay.in.Minutes),]
#13954 outliers en la variable Arrival Delay
outlier_ArrivalDelay= cant_outliers(es.outlier(dataset2$Arrival.Delay.in.Minutes))
outlier_ArrivalDelay
vecoutlier_ArrDel=es.outlier(dataset$Arrival.Delay.in.Minutes)
vecoutlier_ArrDel
regis_outliers_ArrivalDelay=subset(dataset, vecoutlier_ArrDel)
view(regis_outliers_ArrivalDelay %>% arrange(desc(regis_outliers_ArrivalDelay$Arrival.Delay.in.Minutes)))

#Analizo frecuencias de las variables categóricas

columnas_categoricas= subset(dataset, select = which(sapply(dataset, is.character)))
length(colnames(columnas_categoricas))
frecuencias=lapply(columnas_categoricas, table)
lapply(columnas_categoricas, function(x) barplot(table(x), main = paste("Frecuencia de", names(columnas_categoricas)[names(columnas_categoricas) == names(x)], sep = " "), xlab = names(columnas_categoricas)[names(columnas_categoricas) == names(x)], ylab = "Frecuencia"))

#Analizo la correlación de variables que creo que podrían tener relación

#Relación entre el genero y el espacio en las piernas

tabla1= table(dataset$Gender, dataset$Leg.room.service)
tabla1
correlacion1=as.matrix(tabla1) %>% rstatix::cramer_v()
correlacion1

#Relación entre los inconvenientes en el tiempo de llegada/salida y el tipo de viaje
tabla2= table(dataset$Type.of.Travel, dataset$Departure.Arrival.time.convenient)
tablaprop2= prop.table(tabla2)*100
tablaprop2
correlacion2=as.matrix(tabla2) %>% rstatix::cramer_v()
correlacion2

#Relación entre entretenimiento y edad

anova_res = aov( dataset$Age ~ dataset$Inflight.entertainment, dataset)
summary(anova_res)
anova_tab = anova_res %>% sjstats::anova_stats() 
anova_tab$omegasq[1]
ggplot(dataset) + 
  geom_boxplot(aes(x=dataset$Inflight.entertainment, y=dataset$Age))

res_kruskal = rstatix::kruskal_test(dataset, dataset$Age ~ dataset$Inflight.entertainment)
epsilon_sq = res_kruskal$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq

#Relación entre dificultad para sacar el pasaje por booking y edad

anova_res2 = aov( dataset$Age ~ dataset$Ease.of.Online.booking, dataset)
summary(anova_res2)
anova_tab2 = anova_res2 %>% sjstats::anova_stats() 
anova_tab2$omegasq[1]
ggplot(dataset) + 
  geom_boxplot(aes(x=dataset$Ease.of.Online.booking, y=dataset$Age))

res_kruskal2 = rstatix::kruskal_test(dataset, dataset$Age ~ dataset$Ease.of.Online.booking)
epsilon_sq2 = res_kruskal2$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq2


#Relación entre el tipo de viaje y la edad

anova_res3 = aov( dataset$Age ~ dataset$Type.of.Travel, dataset)
summary(anova_res3)
anova_tab3 = anova_res3 %>% sjstats::anova_stats() 
anova_tab3$omegasq[1]
ggplot(dataset) + 
  geom_boxplot(aes(x=dataset$Type.of.Travel, y=dataset$Age))

res_kruskal3 = rstatix::kruskal_test(dataset, dataset$Age ~ dataset$Type.of.Travel)
epsilon_sq3 = res_kruskal3$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq3

#Relacion entre la clase y la edad

anova_res4 = aov( dataset$Age ~ dataset$Ease.of.Online.booking, dataset)
summary(anova_res4)
anova_tab4 = anova_res4 %>% sjstats::anova_stats() 
anova_tab4$omegasq[1]
ggplot(dataset) + 
  geom_boxplot(aes(x=dataset$Class, y=dataset$Age))

res_kruskal4 = rstatix::kruskal_test(dataset, dataset$Age ~ dataset$Ease.of.Online.booking)
epsilon_sq4 = res_kruskal4$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq4


#Relación entre la clase y el tipo de viaje

tabla3= table(dataset$Type.of.Travel, dataset$Class)
tablaprop3= prop.table(tabla3)*100
tablaprop3
correlacion3=as.matrix(tabla3) %>% rstatix::cramer_v()
correlacion3


#Relación entre la clase y distancia de vuelo

anova_res5 = aov( dataset$Flight.Distance ~ dataset$Class, dataset)
summary(anova_res5)
anova_tab5 = anova_res5 %>% sjstats::anova_stats() 
anova_tab5$omegasq[1]
ggplot(dataset, aes(x = Class, y = Flight.Distance, fill = Class)) +
  geom_boxplot(alpha = 0.8) +
  labs(x = "Class", y = "Flight Distance", fill = "Class") +
  ggtitle("Relación entre Class y Flight Distance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


res_kruskal5 = rstatix::kruskal_test(dataset, dataset$Flight.Distance ~ dataset$Class)
epsilon_sq5 = res_kruskal5$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq5


#Relación entre la cantidad de atraso y la satisfaccion

anova_res6 = aov( dataset$Departure.Delay.in.Minutes ~ dataset$satisfaction, dataset)
summary(anova_res6)
anova_tab6 = anova_res6 %>% sjstats::anova_stats() 
anova_tab6$omegasq[1]
ggplot(dataset) + 
  geom_boxplot(aes(x= dataset$satisfaction, y=dataset$Departure.Delay.in.Minutes))

res_kruskal6 = rstatix::kruskal_test(dataset, dataset$Departure.Delay.in.Minutes ~ dataset$satisfaction)
epsilon_sq6 = res_kruskal6$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq6

#Almaceno las columnas numéricas en un data frame

columnas_numericas=subset(dataset2, select = which(sapply(dataset2, is.numeric)))
length(colnames(columnas_numericas))

#Relación entre la Delay y la ubicación de la puerta

anova_res7 = aov( dataset$Departure.Delay.in.Minutes ~ dataset$Gate.location, dataset)
summary(anova_res7)
anova_tab7 = anova_res7 %>% sjstats::anova_stats() 
anova_tab7$omegasq[1]
ggplot(dataset, aes(x = dataset$Gate.location, y = dataset$Departure.Delay.in.Minutes, fill = Class)) +
  geom_boxplot(alpha = 0.8) +
  labs(x = "Gate Location", y = "Departure Delay in Minutes", fill = "Class") +
  ggtitle("Relación entre Departure Delay y Gate Location") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


res_kruskal7 = rstatix::kruskal_test(dataset, dataset$Departure.Delay.in.Minutes ~ dataset$Gate.location)
epsilon_sq7 = res_kruskal7$statistic / 
  ((res_kruskal$n**2 - 1) / (res_kruskal$n + 1))
epsilon_sq7



#RELACIONES INTERESANTES (LO INCLUIDO EN EL PPT)

#Variables Categóricas

#Relación entre la satisfacción y el online boarding

tabla4= table(dataset$satisfaction, dataset$Online.boarding)
tablaprop4= prop.table(tabla4)*100
tablaprop4
correlacion4=as.matrix(tabla4) %>% rstatix::cramer_v()
correlacion4
chisq.test(tabla4)

#Relación entre la satisfaccion y la clase

tabla5= table(dataset$satisfaction, dataset$Class)
tablaprop5= prop.table(tabla5)*100
tablaprop5
correlacion5=as.matrix(tabla5) %>% rstatix::cramer_v()
correlacion5

#Relación entre la clase y el tipo de viaje

tabla3= table(dataset$Type.of.Travel, dataset$Class)
tablaprop3= prop.table(tabla3)*100
tablaprop3
correlacion3=as.matrix(tabla3) %>% rstatix::cramer_v()
correlacion3

#Relacion entre el online bookings y el online boarding

tabla6= table(dataset$Online.boarding, dataset$Ease.of.Online.booking)
tablaprop6= prop.table(tabla6)*100
tablaprop6
correlacion6=as.matrix(tabla6) %>% rstatix::cramer_v()
correlacion6




#Relación entre numéricas

#Matriz de correlación 

cor_matrix <- cor(columnas_numericas)
# Crear la plot con ggcorrplot
ggcorrplot(cor_matrix, 
           hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           ggtheme = ggplot2::theme_gray,
           colors = c("#45B39D" ,"#FFFFFF","#1ABC9C"), 
           title = "Matriz de correlación")

#Relación entre Departure Delay in Minutes y Arrival Delay in Minutes

correlacion= cor(dataset$Arrival.Delay.in.Minutes,use="pairwise", dataset$Departure.Delay.in.Minutes, method="pearson")
correlacion

#Relación entre Numéricas y categóricas

#Relación entre la clase y distancia de vuelo

anova_res5 = aov( dataset$Flight.Distance ~ dataset$Class, dataset)
summary(anova_res5)
anova_tab5 = anova_res5 %>% sjstats::anova_stats() 
anova_tab5$omegasq[1]
ggplot(dataset, aes(x = Class, y = Flight.Distance, fill = Class)) +
  geom_boxplot(alpha = 0.8) +
  labs(x = "Class", y = "Flight Distance", fill = "Class") +
  ggtitle("Relación entre Class y Flight Distance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(dataset, aes(x = Class, y = Flight.Distance, fill = Class)) +
  geom_point(size = 3, shape = 21, color = "black") +
  labs(title = "Class y Flight Distance",
       x = "Class",
       y = "Flight Distance",
       fill = "Class") +
  theme_minimal()

res_kruskal5 = rstatix::kruskal_test(dataset, dataset$Flight.Distance ~ dataset$Class)
epsilon_sq5 = res_kruskal5$statistic / 
  ((res_kruskal5$n**2 - 1) / (res_kruskal5$n + 1))
epsilon_sq5


