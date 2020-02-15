library(datasets) 
head(iris)
#?plot # will provide you with help if you put a ?

plot(iris$Species) #categorical variable
plort(iris$Petal.Length) #quantitative variable
plot(iris$Petal.Width)
plot(iris$Species, iris$Petal.Width) # Cat x quant # automatically will do a box plot
plot(iris$Petal.Length, iris$Petal.width) # Quant pair
plot(iris) # entire matrix of scatter plots

#plot with options
plot(iris$Petal.Length, iris$Petal.width,
     col = "#cc0000", # Hex code for datalab.cc red
     pch = 19, # point character # solid circle
     main = "Iris: Petal Length vs Petal Width",
     xLab = "Petal Length", 
     yLab = "Petal Width")


plot(cos, 0, 2*pi)
plot(exp, 1, 5)
plot(dnorm, -3, +3)

# Formula plot with options
plot(dnorm, -3, +3,
     col = "#cc0000",
     lwd = 5,
     main = "Standard Normal Distribution",
     xlab = "z-scores",
     ylab = "Density")



##################### clean up ###########################

# Clear packages
detach("package:datasets", unload = TRUE)
