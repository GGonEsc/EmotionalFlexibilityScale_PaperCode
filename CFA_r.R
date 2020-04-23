
library(lavaan)
library(semPlot)

#import dataset:
# Firstly opent the excel file and then in the first sheet "ItemData" select all rown in columnn N to AC,
# then, press copy, and then run the following command:
FREE16items <- read.table(file = "clipboard", sep = "\t", header=TRUE)

# View(FREE16items)

#create the models
# 1st order (computes the correlations between variables)
first.model = '
Exp =~ FREE_1+FREE_2+FREE_3+FREE_4+FREE_5+FREE_6+FREE_7+FREE_8
Sup =~ FREE_9+FREE_10+FREE_11+FREE_12+FREE_13+FREE_14+FREE_15+FREE_16
'

# 2nd order (Hierachical model: scales using the latent variables on the first order half )
secOrder.model = '
PosExp =~ FREE_1+FREE_2+FREE_3+FREE_4
NegExp =~ FREE_5+FREE_6+FREE_7+FREE_8
PosSup =~ FREE_9+FREE_10+FREE_11+FREE_12
NegSup =~ FREE_13+FREE_14+FREE_15+FREE_16
Exp =~ PosExp+NegExp
Sup =~ PosSup+NegSup
'


#run the models (test for domain specific factors accounted/controled by a global[generalized] which takes out the correlations)
first.fit = cfa(first.model, data=FREE16items)
secOrder.fit = cfa(secOrder.model, data=FREE16items)
#std.lv=TRUE  # help if the model fit outputs that the model is not positive definite (i.e. the variances are negative)


#fit indices
fitMeasures(first.fit)
fitMeasures(secOrder.fit)

#summaries
summary(first.fit, standardized=TRUE, rsquare=TRUE)
summary(secOrder.fit, standardized=TRUE, rsquare=TRUE)


# Now we can plot the hierachical model:
semPaths(secOrder.fit,
         what = "std", # this argument controls what the color of edges represent. In this case, standardized parameters
         whatLabels = "std", # "est" This argument controls what the edge labels represent. In this case, standardized parameter estimates
         style = "lisrel", # This will plot residuals as arrows, closer to what we use in class
         residScale = 8, # This makes the residuals larger
         theme = "colorblind", # qgraph colorblind friendly theme
         nCharNodes = 0, # Setting this to 0 disables abbreviation of nodes
         reorder = FALSE, # This disables the default reordering
         rotation = 2, # Rotates the plot, 1=standard, 2= left-right, 3=up-side-down
         layout = "tree2", # tree layout options are "tree", "tree2", and "tree3" (reduced the distance)
         cardinal = "lat cov", # This makes the latent covariances connet at a cardinal center point
         curvePivot = TRUE, # Changes curve into rounded straight lines
         sizeMan = 4, # Size of manifest variables
         sizeLat = 10, # Size of latent variables
         mar = c(2,5,2,5.5)#, # Figure margins
)



### Compute Chrombach's alpha for the rest of the scales (report raw)
library(psych)
alpha(FREE16items, check.keys=TRUE)#items that correlate negatively with the overall scale will be reverse coded
alpha(FREE16items[c("FREE_1","FREE_2","FREE_3","FREE_4")], check.keys=TRUE)
alpha(FREE16items[c("FREE_5","FREE_6","FREE_7","FREE_8")], check.keys=TRUE)
alpha(FREE16items[c("FREE_9","FREE_10","FREE_11","FREE_12")], check.keys=TRUE)
alpha(FREE16items[c("FREE_13","FREE_14","FREE_15","FREE_16")], check.keys=TRUE)
alpha(FREE16items[c("FREE_1","FREE_2","FREE_3","FREE_4","FREE_5","FREE_6","FREE_7","FREE_8")], check.keys=TRUE)
alpha(FREE16items[c("FREE_9","FREE_10","FREE_11","FREE_12","FREE_13","FREE_14","FREE_15","FREE_16")], check.keys=TRUE)
# These were confirmed in in MATLAB. Then, I used Matlab for the rest of the questionnaires.




