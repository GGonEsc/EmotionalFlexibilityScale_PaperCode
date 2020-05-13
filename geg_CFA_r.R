
library(lavaan)
library(semPlot)

#import dataset:
# Firstly opent the excel file and then in the first sheet "ItemData" select all rown in columnn N to AC,
# then, press copy, and then run the following command:
FREE16items <- read.table(file = "clipboard", sep = "\t", header=TRUE)

# View(FREE16items)

# # create the models:
# single factor model 
oneFactor.model = ' Flex =~ FREE_1+FREE_2+FREE_3+FREE_4+FREE_5+FREE_6+FREE_7+FREE_8 + FREE_9+FREE_10+FREE_11+FREE_12+FREE_13+FREE_14+FREE_15+FREE_16 '

# 1st order dual factor = enhance & supress abilities (computes the covariance between variables)
firstDual.model = ' Exp =~ FREE_1+FREE_2+FREE_3+FREE_4+FREE_5+FREE_6+FREE_7+FREE_8
                    Sup =~ FREE_9+FREE_10+FREE_11+FREE_12+FREE_13+FREE_14+FREE_15+FREE_16 '

# 1st order dual factor = positive & negative valencies (computes the covariance between variables)
secDual.model = ' Pos =~ FREE_1+FREE_2+FREE_3+FREE_4+ FREE_9+FREE_10+FREE_11+FREE_12
                  Neg =~ FREE_5+FREE_6+FREE_7+FREE_8 +FREE_13+FREE_14+FREE_15+FREE_16 '

# 2nd order (Hierachical model: scales using the latent variables on the first order half )
secOrder.model = ' PosExp =~ FREE_1+FREE_2+FREE_3+FREE_4
                   NegExp =~ FREE_5+FREE_6+FREE_7+FREE_8
                   PosSup =~ FREE_9+FREE_10+FREE_11+FREE_12
                   NegSup =~ FREE_13+FREE_14+FREE_15+FREE_16
                   Exp =~ PosExp+NegExp
                   Sup =~ PosSup+NegSup '

# The new reduced correlation model (4 sub-scales without 2nd order factors)
# This model was created because the hierarchical model with 2nd order factors presents redundant co-variances (i.e. standarized factor loadings that are = 1.003 and some negative variances)
hierRed.model <- ' PosExp =~ FREE_1+FREE_2+FREE_3+FREE_4
                   NegExp =~ FREE_5+FREE_6+FREE_7+FREE_8
                   PosSup =~ FREE_9+FREE_10+FREE_11+FREE_12
                   NegSup =~ FREE_13+FREE_14+FREE_15+FREE_16 '


# # Fit the models (test for domain specific factors accounted/controled by a global[generalized] which takes out the correlations)
oneFactor.fit = cfa(oneFactor.model, data=FREE16items) 
firstDual.fit = cfa(first.model, data=FREE16items)
secDual.fit = cfa(secDual.model, data=FREE16items) # dual factor = valence model

# Original hierarchical model including 2nd order factors (i.e., supress and enhance, as described in [Burton and Bonanno, 2016; Chen and Bonanno, 2018])
secOrder.fit = cfa(secOrder.model, data=FREE16items)

# reduced model
hierRed.fit <- cfa(hierRed.model, data=FREE16items) 
# In this model, constraining the latent factors to have a mean of 0 and a variance of 1 (setting: std.lv=TRUE), 
# i.e., forcing the latent factors to be correlation instead of covariances results in equivalent standarized parameter estimates.

# reduced model with no covariances (to test the hypothesis of co-variance betwen latent factors in the prev model)
hierRed2.fit = cfa(hierRed.model, data=FREE16items, orthogonal = TRUE)


# # fit indices
fitMeasures(oneFactor.fit)
fitMeasures(firstDual.fit)
fitMeasures(secDual.fit)
fitMeasures(secOrder.fit)

# # get summaries
summary(oneFactor.fit, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)
summary(firstDual.fit, fit.measures = TRUE, standardized=TRUE, rsquare=TRUE)
summary(secDual.fit, fit.measures = TRUE, standardized=TRUE, rsquare=TRUE)

summary(secOrder.fit, fit.measures = TRUE, standardized=TRUE, rsquare=TRUE)
summary(hierRed.fit, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

# # get parameter estimates
parameterEstimates(oneFactor.fit, standardized=TRUE, rsquare = TRUE)
parameterEstimates(firstDual.fit, standardized=TRUE, rsquare = TRUE)
parameterEstimates(secDual.fit, standardized=TRUE, rsquare = TRUE)

parameterEstimates(secOrder.fit, standardized=TRUE, rsquare = TRUE)
#parameterEstimates(hierRed.fit, rsquare = TRUE)
parameterEstimates(hierRed.fit, standardized=TRUE, rsquare = TRUE)

# # Seeing the model specification in detail
# Of particular interest may be the covariance matrix (see the '$psi')
#lavInspect(oneFactor.fit,"standardized") # psi --> empty in this model (don't really need to run!!)
lavInspect(firstDual.fit,"standardized") # 
lavInspect(secDual.fit,"standardized") # 

lavInspect(secOrder.fit,"standardized") # 
lavInspect(hierRed.fit,"standardized")

# column est.std contains the standarized parameter estimates (correlations/covariances)
standardizedSolution(oneFactor.fit) 
standardizedSolution(secOrder.fit) 
standardizedSolution(hierRed.fit)


# # Statistically compare the global fit of the models against the reduced hierarchical one
# --> look at the chisq-difference, corresponding DoF-difference and the p-val
# anova  for (lavaan-class objects) is a wrapper r around the function lavTestLRT (likelihood ratio test) 
#
# First check the difference with the reduced hierachical model and the 2nd order factor Model
anova(secOrder.fit, hierRed.fit) # compare the fits between the secondOrder- and the reduced-hirarchical models
# all models are compared against the reduced hierarchical model
anova(oneFactor.fit, hierRed.fit) 
anova(firstDual.fit, hierRed.fit) 
anova(secDual.fit, hierRed.fit) 
anova(hierRed2.fit, hierRed.fit) # not included in the paper since ist just a theory testing (i.e., there no-covariance between all latent factors)


# # Now plot the hierachical model:
semPaths(hierRed.fit,
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




