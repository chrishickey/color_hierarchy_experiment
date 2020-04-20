#install.packages("userfriendlyscience")
#install.packages("FactoMineR")
#install.packages("car")
#install.packages("RVAideMemoire")
#install.packages("gplots")
#install.packages("ggpubr")

library(userfriendlyscience) # oneway
library(car) # leveneTest
library(RVAideMemoire) # byf.shapiro
library(gplots) # plotmeans
library(psych)
# SET WORKING DIRECTORY HERE
setwd('/home/chris/Desktop/color_hierarchy_experiments/experiment1')

# SET file.name TO THE RESULTS FILE YU WANT TO RUN ANOVA ANALYSIS ON !! 

#file.name <- "results/bgr_500_Rprocessible.csv"
file.name <- "results/rgb_500_Rprocessible.csv"
#file.name <- "results/ybr_500_Rprocessible.csv"
#file.name <- "results/yuv_500_Rprocessible.csv"
#file.name <- "results/opp_500_Rprocessible.csv"

group_df <- read.csv(file.name)
sapply(group_df, class)

# transform from 'integer' to 'factor'
group_df <- transform(group_df, colour = factor(colour))
sapply(group_df, class)

# Descriptive statistics
describeBy(group_df, group_df$colour)

# ANOVA condition 1: Check normality ##########
# Shapiro-Wilk test to check normality.
byf.shapiro(epoch ~ colour, data = group_df)
# If p-value of a group is greater than 0.05, then the group follows normality.
# Otherwise, it does not follow normality.

# ANOVA condition 2: Independency ##########
# Independency is guaranteed.

# ANOVA condition 3: Homogeneity of variances ##########
# Levene's test to check the homogeneity of variances
leven.test.result <- leveneTest(epoch ~ colour, data = group_df)
print(leven.test.result)


# If Pr(>F) is greater than 0.05, then go to (1) and perform ANOVA and the Tukey post-hoc test.
# Otherwise, go to (2) and perform the Welch's ANOVA and the Games-Howell post-hoc test.
leven.test.pValue <- leven.test.result$`Pr(>F)`[1]
if (leven.test.pValue >= 0.05) {
  one.way <- oneway(y = group_df$epoch, x = group_df$colour, posthoc = 'tukey')
  print(one.way)
} else {
  # (2.2) Post-hoc: games-howell ##########
  one.way <- oneway(y = group_df$epoch, x = group_df$colour, posthoc = 'games-howell')
  print(one.way)
}

plotmeans(epoch ~ colour, data = group_df, frame = TRUE)
