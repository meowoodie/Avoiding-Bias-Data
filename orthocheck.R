# Load packages and set theme
library(ggpubr)
library(ggpmisc)
library(reshape2)
library(ggplot2)

## Scatter plots for comparison

# configuration
file.path = '/Users/woodie/Desktop/workspace/Avoiding-Bias-Data/data/robbery.biased.keywords.txt'
keywords  = c('black', 'males', 'black_males')
target    = c('robbery')

# data preprocessing
keywords.df = read.csv(file=file.path, header=FALSE, sep=",")
comparison.df = keywords.df[, c(1, 2)]
comparison.df = cbind(comparison.df, rep(keywords[1], length(comparison.df)))
colnames(comparison.df) = c('Robbery.TFIDF', 'Keywords.TFIDF', 'keywords')
for (i in 3:ncol(keywords.df)) {
  sub.comparison.df = keywords.df[, c(1, i)]
  sub.comparison.df = cbind(sub.comparison.df, rep(keywords[i-1], length(sub.comparison.df)))
  colnames(sub.comparison.df) = c('Robbery.TFIDF', 'Keywords.TFIDF', 'keywords')
  comparison.df = rbind(comparison.df, sub.comparison.df)
}
comparison.df = comparison.df[
  comparison.df$Robbery.TFIDF>0 &
  comparison.df$Robbery.TFIDF<0.25 &
  comparison.df$Keywords.TFIDF>0,]

# set theme
theme_set(
  theme_bw() +
    theme(legend.position = "top")
)
# scatter plot and regression line
p = ggplot(comparison.df, aes(Keywords.TFIDF, Robbery.TFIDF)) +
  geom_point(aes(color = keywords), size = 3, alpha = 0.6) +
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  facet_wrap(~keywords)

p + stat_cor(aes(color = keywords), label.y = 0.24)




## Covariance matrix plot

# configuration
file.path = '/Users/woodie/Desktop/workspace/Avoiding-Bias-Data/data/recon.10.biased.keywords.txt'
keywords  = c('burglary', 'robbery', 'carjacking', 'stole', 'jewelry', 'arrestee', 'jail',
              'shot', 'black', 'male', 'males', 'black_male', 'black_males')

# Get upper triangle of the correlation matrix
get_upper_tri = function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

# data preprocessing
keywords.df = read.csv(file=file.path, header=FALSE, sep=",")
colnames(keywords.df) = keywords

# calculate covariance matrix by extracting keywords pairs from raw data
cormat.list = list() 
pairs       = combn(keywords, 2)
for(i in 1:ncol(pairs)) {
  comparison.df = keywords.df[, pairs[,i]]                          # get cols of specific keywords
  comparison.df = comparison.df[rowSums(comparison.df <= 0) <= 0, ] # remove rows with zero value
  # zero padding for those pairs of keywords without data entries
  if (nrow(comparison.df) <= 1){
    cormat.list[[i]] = data.frame('Var1'=pairs[,i][1], 'Var2'=pairs[,i][2], 'value'=0)
  }
  # otherwise calculate their corvariance according to their data entries
  else{
    cormat        = round(cor(comparison.df), 2)
    upper.cormat  = get_upper_tri(cormat)
    melted.cormat = melt(upper.cormat, na.rm = TRUE)
    cormat.list[[i]] = melted.cormat
  }
}
merged.cormat = do.call(rbind, cormat.list)
merged.cormat = unique(merged.cormat) # remove duplicate rows in dataframe

# matrix plot
cust_breaks = keywords # reorder breaks for the plot
ggheatmap   = ggplot(merged.cormat, aes(Var2, Var1, fill = value)) +
  scale_x_discrete(limits=cust_breaks) + 
  scale_y_discrete(limits=cust_breaks) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()

# add correlation coefficients on the heatmap
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 2) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.5, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))
