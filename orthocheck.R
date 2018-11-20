# Load packages and set theme
library(ggpubr)
library(ggpmisc)

# configuration
file.path = '/Users/woodie/Desktop/workspace/Avoiding-Bias-Data/data/robbery.biased.keywords.txt'
keywords  = c('black', 'males', 'black_males')

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
p <- ggplot(comparison.df, aes(Keywords.TFIDF, Robbery.TFIDF)) +
  geom_point(aes(color = keywords), size = 3, alpha = 0.6) +
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07"))+
  facet_wrap(~keywords)

formula <- y ~ x
p + 
  stat_smooth( aes(color = keywords, fill = keywords), method = "lm") +
  stat_cor(aes(color = keywords), label.y = 0.24)+
  stat_poly_eq(
    aes(color = keywords, label = ..eq.label..),
    formula = formula, label.y = 0.22, parse = TRUE)

