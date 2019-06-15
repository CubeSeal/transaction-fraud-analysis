# library(dplyr)
library(ggplot2)
library(gganimate)

credit_card.df <- read.csv("./creditcard.csv")

# tr <- sample(nrow(credit_card.df), 100000)
# nw <- sample(nrow(credit_card.df), 100000)

# credit_card.lm <- lm(Class ~ ., data = credit_card.df)

credit_card.logit <- glm(Class ~ .,
                          data = credit_card.df,
                          family = binomial(link = "logit"))

credit_card.logit.pr <-
    as.numeric(credit_card.logit$fitted.values > 0.5)
credit_card.conf <- addmargins(table(credit_card.df$Class,
                                     credit_card.logit.pr))
credit_card.conf

# credit_card.lm.pr <- predict(credit_card.lm, credit_card.df[nw, ])
# credit_card.logit.pr <- predict(credit_card.logit, credit_card.df[nw, ])