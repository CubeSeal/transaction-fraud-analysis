# library(dplyr)
library(ggplot2)
library(keras)

credit_card.df <- read.csv("./creditcard.csv")

# tr <- sample(nrow(credit_card.df), 100000)
# nw <- sample(nrow(credit_card.df), 100000)

# credit_card.lm <- lm(Class ~ ., data = credit_card.df)

# credit_card.logit <- glm(Class ~ .,
#                           data = credit_card.df,
#                           family = binomial(link = "logit"))
# 
# credit_card.logit.pr <-
#     as.numeric(credit_card.logit$fitted.values > 0.5)
# credit_card.conf <- addmargins(table(credit_card.df$Class,
#                                      credit_card.logit.pr))

# credit_card.conf

# credit_card.lm.pr <- predict(credit_card.lm, credit_card.df[nw, ])
# credit_card.logit.pr <- predict(credit_card.logit, credit_card.df[nw, ])

# Less go deep learning time :)

tr <- sample(nrow(credit_card.df), 100000)
nw <- sample(nrow(credit_card.df), 100000)

model <- keras_model_sequential() %>%
	layer_dense(units = 10, activation = "sigmoid", input_shape = c(29)) %>%
	layer_dense(units = 1, activation = "sigmoid") %>%
	compile(
		optimizer = "sgd",
		loss = 'mse',
		metrics = c('accuracy')
	)

tr_data <- as.matrix(credit_card.df[tr, c(-1, -31)])
test_data <- as.matrix(credit_card.df[nw, c(-1, -31)])

history <- model %>%
	fit(tr_data,
		credit_card.df[tr, 31],
		epochs = 30,
		batch_size = 100)

model %>% evaluate(test_data, credit_card.df[nw, 31])
