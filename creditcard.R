# Packages
library(dplyr)
library(keras)

# Import data
credit_card.df <- read.csv("./creditcard.csv")

# Create training and test data sets
tr <- sample(nrow(credit_card.df), 100000, replace = T)
nw <- sample(nrow(credit_card.df), 100000, replace = T)

tr_data <- as.matrix(credit_card.df[tr, c(-1, -31)])
tr_output <- credit_card.df[tr, 31]
nw_data <- as.matrix(credit_card.df[nw, c(-1, -31)])
nw_output <- credit_card.df[nw, 31]

# Create regression models
credit_card.lm <- lm(Class ~ ., data = credit_card.df[tr, ])
credit_card.logit <- glm(Class ~ .,
                          data = credit_card.df[tr, ],
                          family = binomial(link = "logit"))

# Create classifier for logit
credit_card.logit.pr <-
    as.numeric(credit_card.logit$fitted.values > 0.5)

# Confusion table for logit predictor
credit_card.logit.conf <- addmargins(table(credit_card.df[nw, 'Class'],
                                     credit_card.logit.pr))

# Predicting with regression models
credit_card.lm.pr <- predict(credit_card.lm, credit_card.df[nw, ])
credit_card.logit.pr <- predict(credit_card.logit, credit_card.df[nw, ])

# Create deep learning model
credit_card.dl <- keras_model_sequential() %>%
	layer_dense(units = 20, activation = "relu", input_shape = c(29)) %>%
	layer_dense(units = 10, activation = "relu") %>%
	layer_dense(units = 1, activation = "sigmoid") %>%
	compile(
		optimizer = "sgd",
		loss = 'binary_crossentropy',
		metrics = c('binary_accuracy')
	)

# Train dl model
history <- credit_card.dl %>%
	fit(tr_data,
		tr_output,
		epochs = 30,
		batch_size = 100)

# Evaluate dl model
credit_card.dl %>% evaluate(nw_data, nw_output)
credit_card.dl.predict <- credit_card.dl %>% predict_classes(nw_data)
credit_card.dl.conf <- addmargins(table(credit_card.dl.predict, nw_output))

# Comparision of Logit and DL predictors
credit_card.logit.conf
credit_card.dl.conf
