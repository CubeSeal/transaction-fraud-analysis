# Packages
library(data.table)
library(dplyr)
library(keras)

# Import data
df <- fread("./creditcard.csv", data.table = F)

# Create training and test data sets
tr <- 1:nrow(df) %in% sample(nrow(df), 10^5, replace = F)
nw <- !tr

tr_data <- as.matrix(df[tr, c(-1, -31)])
tr_output <- df[tr, 31]
nw_data <- as.matrix(df[nw, c(-1, -31)])
nw_output <- df[nw, 31]

# Create regression models
lm <- lm(Class ~ ., data = df[tr, ])
logit <- glm(Class ~ .,
                          data = df[tr, ],
                          family = binomial(link = "logit"))

# Create classifier for logit
logit.pr <- as.numeric(predict.glm(logit, df[nw, ], type = "response") > 0.5)

# Load deap learning model from hdf5 file
dl <- load_model_hdf5("./DL.hdf5")

# Create deep learning model
# dl <- keras_model_sequential() %>%
# 	layer_dense(units = 20, activation = "elu", input_shape = c(29)) %>%
# 	layer_dense(units = 10, activation = "elu") %>%
# 	layer_dense(units = 5, activation = "elu") %>%
# 	layer_dense(units = 1, activation = "sigmoid")
# dl %>% compile(
# 		optimizer = optimizer_sgd(lr = 0.01,
# 								  momentum = 0,
# 								  decay = 0),
# 		loss = 'binary_crossentropy',
# 		metrics = c('binary_accuracy')
# 	)

# Train dl model
history <- dl %>%
	fit(tr_data,
		tr_output,
		epochs = 10,
		batch_size = 100)

# Evaluate dl model
dl %>% evaluate(nw_data, nw_output)

# Confusion table for logit predictor
logit.conf <- addmargins(table(logit.pr, nw_output))

# Confusion table for dl predictor
dl.predict <- dl %>% predict_classes(nw_data)
dl.conf <- addmargins(table(dl.predict, nw_output))

# Comparision of Logit and DL predictors
logit.conf
dl.conf

# Save model parameters to hdf5 file
dl %>% save_model_hdf5("./DL.hdf5")
