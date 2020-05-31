install.packages('BatchGetSymbols')
install.packages('plotly')
install.packages('minimax')

library('minimax')
library(plotly)
library(BatchGetSymbols)
library('keras')
library('tensorflow')

#Загрузка и подготовка данных (DAX)
tickers <- c('^GDAXI')
first.date <- Sys.Date() - 360*5
last.date <- Sys.Date()

yts <- BatchGetSymbols(tickers = tickers,
                       first.date = first.date,
                       last.date = last.date,
                       cache.folder = file.path(tempdir(),
                                                'BGS_Cache') )

y <-  yts$df.tickers$price.close
myts <-  data.frame(index = yts$df.tickers$ref.date, price = y, vol = yts$df.tickers$volume)
myts <-  myts[complete.cases(myts), ]
myts <-  myts[-seq(nrow(myts) - 1200), ]
myts$index <-  seq(nrow(myts))

#Стандартизация
myts <- data.frame(index = rminimax(myts$index), price = rminimax(myts$price), vol= rminimax(myts$vol))


plot_ly(myts, x = ~index, y = ~price, type = "scatter", mode = "markers", color = ~vol)
acf(myts$price, lag.max = 3000)

#Деление выборки на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <-  50

#Создание массивов

x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags))

### Обучение сети SM ###

SM <- keras_model_sequential() %>%
  layer_dense(units = 1000, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'sigmoid')

#SM - rmspop, mse

SM %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse')

SM %>% fit(x.train, y.train, epochs = 5, batch_size = 50)
a1 <- 0.0850

# SM - rmspop, mape

SM %>% compile(
  optimizer = 'rmsprop',
  loss = 'mape')

SM %>% fit(x.train, y.train, epochs = 18, batch_size = 50)
a2 <- 99.1522

# SM - adam, mse

SM %>% compile(
  optimizer = 'adam',
  loss = 'mse')

SM %>% fit(x.train, y.train, epochs = 10, batch_size = 50)
b1 <- 0.0854

# SM - adam, mape

SM %>% compile(
  optimizer = 'adam',
  loss = 'mape')

SM %>% fit(x.train, y.train, epochs = 20, batch_size = 50)
b2 <- 100.4834


### Обучение сети RNN ###

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 50) %>%
  layer_simple_rnn(units = 50) %>%
  layer_dense(units = 1, activation = "sigmoid")

# RNN - rmsprop, mse

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
c1 <- 0.0849

# RNN - rmsprop, mape

model %>% compile(
  optimizer = "rmsprop",
  loss = "mape",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
c2 <- 104.1117

# RNN - adam, mse

model %>% compile(
  optimizer = "adam",
  loss = "mse",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
d1 <- 0.0860

# RNN - adam, mape

model %>% compile(
  optimizer = "adam",
  loss = "mape",
)

history <- model %>% fit(
  x.train, y.train,
  epochs = 10,
  batch_size = 50,
  validation_split = 0.2
)
d2 <- 99.0145


### Подготовка данных LSTM ###

# Стандартизация
msd.price <-  c(mean(myts$price), sd(myts$price))
msd.vol <-  c(mean(myts$vol), sd(myts$vol))
myts$price <-  (myts$price - msd.price[1])/msd.price[2]
myts$vol <-  (myts$vol - msd.vol[1])/msd.vol[2]

# Деление на тестовую и тренировочную
datalags = 20
train <-  myts[seq(1000 + datalags), ]
test <-  myts[1000 + datalags + seq(200 + datalags), ]
batch.size <- 50

# Создание массивов
x.train <-  array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test <-  array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], dim = c(nrow(test) - datalags, datalags, 2))
y.test <-  array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

### Обучение сети LSTM ###

model <- keras_model_sequential()  %>%
  layer_lstm(units = 100,
             input_shape = c(datalags, 2),
             batch_size = batch.size,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1)

# LSTM - rmsprop, mse

model %>%
  compile(loss = 'mse', optimizer = 'rmsprop')

model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
e1 <- 1.0051

# LSTM - rmsprop, mape

model %>%
  compile(loss = 'mape', optimizer = 'rmsprop')

model %>% fit(x.train, y.train, epochs = 8, batch_size = batch.size)
e2 <- 119.2313

# LSTM - adam, mse

model %>%
  compile(loss = 'mse', optimizer = 'adam')

model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
f1 <- 1.0108

# LSTM - adam, mape

model %>%
  compile(loss = 'mape', optimizer = 'adam')

model %>% fit(x.train, y.train, epochs = 10, batch_size = batch.size)
f2 <- 146.6467


### Итоговая таблица ###
mse <- c(a1, b1, c1, d1, e1, f1)
mape <- c(a2, b2, c2, d2, e2, f2)

itog <- data.frame('Neural Net' = c(rep('SM', 2), rep('RNN', 2), rep('LSTM', 2)), 
                    'optimizer' = rep(c('rmsprop', 'adam'), 3),
                    'mse' = mse,
                    'mape' = mape)
itog

# Наилучшей является нейронная сеть RNN (rmsprop, mse)
# loss=0.0849
