#��������� ����������� ����������
library('keras')
library('tensorflow')

# ��������� ���� ������ � ����������� �������
mnist <- dataset_mnist()

#��������� ��� ���� �� ������������� � �������� �������,
#� ��, � ���� �������, �� ������� ����������� ������� � ���� �����
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#������ ����������� ��������� ����
network <- keras_model_sequential()%>%
  layer_dense (units=512, activation='relu', input_shape = c(28*28)) %>%
  layer_dense(units=10, activation='softmax')

#��������� ��������� ���������� � ������
network %>% compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=c('accuracy'))

#������ ������� �� ������� � �������� ������������
train_images <- array_reshape(train_images, c(60000, 28*28))

#������ ������� ��������
train_images <- train_images/255

#�� �� ����������� � ��������� �������
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

#������� ��������� ��� �������
train_labels <- to_categorical(train_labels)
test_labels <-to_categorical(test_labels)

#��������� ��������� ���� �� 15 ������
network %>% fit(train_images, train_labels, epochs=15, batch_size=128)

#�������� ���������� ������ ��������� 99,98%

# ��������� ���������� ������ � �������� ������
metric <- network %>% evaluate(test_images, test_labels)

# �������� �� �������� ������� ��������� 98,21%

# ������������� ������ � ��������� 10 �������� �� �������� �������
prediction1 <- network %>% predict_classes(test_images[1:10,])
prediction2 <- network %>% predict_classes(test_images[9991:10000,])

# ������� �������� �����, ���������� � ������ � ��������� 10 ��������
test_labels1 <-mnist$test$y[1:10]
test_labels2 <-mnist$test$y[9991:10000]

# ������ ������� � ����������� �������
tabl <- t(rbind(prediction1, test_labels1, prediction2, test_labels2))

# �� ������� �����, ��� ��� ������������� ���������� ������� � ���������

# ���������� ���������� ���������� ����� �� ��������� 10 ������
for (i in 9991:10000) {
  image(as.matrix(mnist$test$x[i, 1:28, 1:28]))
  }

