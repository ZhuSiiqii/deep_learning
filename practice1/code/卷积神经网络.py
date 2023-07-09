import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# print(type(train_images))
# print(type(train_labels))
# print(train_images.shape)
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # 卷积层1， 3*3卷积核
    layers.MaxPooling2D((2, 2)),  # 池化层1,2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层2， 卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层2,2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层3,卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层3,2*2采样
    layers.Flatten(),  # Flattern层，连接卷积层和全连接层
    layers.Dense(64, activation='relu'),  # 全连接层，特征进一步提取
    layers.Dropout(0.2),
    layers.Dense(10)  # 输出层，输出预期结果
])

model.summary()  # 打印网络结构

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_split=0.1)

# 准确率accuracy
plt.figure(1)
plt.plot(history.history['accuracy'], c='r', label='train')
plt.plot(history.history['val_accuracy'], c='g', label='validation')
plt.legend()
plt.title('Train and Validation Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')

# loss值
plt.figure(2)
plt.plot(history.history['loss'], c='r', label='train')
plt.plot(history.history['val_loss'], c='g', label='validation')
plt.legend()
plt.title('Train and Validation Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

