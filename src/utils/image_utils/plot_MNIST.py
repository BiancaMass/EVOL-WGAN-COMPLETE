from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


num = 20
images = X_train[:num]
labels = Y_train[:num]

num_row = 4
num_col = 5

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1*num_col, 1.5*num_row))
fig.suptitle("Sample images from the MNIST dataset", fontsize=22)
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f'{labels[i]}', fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.show()