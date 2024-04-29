from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

mask = (Y_train == 0) | (Y_train == 1)
X_filtered = X_train[mask]
Y_filtered = Y_train[mask]

num = 20
images = X_filtered[:num]
labels = Y_filtered[:num]

num_row = 4
num_col = 5

# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1*num_col, 1.5*num_row))
fig.suptitle("Sample images of digits 0 and 1 from the MNIST dataset", fontsize=22)
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()