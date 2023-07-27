import gzip
import numpy as np
import torch.nn as nn
from urllib import request
from matplotlib import pyplot as plt
import torch
from torch import optim
import torch.utils.data
from torch.nn.functional import one_hot


def show_images(images, labels):
    # Extract the image indices and reshaped pixels
    pixels = images.reshape(-1, 28, 28)
    # Create a figure with subplots for each image
    fig, axs = plt.subplots(
        ncols=5, nrows=10, figsize=(10, 3 * len(images))
    )

    for i in range(len(images)):
        # Display the image and its label
        axs[i//5, i % 5].imshow(pixels[i], cmap="gray")
        axs[i//5, i % 5].set_title("Label: {}".format(labels[i]))

        # Remove the tick marks and axis labels
        axs[i//5, i % 5].set_xticks([])
        axs[i//5, i % 5].set_yticks([])

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the figure
    plt.show()


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Set the number of hidden units
        self.num_hidden = 8
        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),  # input size: 28*28, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, self.num_hidden),  # input size: 256, output size: num_hidden
            nn.ReLU(),  # apply the ReLU activation function
        )
        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),  # input size: num_hidden, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, 784),  # input size: 256, output size: 784
            nn.Sigmoid(),  # apply the sigmoid activation function to compress the output to a range of (0, 1)
        )

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded


class VAE(AutoEncoder):
    def __init__(self):
        super().__init__()
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)    # 均值
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)  # 方差

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.num_hidden).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples


class ConditionalVAE(VAE):
    # VAE implementation from the article linked above
    def __init__(self, num_classes):
        super().__init__()
        # Add a linear layer for the class label
        self.label_projector = nn.Sequential(
            nn.Linear(num_classes, self.num_hidden),
            nn.ReLU(),
        )

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label

    def forward(self, x, y):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(self.condition_on_label(z, y))
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples, y):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.num_hidden).to(device)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(self.condition_on_label(z, y))
        # Return the generated samples
        return samples


# Download the files
url = "http://yann.lecun.com/exdb/mnist/"
filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
data = []
for filename in filenames:
#    print("Downloading", filename)
#    request.urlretrieve(url + filename, filename)
    with gzip.open(filename, 'rb') as f:
        if 'labels' in filename:
            # frombuffer(流文件，返回值类型， 从第几位开始读)
            data.append(np.frombuffer(f.read(), np.uint8, offset=8))
        else:
            # reshape（-1表示自动计算，28*28）
            data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28))

# Split into training and testing sets
X_train, y_train, X_test, y_test = data
# Normalize the pixel values
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# show_images(X_train[:6], y_train[:6])
# Convert the training data to PyTorch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
# Create the autoencoder model and optimizer
model = ConditionalVAE(num_classes=10)
learning_rate = 0.001
num_epochs = 5
batch_size = 32
num_samples = 50
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
criterion = nn.MSELoss()

# Set the device to GPU if available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)

# Create a DataLoader to handle batching of the training data

train_loader = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
# Training loop
loss_list = []
for epoch in range(num_epochs):
    total_loss = 0.0
    cnt = 0
    for batch_idx, (data, label) in enumerate(train_loader, 0):
        # Get a batch of training data and move it to the device
        data = data.to(device)
        label = one_hot(torch.LongTensor(label), 10)
        label = label.to(device)

        # Forward pass
        encoded, decoded, mu, log_var = model(data, label)

        # Compute the loss and perform backpropagation
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = criterion(decoded, data) + 3 * KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the running loss
        total_loss += loss.item()
        cnt += 1

    # Print the epoch loss
    epoch_loss = total_loss / cnt
    loss_list.append(epoch_loss)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )

plt.figure(1)
x = np.linspace(0, num_epochs - 1, num_epochs)
plt.plot(x, loss_list, c='r', label="train_loss")
plt.legend()
plt.title("Train and Test loss  lr={}".format(learning_rate))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

random_labels = [0]*5+[1]*5+[2]*5+[3]*5+[4]*5+[5]*5+[6]*5+[7]*5+[8]*5+[9]*5
y_onehot = one_hot(torch.LongTensor(random_labels), 10).to(device)
show_images(
     model.sample(num_samples, y_onehot)
     .cpu().detach().numpy(), labels=random_labels
)


