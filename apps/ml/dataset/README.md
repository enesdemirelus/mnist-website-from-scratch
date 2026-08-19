# Dataset

This project uses the MNIST dataset in CSV format:

- `mnist_train.csv` — 60,000 training images
- `mnist_test.csv` — 10,000 test images

Each row is a single image: the first column is the label (0–9),
followed by 784 pixel values (28×28, grayscale 0–255).

These files are gitignored due to size. Download them from
[\[source\]](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) and place them in this folder before running training.
