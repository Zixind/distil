import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--Net_trained', type=int, default=100, choices=[20,50,80,100])
parser.add_argument('--sample_size', type=int, default=100, choices=[5, 10, 20, 30, 100, 50, 80, 70, 120])
main_args = parser.parse_args()


def read_loss(Net_trained, sample_size, dataset = 'CIFAR10'):
# Let's say your files are in the directory 'dir_path'
    dir_path = '{}/OT/Net_Trained_{}_Samples_{}'.format(dataset, Net_trained, sample_size)

# Get a list of all files in the directory
    files = os.listdir(dir_path)

# Initialize a variable to hold the sum of all values
    total = 0

# Loop over all files
    for file_name in files:
        with open(os.path.join(dir_path, file_name), 'r') as file:
        # Read the contents of the file
            contents = file.read()

        # Convert the contents to a float and add to total
            total += float(contents)

# Compute the average
    average = total / len(files)

    print('The average is:', average)

    return average

for net_trained in [20,50,80,100]:
    print('For Net Trained with collect samples: ', net_trained)
    for sample_size in [10, 20, 50, 100]:
        read_loss(Net_trained = net_trained, sample_size = sample_size)
    print('Done for one fixed sample size')