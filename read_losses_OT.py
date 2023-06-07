import os
import argparse
import statistics


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--Net_trained', type=int, default=100, choices=[20,50,80,100])
parser.add_argument('--sample_size', type=int, default=100, choices=[5, 10, 20, 30, 100, 50, 80, 70, 120])
main_args = parser.parse_args()


def read_loss(Net_trained, sample_size, dataset = 'CIFAR10'):
    dir_path = '{}/OT/Net_Trained_{}_Samples_{}'.format(dataset, Net_trained, sample_size)
    files = os.listdir(dir_path)

    losses = [] # List to store all losses

    for file_name in files:
        with open(os.path.join(dir_path, file_name), 'r') as file:
            contents = file.read()
            losses.append(float(contents))  # Store each loss in the list

    average = sum(losses) / len(losses)
    stddev = statistics.stdev(losses)  # Calculate standard deviation

    print('The average is:', average)
    # print('The standard deviation is:', stddev)

    return average, stddev

for net_trained in [20,50,80,100]:
    print('For Net Trained with collect samples: ', net_trained)
    for sample_size in [10, 20, 50, 100]:
        read_loss(Net_trained = net_trained, sample_size = sample_size)
    print('Done for one fixed sample size')