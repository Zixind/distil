import os
import argparse
import statistics
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SVHN', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--Net_trained', type=int, default=100, choices=[20,50,80,100])
parser.add_argument('--sample_size', type=int, default=100, choices=[5, 10, 20, 30, 100, 50, 80, 70, 120])
main_args = parser.parse_args()


def read_loss(Net_trained, sample_size, dataset = main_args.dataset):
    dir_path = '{}/OT/Net_Trained_{}_Samples_{}'.format(dataset, Net_trained, sample_size)
    files = os.listdir(dir_path)

    losses = [] # List to store all losses

    for file_name in files:
        with open(os.path.join(dir_path, file_name), 'r') as file:
            contents = file.read()
            losses.append(float(contents))  # Store each loss in the list

    average = sum(losses) / len(losses)
    try:
        stddev = statistics.stdev(losses)  # Calculate standard deviation
    except:
        stddev = 0
    size = len(losses)

    print('The average is:', average)
    # print('The standard deviation is:', stddev)

    return average, stddev, size

# for net_trained in [20,50,80,100]:
#     print('For Net Trained with collect samples: ', net_trained)
#     for sample_size in [10, 20, 50, 100]:
#         read_loss(Net_trained = net_trained, sample_size = sample_size)
#     print('Done for one fixed sample size')

average_losses = {}

for net_trained in [20,50,80,100]:
    average_losses[net_trained] = []
    average_cum = 0
    sizes_cum = 0
    # for sample_size in [10, 20, 50, 100]:
    sample_size = 100
        
    average, stddev, size = read_loss(Net_trained = net_trained, sample_size = sample_size)
    average_cum += average * size
    sizes_cum += size
    average = average_cum/sizes_cum
    average_losses[net_trained].append(average)

print(average_losses)
# # Plot the average loss values
# for net_trained, losses in average_losses.items():
#     plt.plot([10, 20, 50, 100], losses, label=f'Number of Utility Samples for DeepSets_OT Training: {net_trained}')

# plt.xlabel('Number of Utility Samples for evaluation')
# plt.ylabel('Average MSEs')
# plt.legend(fontsize = 8)
# plt.savefig('Average Loss vs. Sample Size for Dataset {}.png'.format(main_args.dataset))
# plt.show()