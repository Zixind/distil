import os
import argparse
import statistics
import matplotlib.pyplot as plt
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SVHN', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
# parser.add_argument('--Net_trained', type=int, default=100, choices=[20,50,80,100])
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

def calc_avg_losses(dataset = 'CIFAR10'):
    average_losses = {}
    stddev_losses = {}


    for net_trained in [20,50,80,100]:
        average_losses[net_trained] = []
        stddev_losses[net_trained] = []
        average_cum = 0
        sizes_cum = 0
        # for sample_size in [10, 20, 50, 100]:
        sample_size = main_args.sample_size
        
        average, stddev, size = read_loss(Net_trained = net_trained, sample_size = sample_size, dataset = dataset)
        average_cum += average * size
        sizes_cum += size
        average = average_cum/sizes_cum
        average_losses[net_trained].append(average)
        stddev_losses[net_trained].append(stddev)

    print(average_losses)
    return average_losses, stddev_losses




# define the plotting function
def plot_avg_std_losses(average_losses_dict, stddev_losses_dict, datasets):
    fig, ax = plt.subplots()

    # define the different markers for each dataset
    markers = ['o', 's', 'd']
    net_trained_values = [20,50,80,100]

    # iterate over the datasets
    for idx, dataset in enumerate(datasets):
        average_losses = average_losses_dict[dataset]
        stddev_losses = stddev_losses_dict[dataset]

        avg_losses = [average_losses[net_trained][0] for net_trained in net_trained_values]
        std_losses = [stddev_losses[net_trained][0] for net_trained in net_trained_values]

        # plot averages with error bars
        ax.errorbar(net_trained_values, avg_losses, yerr=std_losses, marker=markers[idx], label=f'{dataset}', capsize=5)

    ax.set_title(f'Average Losses of DeepSets + OT and Standard Deviation')
    ax.set_xlabel('Net Trained')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('Average Loss vs. Sample Size.png')
    plt.show()
    

# Call function
datasets = ['CIFAR10', 'MNIST', 'SVHN']
average_losses_dict = {}
stddev_losses_dict = {}

for dataset in datasets:
    average_losses, stddev_losses = calc_avg_losses(dataset=dataset)
    average_losses_dict[dataset] = average_losses
    stddev_losses_dict[dataset] = stddev_losses

plot_avg_std_losses(average_losses_dict, stddev_losses_dict, datasets)


# def plot_avg_losses(average_losses, dataset, ax):
#     # Iterate over each net_trained and its corresponding average losses
#     net_trained, avg_losses = zip(*average_losses.items())
    
#     ax.plot(net_trained, avg_losses, label=dataset)

# # List of datasets
# datasets = ['CIFAR10', 'MNIST', 'SVHN']

# # Create a new figure and axis
# fig, ax = plt.subplots()

# # For each dataset
# for dataset in datasets:
#     # Calculate average losses
#     average_losses = calc_avg_losses(dataset)

#     # Plot average losses on the given axis
#     plot_avg_losses(average_losses, dataset, ax)

# ax.set_title('Average Losses')  # Set title
# ax.set_xlabel('Sample Size')  # Set x-axis label
# ax.set_ylabel('Average Loss')  # Set y-axis label
# ax.legend()  # Show legend
# plt.savefig('Average Loss vs. Sample Size.png')
# plt.show()  # Display the plot






# for dataset in ['CIFAR10', 'SVHN', 'MNIST']:
#     avg_losses = calc_avg_losses(dataset = dataset)
    
    
# # Plot the average loss values
# for net_trained, losses in average_losses.items():
#     plt.plot([10, 20, 50, 100], losses, label=f'Number of Utility Samples for DeepSets_OT Training: {net_trained}')

# plt.xlabel('Number of Utility Samples for evaluation')
# plt.ylabel('Average MSEs')
# plt.legend(fontsize = 8)
# plt.savefig('Average Loss vs. Sample Size for Dataset {}.png'.format(main_args.dataset))
# plt.show()