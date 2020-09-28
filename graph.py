import matplotlib.pyplot as plt

options = ['3n', '5n', '7n', '9n']
colors = {'3n':'y', '5n':'c', '7n':'g', '9n':'r'}

# Plot the plain networks without residual connections
plt.subplot(1,2,1)
for n in options:
    filename = "data/plain/" + n

    # Input the train and test accuracy from their files
    acc_list = []
    acc_list_test = []
    try:
        with open(filename + "_train", "r") as f:
            for line in f:
                acc_list.append(1-float(line.strip()))

        with open(filename + "_test", "r") as f:
            for line in f:
                acc_list_test.append(1-float(line.strip()))
    
    # Plot the train and test accuracy
        plt.plot(acc_list, colors[n] + '--')
        plt.plot(acc_list_test, colors[n]+'-')
    except:
        print("No file")

# Create legend and organize graph appearance
plt.ylim(0, .2)
plt.title('Plain')
plt.ylabel('Percent Error')
plt.xlabel('epoch')
plt.legend(['Plain-20', '', 'Plain-32', '', 'Plain-44', '', 'Plain-56'], loc='upper right')



# Plot the Residual Networks 
plt.figure(1)
plt.subplot(1,2,2)
for n in options:
    filename = "data/resnet/" + n

    # Input the train and test accuracy from their files
    acc_list = []
    acc_list_test = []
    try:
        with open(filename + "_train", "r") as f:
            for line in f:
                acc_list.append(1-float(line.strip()))

        with open(filename + "_test", "r") as f:
            for line in f:
                acc_list_test.append(1-float(line.strip()))

    # Plot the train and test accuracy
        plt.plot(acc_list, colors[n] + '--')
        plt.plot(acc_list_test, colors[n]+'-')
    except:
        print("No file")


# Create legend and organize graph appearance
plt.ylim(0, .2)
plt.title('Resnet')
plt.ylabel('Percent Error')
plt.xlabel('epoch')
plt.legend(['ResNet-20', '', 'ResNet-32', '', 'ResNet-44', '', 'ResNet-56'], loc='upper right')
plt.show()
