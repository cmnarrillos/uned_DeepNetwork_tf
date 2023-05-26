
import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = './tests'

# Loop over subdirectories
for root, dirs, files in os.walk(folder_path):
    for directory in dirs:
        subdirectory_path = os.path.join(root, directory)

        test_to_analyze = directory
        filename = './tests/' + test_to_analyze + '/console_log.txt'

        models = []
        validation_accuracy = []
        validation_accuracy_arc = []
        # epochs = []
        # epochs_arc = []
        # ii_epch = 0
        with open(filename, 'r') as f:
            for line in f.readlines():
                if ' NEW CASE: ' in line:
                    if 'ReLU)' in line:
                        activation = ' (ReLU)'
                    elif 'ReLU_mod' in line:
                        activation = ' (ReLU_mod)'
                    else:
                        activation = ' (sigmoid)'
                if 'Architecture: ' in line:
                    models.append(line[13:] + activation)
                    # epochs.append(epochs_arc)
                    # epochs_arc = []
                    # ii_epch = 0
                    validation_accuracy.append(validation_accuracy_arc)
                    validation_accuracy_arc = []
                if 'Expanded training data' in line:
                    models[-1] = models[-1] + ' + Expanded data'
                if 'Dropout' in line:
                    models[-1] = models[-1] + ' + Dropout'
                # if 'Epoch' in line:
                #     ii_epch += 1
                #     epochs_arc.append(ii_epch)
                if 'val_accuracy:' in line:
                    validation_accuracy_arc.append(
                        float(line.split(':')[-1].strip()))

        validation_accuracy.append(validation_accuracy_arc)
        val_acc_np = np.array(validation_accuracy[1:])
        # epochs_np = np.array(epochs[1:])

        # Set the figure size
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plotting
        for i in range(len(models)):
            plt.plot(val_acc_np[i], label=models[i])
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy for Different Models')
        plt.legend()
        plt.grid('minor')

        # Save the figure
        plt.savefig('./tests/' + test_to_analyze + '/validation_acc.png',
                    dpi=900, bbox_inches='tight')

        if test_to_analyze == '2023-05-24_03-51-19':
            # Set the figure size
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plotting
            for i in range(len(models[:-2])):
                plt.plot(val_acc_np[i], label=models[i])
            plt.xlabel('Epochs')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy for Different Models')
            plt.legend()
            plt.grid('minor')

            # Save the figure
            plt.savefig('./tests/' + test_to_analyze +
                        '/validation_acc_2.png',
                        dpi=900, bbox_inches='tight')

        # Close the figure (optional)
        plt.close()

        # Save vars
        np.save('./tests/' + test_to_analyze +
                '/val_acc_np.npy', val_acc_np)
        with open('./tests/' + test_to_analyze +
                  '/models.txt', 'w') as f:
            for string in models:
                f.write(string + '\n')


print('Postprocess Finished')

