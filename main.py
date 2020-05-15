import statistics

from models import advanced_cnn as advanced_cnn
from models import simple_cnn as simple_cnn
from utils import utils as utils

# Parameters for all models
print_epochs = 5
image_pairs = 1000

batch_size = 100
hidden_layers = 128
epochs = 25

# Description: TODO: Add a proper explanation
# The SimpleCNN contains 2 inputs and 1 output. It compares the two images in the two channels and makes a prediction
# straight away to determine for each pair if the first digit is lesser or equal to the second

# The AdvancedCNN  contains 1 input and 10 outputs. It separates the two channels and predicts separately the value of
# each image. Logic comparison is subsequently used to determine for each pair if the first digit is lesser or equal
# to the second

if __name__ == "__main__":
    print(
        '\nTraining and Testing all models. After 10 rounds all relevant statistics will be generated. Consider grabbing a coffee...')

    training_accuracies_simple_cnn = []
    training_accuracies_advanced_cnn = []
    validation_accuracies_simple_cnn = []
    validation_accuracies_advanced_cnn = []
    testing_accuracies_simple_cnn = []
    testing_accuracies_advanced_cnn = []

    for ROUND in range(0, 2):
        # The handling function takes care of the training as well as of the testing of the models

        print('\nThe SimpleCNN is being trained and tested...')
        testing_accuracy_simple_cnn, training_accuracy_simple_cnn, validation_accuracy_simple_cnn = utils.handle_simple_cnn(
            image_pairs, batch_size, epochs, print_epochs, hidden_layers,
            simple_cnn.SimpleConvolutionalNeuralNetwork)

        training_accuracies_simple_cnn.append(statistics.mean(training_accuracy_simple_cnn))
        validation_accuracies_simple_cnn.append(statistics.mean(validation_accuracy_simple_cnn))
        testing_accuracies_simple_cnn.append(testing_accuracy_simple_cnn)

        print('\nThe AdvancedCNN is being trained and tested...')
        testing_accuracy_advanced_cnn, training_accuracy_advanced_cnn, validation_accuracy_advanced_cnn = utils.handle_advanced_cnn(
            image_pairs, batch_size, epochs, print_epochs, hidden_layers,
            advanced_cnn.AdvancedConvolutionalNeuralNetwork)

        training_accuracies_advanced_cnn.append(statistics.mean(training_accuracy_advanced_cnn))
        validation_accuracies_advanced_cnn.append(statistics.mean(validation_accuracy_advanced_cnn))
        testing_accuracies_advanced_cnn.append(testing_accuracy_advanced_cnn)

        print('\nTraining and Testing for all models has been completed!')
        print('Testing 1000 pairs resulted in the following accuracies:')
        print(f'SimpleCNN : {testing_accuracy_simple_cnn:.2f}%')
        print(f'AdvancedCNN : {testing_accuracy_advanced_cnn:.2f}%')

    training_accuracies_simple_cnn_mean = statistics.mean(training_accuracies_simple_cnn)
    training_accuracies_advanced_cnn_mean = statistics.mean(training_accuracies_advanced_cnn)
    validation_accuracies_simple_cnn_mean = statistics.mean(validation_accuracies_simple_cnn)
    validation_accuracies_advanced_cnn_mean = statistics.mean(validation_accuracies_advanced_cnn)
    testing_accuracies_simple_cnn_mean = statistics.mean(testing_accuracies_simple_cnn)
    testing_accuracies_advanced_cnn_mean = statistics.mean(testing_accuracies_advanced_cnn)

    training_accuracies_simple_cnn_std = statistics.stdev(training_accuracies_simple_cnn)
    training_accuracies_advanced_cnn_std = statistics.stdev(training_accuracies_advanced_cnn)
    validation_accuracies_simple_cnn_std = statistics.stdev(validation_accuracies_simple_cnn)
    validation_accuracies_advanced_cnn_std = statistics.stdev(validation_accuracies_advanced_cnn)
    testing_accuracies_simple_cnn_std = statistics.stdev(testing_accuracies_simple_cnn)
    testing_accuracies_advanced_cnn_std = statistics.stdev(testing_accuracies_advanced_cnn)

    #Carful! 100-all indivdual values

    print('\nEvaluation for 10 rounds has been completed!')
