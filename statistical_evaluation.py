import statistics

from models import advanced_cnn as advanced_cnn
from models import simple_cnn as simple_cnn
from utils import utils

print_epochs = 5
image_pairs = 1000

batch_size = 100
hidden_layers = 128
epochs = 25

if __name__ == "__main__":
    print(
        '\nTraining and Testing all models. After 10 rounds all relevant statistics will be generated. Consider grabbing a coffee...')

    training_accuracies_simple_cnn = []
    training_accuracies_advanced_cnn = []
    validation_accuracies_simple_cnn = []
    validation_accuracies_advanced_cnn = []
    testing_accuracies_simple_cnn = []
    testing_accuracies_advanced_cnn = []

    for ROUND in range(0, 10):
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

    print('\nEvaluation for 10 rounds has been completed!')

    print(
        f'SimpleCNN training accuracy mean +/- stdev: {training_accuracies_simple_cnn_mean:.2f}% +/- {training_accuracies_simple_cnn_std:.2f}%')
    print(
        f'SimpleCNN validation accuracy mean +/- stdev: {validation_accuracies_simple_cnn_mean:.2f}% +/- {validation_accuracies_simple_cnn_std:.2f}%')
    print(
        f'SimpleCNN testing accuracy mean +/- stdev: {testing_accuracies_simple_cnn_mean:.2f}% +/- {testing_accuracies_simple_cnn_std:.2f}%')

    print(
        f'AdvancedCNN training accuracy mean +/- stdev: {training_accuracies_advanced_cnn_mean:.2f}% +/- {training_accuracies_advanced_cnn_std:.2f}%')
    print(
        f'AdvancedCNN validation accuracy mean +/- stdev: {validation_accuracies_advanced_cnn_mean:.2f}% +/- {validation_accuracies_advanced_cnn_std:.2f}%')
    print(
        f'AdvancedCNN testing accuracy mean +/- stdev: {testing_accuracies_advanced_cnn_mean:.2f}% +/- {testing_accuracies_advanced_cnn_std:.2f}%')
