from models import advanced_cnn as advanced_cnn
from models import simple_cnn as simple_cnn
from utils import utils as utils

# Parameters for all models
batch_size = 100
nb_epochs = 25
print_step = 5
hidden_layers = 128

# Description: TODO: Add a proper explanation
# The advanced neural network separates the two channels and predicts separately the value of each image.
# Using a simple logic comparison we then determine for each pair if the first digit is lesser or equal to the second

if __name__ == "__main__":
    print('\nTraining and Testing all models. Consider grabbing a coffee...')

    print('\nThe SimpleCNN is being trained and tested...')
    error_rate_simple_cnn = utils.evaluate_simple_cnn(1000, batch_size, nb_epochs, print_step,
                                                      simple_cnn.SimpleConvNet)

    print('\nThe AdvancedCNN is being trained and tested...')
    error_rate_advanced_cnn = utils.evaluate_advanced_cnn(1000, batch_size, nb_epochs, print_step, hidden_layers,
                                              advanced_cnn.AdvancedConvNet)

    print('\nTraining and Testing for all models has been completed!')
    print('Testing 1000 pairs resulted in the following accuracies:')
    print(f'SimpleCNN : {100.0 - error_rate_simple_cnn:.2f}%')
    print(f'AdvancedCNN : {100.0 - error_rate_advanced_cnn:.2f}%')
