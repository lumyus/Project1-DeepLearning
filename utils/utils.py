import torch
from torch import nn
from torch import optim

from utils import dlc_practical_prologue as prologue


def calculate_amount_errors_simple_cnn(model, input, target, batch_size):
    nb_data_errors = 0  # Amount of incorrect classifications
    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(batch_size):
            if target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def calculate_amount_errors_advanced_cnn(model, input, target, batch_size):
    nb_data_errors = 0  # Amount of incorrect classifications
    first_channel = []  # First channel testing
    second_channel = []  # Second channel testing
    result = []
    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(batch_size):
            if k % 2 == 0:  # Load the testing results into two arrays corresponding to the two channels
                first_channel.append(predicted_classes[k])
            else:
                second_channel.append(predicted_classes[k])

    for x in range(len(target)):
        if first_channel[x] > second_channel[
            x]:  # Compare if the first channel's number is greater than the second channel's
            result.append(0)
        else:
            result.append(1)

        if target.data[x] != result[x]:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def train_simple_cnn(model, train_input, train_target, validation_input, validation_target, device, nb_epochs,
                     batch_size, print_step):
    print("\nStarting to train the SimpleCNN!")

    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Capture historical losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    loss = 0

    for epoch in range(nb_epochs + 1):

        for i in range(0, train_input.size(0), batch_size):  # train_input.size(0) = 900

            output = model(train_input.narrow(0, i, batch_size))

            loss = criterion(output, train_target.narrow(0, i, batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create accuracy statistics at a print_step frequency
        if epoch % print_step == 0:
            training_loss.append(loss.item())
            print(f'\nEpoch : {epoch}, Loss: {loss.item():.4f}')

            # Change mode to testing
            model.eval()

            # Calculate training accuracy and print

            accuracy = (calculate_amount_errors_simple_cnn(model, train_input, train_target, batch_size) / (
                train_input.size(0))) * 100
            training_accuracy.append(accuracy)

            print(f'Training Accuracy : {100 - accuracy:.4f}%')

            # Calculate validation accuracy and print

            accuracy = (calculate_amount_errors_simple_cnn(model, validation_input, validation_target, batch_size) / (
                validation_input.size(0))) * 100
            validation_accuracy.append(accuracy)

            print(f'Validation Accuracy : {100 - accuracy:.4f}%')

            # Change mode back to training
            model.train()

    return model, training_loss, training_accuracy, validation_accuracy


def train_model_advanced_cnn(model, train_input, train_target, train_classes, validation_input, validation_target,
                             device, nb_epochs, batch_size, print_step):
    print("\nStarting to train the AdvancedCNN!")

    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Capture historical losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    loss = 0

    for epoch in range(nb_epochs + 1):

        for i in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, i, batch_size))

            loss = criterion(output, train_classes.narrow(0, i, batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create accuracy statistics at a print_step frequency
        if epoch % print_step == 0:
            training_loss.append(loss.item())
            print(f'\nEpoch : {epoch}, Loss: {loss.item():.4f}')

            # Change mode to testing
            model.eval()

            # Calculate training accuracy and print

            accuracy = (calculate_amount_errors_advanced_cnn(model, train_input, train_target, batch_size) / (
                train_input.size(0))) * 100
            training_accuracy.append(accuracy)

            print(f'Training Accuracy : {100 - accuracy:.4f}%')

            # Calculate validation accuracy and print

            accuracy = (calculate_amount_errors_advanced_cnn(model, validation_input, validation_target, batch_size) / (
                validation_input.size(0))) * 100
            validation_accuracy.append(accuracy)

            print(f'Validation Accuracy : {100 - accuracy:.4f}%')

            # Change mode back to training
            model.train()

    return model, training_loss, training_accuracy, validation_accuracy


def evaluate_simple_cnn(nb_pairs, batch_size, nb_epochs, print_step, simple_cnn):
    if torch.cuda.is_available():
        print('\nUsing GPU...\n')
        device = torch.device('cuda')
    else:
        print('\nUsing CPU...\n')
        device = torch.device('cpu')

    model = simple_cnn()

    # Print model characteristics
    print(model)

    # Load model onto device
    model.to(device)

    # Generate training and testing dataset
    train_input, train_target, train_classes, test_input, test_target, test_classes \
        = prologue.generate_pair_sets(nb_pairs)

    # Calculate data mean and standard deviation
    mean = train_input.mean()
    std = train_input.std()

    # Normalize the data (following dlc_practical_proloque.py example)
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Following https://stackoverflow.com/
    # /questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    # Split the training and testing into is not done. We keep 1000 testing pairs since there is enough data available
    # Split the training data into training and validation at 80%:20%
    # Example: 1000 pairs would result in 800 training pairs and 200 validation pairs

    validation_input = train_input[800:1000]
    validation_target = train_target[800:1000]

    train_input = train_input[0:800]
    train_target = train_target[0:800]

    # Train model
    model, training_loss, training_accuracy, validation_accuracy \
        = train_simple_cnn(model, train_input, train_target, validation_input, validation_target,
                           device, nb_epochs, batch_size, print_step)

    print(f'\nTraining completed!')
    print(f'\nTesting started...')

    # Change mode to testing again. This time to do a final performance test.
    # The testing will be done on a unseen dataset
    # Previously model.eval() was called to test during model training

    model.eval()

    errors_in_testing = calculate_amount_errors_simple_cnn(model, test_input, test_target, batch_size)

    total_testing = test_input.size(0)
    error_rate = 100 * errors_in_testing / total_testing

    print(f'\nTesting Completed!')

    return error_rate


def evaluate_advanced_cnn(nb_pairs, batch_size, nb_epochs, print_step, hidden_layers, advanced_cnn):
    if torch.cuda.is_available():
        print("Using GPU!")
        device = torch.device('cuda')
    else:
        print("Using CPU!")
        device = torch.device('cpu')

    model = advanced_cnn(hidden_layers)

    # Print model characteristics
    print(model)

    # Load model onto device
    model.to(device)

    # Generate training and testing dataset
    train_input, train_target, train_classes, test_input, test_target, test_classes \
        = prologue.generate_pair_sets(nb_pairs)

    # Calculate data mean and standard deviation
    mean = train_input.mean()
    std = train_input.std()

    # Normalize the data (following dlc_practical_proloque.py example)
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Following https://stackoverflow.com/
    # /questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    # Split the training and testing into is not done. We keep 1000 testing pairs since there is enough data available
    # Split the training data into training and validation at 80%:20%
    # The two channels of each pair are split into individual two digit classes

    train_input = train_input.view(-1, 1, 14, 14)
    test_input = test_input.view(-1, 1, 14, 14)

    validation_input = train_input[1600:2000]
    validation_target = train_target[800:1000]

    train_input = train_input[0:1600]
    train_target = train_target[0:800]

    train_classes = train_classes.view(-1, 1)
    train_classes = train_classes[0:1600]
    train_classes = train_classes.reshape((-1,))

    # Train model
    model, training_loss, training_accuracy, validation_accuracy \
        = train_model_advanced_cnn(model, train_input, train_target, train_classes, validation_input, validation_target,
                                   device, nb_epochs, batch_size, print_step)

    print(f'\nTraining completed!')
    print(f'\nTesting started...')
    # Change mode to testing again. This time to do a final performance test.
    # The testing will be done on a unseen dataset
    # Previously model.eval() was called to test during model training

    model.eval()

    errors_in_testing = calculate_amount_errors_advanced_cnn(model, test_input, test_target, batch_size)
    total_testing = test_input.size(0)
    error_rate = 100 * errors_in_testing / (total_testing / 2)

    print(f'\nTesting Completed!')

    return error_rate
