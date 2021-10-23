import os
import torch

from torch.utils.data.dataset import random_split
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.utils.dummy_pt_objects import GPT2LMHeadModel
from neural_network import NeuralNetwork
from dataset import CodeDataset


def main():
    # Initialization

    # Set the random seed for reproducibility.
    torch.manual_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    CodeDataset.init(tokenizer, model)

    data_files = []
    for root, dirs, files in os.walk('./data'):
        for file in files:
            if file.endswith('.py'):
                data_files.append(os.path.join(root, file))

    data = data_files
    'Data len: {}'.format(len(data))

    dataset = CodeDataset(data)

    neuralNetwork = NeuralNetwork()

    # Split the dataset into training and validation sets.
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(train_dataset)
    train_loader, test_loader = neuralNetwork.init(train_dataset, test_dataset)
    print(train_loader)

    # Create Optimizer
    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=0.001)

    # Create Loss Function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    neuralNetwork.train(train_loader, 100, optimizer, criterion)

    neuralNetwork.test()

    neuralNetwork.save()

if __name__ == '__main__':
    main()