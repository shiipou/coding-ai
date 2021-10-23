import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import glob
from dataset import CodeDataset

class NeuralNetwork(nn.Module):
    # director where the data checkpoints is located
    CHECKPOINTS_DIR = './checkpoints'

    # flatten the input
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.device = None

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # Initialize the model
    def init(self, trainset, testset, device = 'cpu'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        self.device = device
        self.to(device)

        # Create a dataloader for the training and validation sets.
        train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

        return train_dataloader, test_dataloader

    def save(self, filename):
        """
        Save the model to the requested directory
        """
        torch.save(self.state_dict(), filename)

    def generate_filename(self, epoch, idx):
        """
        Find the filename to save the current model
        """
        if not os.path.exists(NeuralNetwork.CHECKPOINTS_DIR):
            os.makedirs(NeuralNetwork.CHECKPOINTS_DIR)
        return os.path.join(NeuralNetwork.CHECKPOINTS_DIR, '{}-{}.pt'.format(epoch, idx))

    def load(self):
        """
        Load the model from the checkpoint directory if it exists and return the epoch and idx to resume training
        """
        if not os.path.exists(NeuralNetwork.CHECKPOINTS_DIR):
            return 0, 0

        # List all the files in the checkpoints directory
        file_list = glob.glob('{}/*'.format(NeuralNetwork.CHECKPOINTS_DIR))
        checkpoint_path = ''
        _epoch, _i = 0, 0

        # Find the latest checkpoint if the directory is not empty
        if len(file_list) > 0:
            # Get the lastest checkpoint file (Using the last modified time of files)
            checkpoint_path = max(file_list, key=os.path.getctime)

            # Get the epoch and idx from the file name
            _base_name = os.path.basename(checkpoint_path)
            _wihout_ext = os.path.splitext(_base_name)[0]
            _tmp_args = _wihout_ext.split('-')
            _epoch, _i = int(_tmp_args[0]), int(_tmp_args[1])

            # Load the model from the checkpoint
            self.load_state_dict(torch.load(checkpoint_path))
            print('Loaded Epoch {}s i {} checkpoint from {}'.format(_epoch, _i, checkpoint_path))
            _i = _i + 1

        return _epoch, _i

    def train(self, train_loader, epochs, optimizer, loss_fn):
        loaded_epoch, loaded_id = self.load()
        for epoch in range(loaded_epoch, epochs):
            for batch_idx, (data, target) in enumerate(train_loader, start=loaded_id):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    self.save(self.generate_filename(epoch, batch_idx))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, test_loader, device, loss_fn):
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = self.forward(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
