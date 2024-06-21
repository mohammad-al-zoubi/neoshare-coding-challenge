import time
import torch
from torch import nn
from torch.optim import AdamW
from configuration import config
from torch.utils.data import DataLoader, Dataset
from utils import load_json_to_dict, calculate_f1_scores


class LinearClassifier(nn.Module):
    """
    A simple linear classifier with two linear layers and a softmax activation.
    It classifies embeddings of size `in_features` into `num_classes` classes.
    """

    def __init__(self, in_features, hidden_dim, num_classes=8):
        super(LinearClassifier, self).__init__()
        self.linear_1 = nn.Linear(in_features, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        out = self.softmax(out)
        return out


class TextDataset(Dataset):
    def __init__(self, path_to_data):
        self.data = load_json_to_dict(path_to_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        embeddings = torch.tensor(self.data[index]['embeddings'])
        label = self.data[index]['result'] - 1
        return embeddings, torch.tensor(label)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_train_data = r'data/embeddings_training.json'
    path_to_eval_data = r'data/embeddings_test.json'

    # Instantiate the model, dataset, and dataloader
    net = LinearClassifier(in_features=1024, hidden_dim=512, num_classes=8)
    train_dataset = TextDataset(path_to_train_data)
    eval_dataset = TextDataset(path_to_eval_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8)

    optimizer = AdamW(net.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = 0.0

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 99:  # print every 100 mini-batches
                # Evaluation
                net.eval()  # Set the model to evaluation mode
                eval_loss = 0.0
                total_labels = torch.empty(0, dtype=torch.long, device=device)
                total_outputs = torch.empty(0, dtype=torch.long, device=device)
                for data in eval_loader:
                    inputs, labels = data
                    outputs = net(inputs)
                    total_labels = torch.cat((total_labels, labels.to(device)))
                    total_outputs = torch.cat((total_outputs, torch.argmax(outputs, dim=1).to(device)))
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item() * inputs.size(0)
                eval_loss /= len(eval_dataset)
                macro_f1, micro_f1, _ = calculate_f1_scores(total_outputs, total_labels)
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    print('Saving best model...')
                    torch.save(net.state_dict(), 'embeddings_classifier/best_text_classifier.pth')
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}, eval_loss: {eval_loss:.3f}, macro_f1: {macro_f1:.3f}, micro_f1: {micro_f1:.3f}')
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), 'embeddings_classifier/text_classifier.pth')
    print('Finished Training')
