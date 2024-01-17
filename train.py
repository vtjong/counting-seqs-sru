import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import CountingDataset
from models.counting import CountingModel
import wandb

def load_data(count_len, batch_size):
    dataset = CountingDataset(count_len)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, num_epochs, criterion, optimizer, wandb_run):
    for epoch in range(num_epochs):
        for batch in train_loader:
            input = batch['data'].permute(1, 0, 2)
            target = batch['label'].permute(1, 0, 2)
            
            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        wandb_run.log({"Train Loss": loss.item()})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, test_loader, criterion, wandb_run):
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            input = batch['data'].permute(1, 0, 2)
            target = batch['label'].permute(1, 0, 2)
            output = model(input)
            loss = criterion(output, target)
            accuracy = (1 - abs(output.item() - target.item())/target.item()) * 100
            wandb_run.log({"Test Loss": loss.item()})
            wandb_run.log({"Test Accuracy": accuracy})

            print("input", input)
            print("target", target.item())
            print("output", round(output.item()))

if __name__ == '__main__':
    wandb.init(project='counting-model')
    parser = argparse.ArgumentParser(description='SRU Counting Model')
    parser.add_argument('--input_size', type=int, default=3, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=9, help='Hidden size')
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--count_len', type=int, default=100, help='Length of the counting sequence')
    args = parser.parse_args()

    model = CountingModel(args.input_size, args.hidden_size, args.output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, test_loader = load_data(args.count_len, args.batch_size)

    train_model(model, train_loader, args.num_epochs, criterion, optimizer, wandb.run)
    evaluate_model(model, test_loader, criterion, wandb.run)

    wandb.save("model.pth")