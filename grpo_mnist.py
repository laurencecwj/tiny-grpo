import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import wandb

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
err_prob = 0.1

# circle, corner, unclosed_void
value_mapper = {
    1: [0, 0, 0],
    2: [0, 0, 1],
    3: [0, 0, 2],
    4: [0, 3, 0],
    5: [0, 1, 1],
    6: [1, 0, 0],
    7: [0, 1, 0],
    8: [2, 0, 0],
    9: [1, 0, 0],
    0: [1, 0, 0],
}
map_circle_only = True
use_partial_observed = True

@dataclass
class RewardPolicy:
    origin: int
    mistake: List[int]
    errprob: List[float]

    def _setup(self):
        _p = [1 - sum(self.errprob)]
        _p.extend(self.errprob)
        self.probs = torch.tensor(_p)

    def sample(self):
        _idx = torch.multinomial(self.probs, num_samples=1)
        if _idx == 0:
            return self.origin
        else:
            return self.mistake[_idx - 1]


# Define a simple neural network for MNIST with GRPO
class MNIST_GRPO(nn.Module):
    def __init__(self):
        super(MNIST_GRPO, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax (will be handled in loss)
        return x


_mistake_policies = {
    1: RewardPolicy(1, [7], [err_prob]),
    4: RewardPolicy(4, [9], [err_prob]),
    7: RewardPolicy(7, [1], [err_prob]),
    6: RewardPolicy(6, [0], [err_prob]),
    0: RewardPolicy(0, [6], [err_prob]),
}

mistake_policies = {}

def setup_mistake_policies():
    global _mistake_policies, mistake_policies
    if _mistake_policies:
        for _digit, _policy in _mistake_policies.items():
            _policy._setup()
            mistake_policies[_digit] = _policy

def fetch_labels(labels):
    # labels: torch.Tensor 
    # labels.shape: torch.Size([batch_size])
    _result = []
    for _label in labels:
        if _label.item() in mistake_policies:
            _policy = mistake_policies[_label.item()]
            _result.append(_policy.sample())
        else:
            _result.append(_label)
    return torch.tensor(_result)

def get_confusion(predictions, labels):
    correct = [0.] * 10
    confusion = defaultdict(int)
    total = 0.
    for v1, v2 in zip(predictions, labels):
        _v1 = v1 
        _v2 = v2
        if _v1 == _v2:
            correct[_v1] += 1
        else:
            confusion[(_v2, _v1)] += 1
        total += 1
    return correct, total, confusion

def partial_reward(grp_vals, dest):
    _results = []
    for _v in grp_vals:
        _reward, _ = weak_rewards(_v, dest[0])
        _results.append(_reward)
    return _results

def weak_rewards(val, label):
    mapped_vals_0 = value_mapper[val]
    mapped_vals_1 = value_mapper[label]
    if map_circle_only:
        _rew = 1. if mapped_vals_0[0] == mapped_vals_1[0] else 0.
    else:
        _rew = torch.mean((torch.tensor(mapped_vals_0) == torch.tensor(mapped_vals_1)).float())
    return _rew, label

def partial_reward_eval(predictions, labels, reward_map_fn = None):
    correct, total, confusion = get_confusion(predictions, labels)
    if reward_map_fn:
        # return (predictions == labels).sum().item()
        rewards = defaultdict(list)
        for _v1, _v2 in zip(predictions, labels):
            _reward, _label = reward_map_fn(_v1, _v2)
            rewards[_label].append(_reward)
        _outlog = {}
        for _label, _rewards in rewards.items():
            _val = torch.mean(torch.tensor(_rewards)).item()
            _outlog[_label] = _val
        wandb.log(_outlog)
        
    return correct, total, confusion

def calc_rewards(sampled_outputs, labels, partial_observed = False):
    if not partial_observed:
        rewards = (sampled_outputs == labels.view(-1, 1)).float()  # Reward = 1 if correct, else 0
        return rewards
    # labels: torch.Tensor 
    # labels.shape: torch.Size([batch_size])

    _results = []
    for v1, v2 in zip(sampled_outputs.tolist(), labels.view(-1, 1).tolist()):
        _results.append(partial_reward(v1, v2))
    return torch.tensor(_results)

# GRPO Training Function with loss tracking
def train_grpo(model, train_loader, optimizer, epochs=5, num_samples=6, config={}):
    model.train()
    loss_history = []
    _mean_rewards = [] 

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Step 1: Sample multiple predictions
            sampled_outputs = []
            logits = model(images)

            for _ in range(num_samples):
                sampled_outputs.append(torch.multinomial(F.softmax(logits, dim=1), num_samples=1))

            sampled_outputs = torch.cat(sampled_outputs, dim=1)  # Shape: [batch, num_samples]

            # Step 2: Compute Rewards
            labels = fetch_labels(labels)
            labels = labels.to(device)
            # rewards = (sampled_outputs == labels.view(-1, 1)).float()  # Reward = 1 if correct, else 0
            rewards = calc_rewards(sampled_outputs, labels, config['use_partial_observed'])
            rewards = rewards.to(device)

            # Step 3: Compute Advantage
            mean_reward = rewards.mean(dim=1, keepdim=True)
            advantages = rewards - mean_reward  # Compute relative advantage
            _mean_rewards.append(mean_reward.detach().cpu())

            # Step 4: Compute Policy Loss
            log_probs = torch.log_softmax(logits, dim=1)
            selected_log_probs = log_probs.gather(1, sampled_outputs)  # Get log probabilities of sampled outputs
            loss = (-selected_log_probs * advantages).mean()  # Policy Gradient Loss

            # Step 5: Backpropagation and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        loss_history.append(epoch_loss)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        wandb.log({"epoch": epoch+1, "loss": epoch_loss, "time": epoch_time, "mean_reward": torch.mean(torch.concat(_mean_rewards)).item()})

    # save figure
    # Plot Training Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b', label="GRPO Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("GRPO Training Loss Curve")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('loss.png')    

    return loss_history, epoch, epoch_loss

def save_model(model, optimizer, epoch, loss, model_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)

def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']    
    return model, optimizer, epoch, loss

# Function to evaluate the trained model on the MNIST test set
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)  # Get the class with the highest probability

            # correct += (predictions == labels).sum().item()
            _correct, _total, _ = partial_reward_eval(predictions, labels, weak_rewards if use_partial_observed else None)
            correct += sum(_correct)
            total += _total #labels.size(0)

    accuracy = 100 * correct / total
    wandb.log({"accuracy": accuracy, "correct": correct, "total": total})
    return accuracy

def validation(model, transform, config):
    # Load MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Evaluate the trained GRPO model
    grpo_accuracy = evaluate_model(model, test_loader)

    print(f"GRPO Test Accuracy: {grpo_accuracy:.2f}%")

    # Return accuracy result
    # grpo_accuracy

def main(args):
    global map_circle_only, use_partial_observed
    config = {
        'batch_size': 64, 
        'group_number': 32, 
        'err_prob': err_prob,
        'lr': 0.001,
        'epoch': args.epoch, 
        'use_partial_observed': False,
        'map_circle_only': True,
        'tag': args.tag,
        'load': args.load,
        'model_path': f'./grpo_mnist_{args.tag}.pth',
    }
    map_circle_only = config['map_circle_only']
    use_partial_observed = config['use_partial_observed']
    wandb.login()
    wandb.init(
        project="grpo_mnist",
        config=config,
    )
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    setup_mistake_policies()

    # Initialize model, optimizer
    model = MNIST_GRPO()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    if args.load:
        model, optimizer, epoch, loss = load_model(model, optimizer, args.load)
    model.to(device)

    # Train the model using GRPO and collect loss history
    _, epoch, loss = train_grpo(model, train_loader, optimizer, epochs=config['epoch'], num_samples=config['group_number'], config=config)
    save_model(model, optimizer, epoch, loss, config['model_path'])

    model, optimizer, epoch, loss = load_model(model, optimizer, config['model_path'])
    validation(model, transform, config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="grpo mnist experiment")
    parser.add_argument("--load", type=str, help="load pretrained model")
    parser.add_argument("--tag", type=str, help="tag or name of this experiment")
    parser.add_argument("--epoch", type=int, default=30, help="training epoch of this experiment")
    args = parser.parse_args()
    main(args)