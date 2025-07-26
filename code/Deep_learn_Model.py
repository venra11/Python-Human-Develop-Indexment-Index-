#!/usr/bin/env python3
"""
Oregon County Pattern Discovery

"""

import pandas as pd
import torch # torch = the main PyTorch library - does all the math for neural networks
import torch.nn as nn # torch.nn = neural network building blocks - has layers, activation functions, etc.

print("FINDING PATTERNS IN OREGON COUNTIES")
print("="*50)

def load_data():
    """Load our county data"""
    df = pd.read_csv(r"C:\Users\joshu\Downloads\HDI Modeling Data\oregon_counties_standardized.csv")
    # Just loading the data we made earlier
    
    counties = df['county'].tolist()
    X = torch.tensor(df.drop('county', axis=1).values, dtype=torch.float32)
    # Converting to numbers the neural network can use
    
    print(f"Got {len(counties)} counties with {X.shape[1]} indicators")
    return X, counties

class SimpleNetwork(nn.Module):
    """Dead simple neural network"""
    
    def __init__(self):
        super().__init__()
        self.squeeze = nn.Linear(14, 4)
        self.expand = nn.Linear(4, 14)
        # Takes 14 indicators, squeezes to 4 patterns, then tries to rebuild the 14
    
    def forward(self, x):
        patterns = torch.relu(self.squeeze(x))
        rebuilt = self.expand(patterns)
        return rebuilt
        # Squeeze data down then expand back up - if it works, the 4 patterns are good
    
    def get_patterns(self, x):
        return torch.relu(self.squeeze(x))
        # Just get the 4 patterns without rebuilding

def train_network(X):
    """Train the network to find patterns"""
    model = SimpleNetwork()
    optimizer = torch.optim.Adam(model.parameters())
    # Basic setup - Adam optimizer is reliable
    
    print("Training network to find 4 development patterns")
    
    for epoch in range(200):
        rebuilt = model(X)
        loss = torch.mean((rebuilt - X) ** 2)
        # Try to rebuild the data, measure how far off we are
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Standard training loop
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}, Error: {loss.item():.3f}")
    
    print("Training complete")
    return model

def analyze_results(model, X, counties):
    """See what patterns we found"""
    print(f"\nPATTERN ANALYSIS")
    print("="*20)
    
    patterns = model.get_patterns(X).detach().numpy()
    # Get the 4 patterns for each county
    
    pattern_names = ['Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4']
    # Keeping it simple for now
    
    for i, name in enumerate(pattern_names):
        scores = patterns[:, i]
        
        # Find top and bottom counties
        best = scores.argsort()[-3:][::-1]
        worst = scores.argsort()[:3]
        
        print(f"\n{name}:")
        print("  Top counties:")
        for idx in best:
            print(f"    {counties[idx]}: {scores[idx]:.2f}")
        print("  Bottom counties:")
        for idx in worst:
            print(f"    {counties[idx]}: {scores[idx]:.2f}")

def run_analysis():
    """Do everything"""
    X, counties = load_data()
    model = train_network(X)
    analyze_results(model, X, counties)
    
    print(f"\n" + "="*30)
    print("ANALYSIS COMPLETE")
    print("="*30)

if __name__ == "__main__":
    run_analysis()