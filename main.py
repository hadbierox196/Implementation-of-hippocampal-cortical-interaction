"""
Assignment 4: Complementary Learning Systems
Hippocampal-Cortical Memory Consolidation Model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("="*70)
print("COMPLEMENTARY LEARNING SYSTEMS")
print("="*70)

#%% NEURAL NETWORK MODULES

class NeuralModule:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, name="Module"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.name = name
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
    def train_step(self, X, y):
        output = self.forward(X)
        self.backward(X, y, output)
        loss = np.mean((output - y) ** 2)
        return loss
    
    def predict(self, X):
        return self.forward(X)
    
    def compute_loss(self, X, y):
        output = self.forward(X)
        return np.mean((output - y) ** 2)


class HippocampalModule(NeuralModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        super().__init__(input_size, hidden_size, output_size, learning_rate, name="Hippocampus")
        self.episodic_memory = []
        self.max_memory_size = 100
        
    def store_episode(self, X, y):
        self.episodic_memory.append((X.copy(), y.copy()))
        if len(self.episodic_memory) > self.max_memory_size:
            self.episodic_memory.pop(0)
    
    def sample_memories(self, n_samples):
        if len(self.episodic_memory) == 0:
            return None, None
        
        n_samples = min(n_samples, len(self.episodic_memory))
        indices = np.random.choice(len(self.episodic_memory), n_samples, replace=False)
        
        X_replay = []
        y_replay = []
        for idx in indices:
            X, y = self.episodic_memory[idx]
            X_replay.append(X)
            y_replay.append(y)
        
        return np.vstack(X_replay), np.vstack(y_replay)


class CorticalModule(NeuralModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        super().__init__(input_size, hidden_size, output_size, learning_rate, name="Cortex")


class ComplementaryLearningSystem:
    def __init__(self, input_size, hidden_size, output_size, 
                 hippo_lr=0.1, cortex_lr=0.01):
        self.hippocampus = HippocampalModule(input_size, hidden_size, output_size, hippo_lr)
        self.cortex = CorticalModule(input_size, hidden_size, output_size, cortex_lr)
        
        self.history = {
            'hippo_loss': [],
            'cortex_loss': [],
            'combined_loss': [],
            'replay_count': 0
        }
    
    def train_episode(self, X, y, store_in_hippo=True):
        hippo_loss = self.hippocampus.train_step(X, y)
        cortex_loss = self.cortex.train_step(X, y)
        
        if store_in_hippo:
            self.hippocampus.store_episode(X, y)
        
        return hippo_loss, cortex_loss
    
    def replay_consolidation(self, n_replay_samples=10, n_iterations=5):
        if len(self.hippocampus.episodic_memory) == 0:
            return
        
        for _ in range(n_iterations):
            X_replay, y_replay = self.hippocampus.sample_memories(n_replay_samples)
            if X_replay is not None:
                self.cortex.train_step(X_replay, y_replay)
                self.history['replay_count'] += 1
    
    def predict(self, X, use_module='both'):
        if use_module == 'hippocampus':
            return self.hippocampus.predict(X)
        elif use_module == 'cortex':
            return self.cortex.predict(X)
        else:
            hippo_pred = self.hippocampus.predict(X)
            cortex_pred = self.cortex.predict(X)
            return 0.5 * hippo_pred + 0.5 * cortex_pred


#%% GENERATE EPISODIC MEMORIES

def generate_episodic_memories(n_episodes=50, input_size=20, output_size=10, 
                               similarity=0.0):
    episodes = []
    
    for i in range(n_episodes):
        if i == 0 or similarity == 0:
            X = np.random.randn(1, input_size) * 0.5
        else:
            X = similarity * episodes[-1][0] + (1 - similarity) * np.random.randn(1, input_size) * 0.5
        
        y = np.zeros((1, output_size))
        y[0, i % output_size] = 1
        
        episodes.append((X, y))
    
    return episodes


#%% EXPERIMENT 1: BASIC LEARNING

print("\n" + "="*70)
print("EXPERIMENT 1: Basic Learning Curves")
print("="*70)

def experiment_basic_learning():
    input_size = 20
    hidden_size = 50
    output_size = 10
    n_episodes = 30
    n_epochs = 10
    
    episodes = generate_episodic_memories(n_episodes, input_size, output_size)
    cls = ComplementaryLearningSystem(input_size, hidden_size, output_size,
                                     hippo_lr=0.1, cortex_lr=0.01)
    
    hippo_losses = []
    cortex_losses = []
    
    print(f"\nTraining on {n_episodes} episodes, {n_epochs} epochs each...")
    
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        epoch_hippo_loss = []
        epoch_cortex_loss = []
        
        for X, y in episodes:
            hippo_loss, cortex_loss = cls.train_episode(X, y)
            epoch_hippo_loss.append(hippo_loss)
            epoch_cortex_loss.append(cortex_loss)
        
        hippo_losses.append(np.mean(epoch_hippo_loss))
        cortex_losses.append(np.mean(epoch_cortex_loss))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, n_epochs + 1), hippo_losses, 
            marker='o', linewidth=2, markersize=8,
            label='Hippocampus (LR=0.1)', color='red')
    ax.plot(range(1, n_epochs + 1), cortex_losses, 
            marker='s', linewidth=2, markersize=8,
            label='Cortex (LR=0.01)', color='blue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Learning Curves: Hippocampus vs Cortex',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('cls_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Final hippocampus loss: {hippo_losses[-1]:.4f}")
    print(f"✓ Final cortex loss: {cortex_losses[-1]:.4f}")
    
    return cls, episodes

cls, episodes = experiment_basic_learning()


#%% EXPERIMENT 2: CONSOLIDATION AND FORGETTING

print("\n" + "="*70)
print("EXPERIMENT 2: Memory Consolidation Through Replay")
print("="*70)

def experiment_consolidation():
    input_size = 20
    hidden_size = 50
    output_size = 10
    n_old_episodes = 20
    n_new_episodes = 20
    n_days = 30
    
    print("\nGenerating episodes...")
    old_episodes = generate_episodic_memories(n_old_episodes, input_size, output_size)
    new_episodes = generate_episodic_memories(n_new_episodes, input_size, output_size)
    
    # WITH replay
    print("\nCondition 1: WITH replay...")
    cls_with_replay = ComplementaryLearningSystem(input_size, hidden_size, output_size)
    
    old_memory_with_replay = []
    new_memory_with_replay = []
    
    for day in tqdm(range(10), desc="Learning old memories"):
        for X, y in old_episodes:
            cls_with_replay.train_episode(X, y)
        cls_with_replay.replay_consolidation(n_replay_samples=10, n_iterations=5)
    
    for day in tqdm(range(10, n_days), desc="Learning new + consolidation"):
        for X, y in new_episodes:
            cls_with_replay.train_episode(X, y)
        
        cls_with_replay.replay_consolidation(n_replay_samples=15, n_iterations=10)
        
        old_loss = np.mean([cls_with_replay.cortex.compute_loss(X, y) for X, y in old_episodes])
        new_loss = np.mean([cls_with_replay.cortex.compute_loss(X, y) for X, y in new_episodes])
        old_memory_with_replay.append(old_loss)
        new_memory_with_replay.append(new_loss)
    
    # WITHOUT replay
    print("\nCondition 2: WITHOUT replay...")
    cls_without_replay = ComplementaryLearningSystem(input_size, hidden_size, output_size)
    
    old_memory_without_replay = []
    new_memory_without_replay = []
    
    for day in range(10):
        for X, y in old_episodes:
            cls_without_replay.train_episode(X, y)
    
    for day in tqdm(range(10, n_days), desc="Learning new WITHOUT consolidation"):
        for X, y in new_episodes:
            cls_without_replay.train_episode(X, y)
        
        old_loss = np.mean([cls_without_replay.cortex.compute_loss(X, y) for X, y in old_episodes])
        new_loss = np.mean([cls_without_replay.cortex.compute_loss(X, y) for X, y in new_episodes])
        old_memory_without_replay.append(old_loss)
        new_memory_without_replay.append(new_loss)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    days = range(11, n_days + 1)
    
    ax1.plot(days, old_memory_with_replay, 
            marker='o', linewidth=2, markersize=6,
            label='With replay', color='green')
    ax1.plot(days, old_memory_without_replay, 
            marker='s', linewidth=2, markersize=6,
            label='Without replay', color='red')
    
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Old Memory Loss (MSE)', fontsize=12)
    ax1.set_title('Forgetting Curve: Old Memories',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    
    ax2.plot(days, new_memory_with_replay, 
            marker='o', linewidth=2, markersize=6,
            label='With replay', color='green')
    ax2.plot(days, new_memory_without_replay, 
            marker='s', linewidth=2, markersize=6,
            label='Without replay', color='red')
    
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('New Memory Loss (MSE)', fontsize=12)
    ax2.set_title('Learning Curve: New Memories',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cls_consolidation_forgetting.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Old memory WITH replay: {old_memory_with_replay[-1]:.4f}")
    print(f"✓ Old memory WITHOUT replay: {old_memory_without_replay[-1]:.4f}")

experiment_consolidation()


#%% EXPERIMENT 3: CATASTROPHIC INTERFERENCE

print("\n" + "="*70)
print("EXPERIMENT 3: Catastrophic Interference")
print("="*70)

def experiment_interference():
    input_size = 20
    hidden_size = 50
    output_size = 10
    n_episodes = 15
    
    similarity_levels = [0.0, 0.3, 0.6, 0.9]
    results = {
        'similarity': similarity_levels,
        'hippo_interference': [],
        'cortex_interference': [],
        'hippo_with_replay': [],
        'cortex_with_replay': []
    }
    
    print("\nTesting interference at different similarity levels...")
    
    for similarity in tqdm(similarity_levels, desc="Similarity"):
        old_episodes = generate_episodic_memories(n_episodes, input_size, output_size, 
                                                  similarity=0.0)
        new_episodes = generate_episodic_memories(n_episodes, input_size, output_size, 
                                                  similarity=similarity)
        
        # WITHOUT replay
        cls_no_replay = ComplementaryLearningSystem(input_size, hidden_size, output_size)
        
        for _ in range(5):
            for X, y in old_episodes:
                cls_no_replay.train_episode(X, y)
        
        old_loss_hippo_before = np.mean([cls_no_replay.hippocampus.compute_loss(X, y) 
                                         for X, y in old_episodes])
        old_loss_cortex_before = np.mean([cls_no_replay.cortex.compute_loss(X, y) 
                                          for X, y in old_episodes])
        
        for _ in range(5):
            for X, y in new_episodes:
                cls_no_replay.train_episode(X, y, store_in_hippo=False)
        
        old_loss_hippo_after = np.mean([cls_no_replay.hippocampus.compute_loss(X, y) 
                                        for X, y in old_episodes])
        old_loss_cortex_after = np.mean([cls_no_replay.cortex.compute_loss(X, y) 
                                         for X, y in old_episodes])
        
        results['hippo_interference'].append(old_loss_hippo_after - old_loss_hippo_before)
        results['cortex_interference'].append(old_loss_cortex_after - old_loss_cortex_before)
        
        # WITH replay
        cls_with_replay = ComplementaryLearningSystem(input_size, hidden_size, output_size)
        
        for _ in range(5):
            for X, y in old_episodes:
                cls_with_replay.train_episode(X, y)
        
        for _ in range(5):
            for X, y in new_episodes:
                cls_with_replay.train_episode(X, y, store_in_hippo=False)
            cls_with_replay.replay_consolidation(n_replay_samples=10, n_iterations=3)
        
        old_loss_hippo_replay = np.mean([cls_with_replay.hippocampus.compute_loss(X, y) 
                                         for X, y in old_episodes])
        old_loss_cortex_replay = np.mean([cls_with_replay.cortex.compute_loss(X, y) 
                                          for X, y in old_episodes])
        
        results['hippo_with_replay'].append(old_loss_hippo_replay - old_loss_hippo_before)
        results['cortex_with_replay'].append(old_loss_cortex_replay - old_loss_cortex_before)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['similarity'], results['hippo_interference'], 
            marker='o', linewidth=2, markersize=8,
            label='Without replay', color='red')
    ax1.plot(results['similarity'], results['hippo_with_replay'], 
            marker='s', linewidth=2, markersize=8,
            label='With replay', color='green')
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Memory Similarity', fontsize=12)
    ax1.set_ylabel('Interference (Δ Loss)', fontsize=12)
    ax1.set_title('Hippocampal Interference',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results['similarity'], results['cortex_interference'], 
            marker='o', linewidth=2, markersize=8,
            label='Without replay', color='red')
    ax2.plot(results['similarity'], results['cortex_with_replay'], 
            marker='s', linewidth=2, markersize=8,
            label='With replay', color='green')
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Memory Similarity', fontsize=12)
    ax2.set_ylabel('Interference (Δ Loss)', fontsize=12)
    ax2.set_title('Cortical Interference',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cls_interference.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Max cortical interference (no replay): {max(results['cortex_interference']):.4f}")
    print(f"✓ Max cortical interference (with replay): {max(results['cortex_with_replay']):.4f}")

experiment_interference()


#%% EXPERIMENT 4: HIPPOCAMPAL LESION

print("\n" + "="*70)
print("EXPERIMENT 4: Hippocampal Lesion Simulation")
print("="*70)

def experiment_hippocampal_lesion():
    input_size = 20
    hidden_size = 50
    output_size = 10
    n_episodes = 20
    n_days = 40
    
    episodes = generate_episodic_memories(n_episodes, input_size, output_size)
    
    lesion_days = [1, 15, 30]
    results = {day: {'cortex_loss': [], 'hippo_loss': []} for day in lesion_days}
    results['control'] = {'cortex_loss': [], 'hippo_loss': []}
    
    print("\nSimulating hippocampal lesions...")
    
    # Control
    print("\nControl: No lesion")
    cls_control = ComplementaryLearningSystem(input_size, hidden_size, output_size)
    
    for day in tqdm(range(1, n_days + 1), desc="Control"):
        for X, y in episodes:
            cls_control.train_episode(X, y)
        cls_control.replay_consolidation(n_replay_samples=10, n_iterations=5)
        
        cortex_loss = np.mean([cls_control.cortex.compute_loss(X, y) for X, y in episodes])
        hippo_loss = np.mean([cls_control.hippocampus.compute_loss(X, y) for X, y in episodes])
        results['control']['cortex_loss'].append(cortex_loss)
        results['control']['hippo_loss'].append(hippo_loss)
    
    # Lesion conditions
    for lesion_day in lesion_days:
        print(f"\nLesion at Day {lesion_day}")
        cls_lesion = ComplementaryLearningSystem(input_size, hidden_size, output_size)
        
        for day in tqdm(range(1, n_days + 1), desc=f"Lesion Day {lesion_day}"):
            if day < lesion_day:
                for X, y in episodes:
                    cls_lesion.train_episode(X, y)
                cls_lesion.replay_consolidation(n_replay_samples=10, n_iterations=5)
            elif day == lesion_day:
                cls_lesion.hippocampus = HippocampalModule(input_size, hidden_size, 
                                                          output_size, learning_rate=0.1)
                print(f"  ** LESION at Day {lesion_day} **")
            else:
                for X, y in episodes:
                    cls_lesion.cortex.train_step(X, y)
            
            cortex_loss = np.mean([cls_lesion.cortex.compute_loss(X, y) for X, y in episodes])
            hippo_loss = np.mean([cls_lesion.hippocampus.compute_loss(X, y) for X, y in episodes])
            results[lesion_day]['cortex_loss'].append(cortex_loss)
            results[lesion_day]['hippo_loss'].append(hippo_loss)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    days = range(1, n_days + 1)
    colors = ['blue', 'orange', 'red', 'green']
    
    ax1.plot(days, results['control']['cortex_loss'], 
            linewidth=2.5, label='Control', color=colors[0])
    
    for i, lesion_day in enumerate(lesion_days):
        ax1.plot(days, results[lesion_day]['cortex_loss'], 
                linewidth=2, label=f'Lesion Day {lesion_day}', 
                color=colors[i+1], linestyle='--')
        ax1.axvline(x=lesion_day, color=colors[i+1], linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Cortical Memory Loss', fontsize=12)
    ax1.set_title('Effect of Hippocampal Lesion on Cortical Memory',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(days, results['control']['hippo_loss'], 
            linewidth=2.5, label='Control', color=colors[0])
    
    for i, lesion_day in enumerate(lesion_days):
        ax2.plot(days, results[lesion_day]['hippo_loss'], 
                linewidth=2, label=f'Lesion Day {lesion_day}', 
                color=colors[i+1], linestyle='--')
        ax2.axvline(x=lesion_day, color=colors[i+1], linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Hippocampal Memory Loss', fontsize=12)
    ax2.set_title('Effect of Hippocampal Lesion on Hippocampal Memory',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('cls_hippocampal_lesion.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Lesion simulation complete")
    print("\nFinal cortical memory (Day 40):")
    print(f"  Control: {results['control']['cortex_loss'][-1]:.4f}")
    for lesion_day in lesion_days:
        print(f"  Lesion Day {lesion_day}: {results[lesion_day]['cortex_loss'][-1]:.4f}")

experiment_hippocampal_lesion()


#%% EXPERIMENT 5: COMPREHENSIVE ANALYSIS

print("\n" + "="*70)
print("EXPERIMENT 5: Comprehensive System Analysis")
print("="*70)

def comprehensive_analysis():
    input_size = 20
    hidden_size = 50
    output_size = 10
    n_episodes = 25
    n_days = 30
    
    episodes = generate_episodic_memories(n_episodes, input_size, output_size)
    cls = ComplementaryLearningSystem(input_size, hidden_size, output_size)
    
    metrics = {
        'day': [],
        'hippo_loss': [],
        'cortex_loss': [],
        'hippo_memory_size': [],
        'replay_count': []
    }
    
    print("\nRunning comprehensive analysis...")
    
    for day in tqdm(range(1, n_days + 1), desc="Training days"):
        for X, y in episodes:
            cls.train_episode(X, y)
        
        cls.replay_consolidation(n_replay_samples=15, n_iterations=10)
        
        metrics['day'].append(day)
        metrics['hippo_loss'].append(
            np.mean([cls.hippocampus.compute_loss(X, y) for X, y in episodes]))
        metrics['cortex_loss'].append(
            np.mean([cls.cortex.compute_loss(X, y) for X, y in episodes]))
        metrics['hippo_memory_size'].append(len(cls.hippocampus.episodic_memory))
        metrics['replay_count'].append(cls.history['replay_count'])
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(metrics['day'], metrics['hippo_loss'], 
            marker='o', linewidth=2, markersize=4,
            label='Hippocampus', color='red', alpha=0.7)
    ax1.plot(metrics['day'], metrics['cortex_loss'], 
            marker='s', linewidth=2, markersize=4,
            label='Cortex', color='blue', alpha=0.7)
    ax1.set_xlabel('Day', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Learning Dynamics Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(metrics['day'], metrics['hippo_memory_size'], 
            linewidth=2, color='purple')
    ax2.set_xlabel('Day', fontsize=11)
    ax2.set_ylabel('Episodic Memory Size', fontsize=11)
    ax2.set_title('Hippocampal Memory Buffer', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Max capacity')
    ax2.legend(fontsize=9)
    
    ax3 = fig.add_subplot(gs[1, 1])
    replay_per_day = np.diff([0] + metrics['replay_count'])
    ax3.bar(metrics['day'], replay_per_day, color='orange', alpha=0.7)
    ax3.set_xlabel('Day', fontsize=11)
    ax3.set_ylabel('Replay Events', fontsize=11)
    ax3.set_title('Nightly Consolidation Activity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(gs[2, 0])
    improvement = np.array(metrics['hippo_loss']) / np.array(metrics['cortex_loss'])
    ax4.plot(metrics['day'], improvement, linewidth=2, color='green')
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Day', fontsize=11)
    ax4.set_ylabel('Hippo Loss / Cortex Loss', fontsize=11)
    ax4.set_title('Relative Performance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = f"""
COMPLEMENTARY LEARNING SYSTEMS

System Parameters:
• Hippocampal LR: {cls.hippocampus.lr:.3f}
• Cortical LR: {cls.cortex.lr:.3f}
• LR Ratio: {cls.hippocampus.lr/cls.cortex.lr:.1f}x

Final Performance (Day {n_days}):
• Hippo Loss: {metrics['hippo_loss'][-1]:.4f}
• Cortex Loss: {metrics['cortex_loss'][-1]:.4f}

Consolidation:
• Total Replays: {metrics['replay_count'][-1]:,}
• Avg/Night: {metrics['replay_count'][-1]/n_days:.1f}
• Memory Buffer: {metrics['hippo_memory_size'][-1]}/100
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Complementary Learning Systems: Comprehensive Analysis',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('cls_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Comprehensive analysis complete")

comprehensive_analysis()

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE")
print("="*70)
