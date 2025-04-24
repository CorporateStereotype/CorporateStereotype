import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# 🎯 FFZ Stabilizer for Gradient Scaling
class FFZStabilizer:
    epsilon = 1e-8  # Prevent division by zero

    @staticmethod
    def stabilize_gradient(gradient):
        """Stabilize gradients to prevent runaway values."""
        return gradient / (np.sqrt(np.abs(gradient)) + FFZStabilizer.epsilon)

# 🎯 Quantum Synchronizer with Phase-Enhanced Scaling
class QuantumSynchronizer:
    @staticmethod
    def entangle_qubits(num_qubits):
        """Generate entangled qubits for synchronization."""
        circuit = QuantumCircuit(num_qubits)
        circuit.h(0)  # Apply Hadamard to create superposition
        for i in range(1, num_qubits):
            circuit.cx(0, i)  # Create entanglement

        statevector = Statevector.from_instruction(circuit)
        return statevector.data  # Return the quantum state as an array

    @staticmethod
    def synchronize_gradients(classical_gradients):
        """Synchronize gradients using quantum metadata with phase shifts."""
        quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
        phase_adjusted_gradients = [
            classical_gradients[i] * np.abs(quantum_data[i]) * np.exp(1j * np.angle(quantum_data[i]))
            for i in range(len(classical_gradients))
        ]
        print(f"Synchronized Gradients: {phase_adjusted_gradients}")
        return phase_adjusted_gradients

# 🎯 Agent Class
class Agent:
    def __init__(self, name):
        self.name = name
        self.weights = np.random.randn(10)  # Initialize weights
        self.weights_history = []  # Track weight history for visualization
        self.gradient_history = []  # Track gradients for stability visualization

    def compute_gradient(self):
        """Simulate gradient computation with variability."""
        return np.random.randn(10) * np.random.uniform(0.5, 1.5)

    def apply_gradient(self, gradient):
        """Apply computed gradient to weights."""
        self.weights -= 0.01 * gradient  # Apply gradient update
        self.weights_history.append(self.weights.copy())  # Track weights
        self.gradient_history.append(np.linalg.norm(gradient))  # Track gradient magnitude
        print(f"{self.name} updated weights: {self.weights}")

# 🎯 Hybrid Controller (Federated Aggregation + Quantum Sync)
class HybridController:
    def __init__(self, agents, use_quantum=True):
        self.agents = agents
        self.use_quantum = use_quantum
        self.agent_gradients = []

    def synchronize_training(self):
        """Synchronize gradients across all agents."""
        self.agent_gradients.clear()
        for agent in self.agents:
            gradient = FFZStabilizer.stabilize_gradient(agent.compute_gradient())
            self.agent_gradients.append(gradient)

        if self.use_quantum:
            self.agent_gradients = QuantumSynchronizer.synchronize_gradients(self.agent_gradients)

    def optimize(self):
        """Optimize weights for all agents."""
        for agent, gradient in zip(self.agents, self.agent_gradients):
            agent.apply_gradient(gradient)

# 🎯 Visualization: Weight Evolution Over Rounds
def plot_weight_changes(agents, rounds):
    """Improved weight visualization."""
    plt.figure(figsize=(12, 6))

    # Assign colors per client
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))

    for agent_idx, agent in enumerate(agents):
        weights = np.array(agent.weights_history)
        for i in range(weights.shape[1]):  # Iterate over each weight
            plt.plot(range(rounds), weights[:, i], color=colors[agent_idx], alpha=0.6, label=f"{agent.name}" if i == 0 else "")

    plt.title("Weight Evolution Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Weight Values")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# 🎯 Performance Metrics: Track Gradient Magnitude Over Time
def plot_gradient_magnitude(agents, rounds):
    """Visualize gradient stability over rounds."""
    plt.figure(figsize=(10, 5))

    for agent in agents:
        plt.plot(range(rounds), agent.gradient_history, label=f"{agent.name}")

    plt.title("Gradient Magnitude Over Training Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# 🚀 Main Execution
if __name__ == "__main__":
    num_clients = 5
    num_rounds = 5
    agents = [Agent(f"Client {i+1}") for i in range(num_clients)]
    controller = HybridController(agents, use_quantum=True)

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")
        controller.synchronize_training()
        controller.optimize()

    # ✅ Visualize weight evolution
    plot_weight_changes(agents, num_rounds)

    # ✅ Visualize gradient stability
    plot_gradient_magnitude(agents, num_rounds)

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Synchronize gradients using quantum metadata with phase shifts."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) * np.exp(1j * np.angle(quantum_data[i])))
        for i in range(len(classical_gradients))
    ]
    print(f"Synchronized Gradients: {phase_adjusted_gradients}")
    return phase_adjusted_gradients

     

class FFZStabilizer:
    epsilon = 1e-8  # Prevents division by zero

    @staticmethod
    def stabilize_gradient(gradient):
        """Ensure small gradients are not completely zeroed out."""
        return gradient / (np.sqrt(np.abs(gradient)) + FFZStabilizer.epsilon) + 1e-4  # Added offset

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure balanced gradient scaling across all clients."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.mean(np.abs(quantum_data))  # Normalize quantum influence
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) * np.exp(1j * np.angle(quantum_data[i])) / scaling_factor)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

def plot_gradient_distribution(agents, rounds):
    """Visualize contribution of each client to gradient updates."""
    plt.figure(figsize=(10, 5))

    for agent in agents:
        magnitudes = [np.linalg.norm(grad) for grad in agent.weights_history]
        plt.plot(range(rounds), magnitudes, label=f"{agent.name}")

    plt.title("Gradient Magnitude Contribution Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

     

plot_gradient_distribution(agents, num_rounds)

     

class FFZStabilizer:
    epsilon = 1e-6  # Prevent division by zero and excessively small gradients

    @staticmethod
    def stabilize_gradient(gradient):
        """Ensure gradients aren't completely zeroed out."""
        stabilized = gradient / (np.sqrt(np.abs(gradient)) + FFZStabilizer.epsilon)
        return stabilized + 1e-3  # Small offset to ensure all clients get updates

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure equal scaling across all clients during synchronization."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.mean(np.abs(quantum_data))  # Normalize quantum influence

    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor)  # Normalized scaling
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

import seaborn as sns

def plot_weight_heatmap(agents):
    """Generate a heatmap showing how much each client has updated."""
    weight_deltas = [np.mean(np.abs(np.diff(np.array(agent.weights_history), axis=0)), axis=1) for agent in agents]
    heatmap_data = np.array(weight_deltas)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(heatmap_data.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Weight Change Heatmap Across Clients")
    plt.show()

     

plot_weight_heatmap(agents)

     

def apply_gradient(self, gradient):
    """Apply computed gradient with dynamic adjustment."""
    adaptive_lr = max(0.01, np.linalg.norm(gradient) * 0.05)  # Scale learning rate
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure all clients receive meaningful quantum-modulated updates."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))  # Scale based on max quantum influence

    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor + 1e-3)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

import seaborn as sns

def plot_gradient_heatmap(agents):
    """Visualize gradient influence across clients."""
    gradient_magnitudes = np.array([agent.gradient_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(gradient_magnitudes, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(gradient_magnitudes.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Gradient Magnitude Heatmap")
    plt.show()

     

plot_gradient_heatmap(agents)

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure all clients receive meaningful quantum-modulated updates."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))  # Scale based on max quantum influence

    # Prevent clients from getting tiny updates by enforcing a minimum scaling factor
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

def apply_gradient(self, gradient):
    """Apply computed gradient with dynamic adjustment."""
    adaptive_lr = max(0.02, np.linalg.norm(gradient) * 0.05)  # Increase minimum LR
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

import seaborn as sns

def plot_gradient_heatmap(agents):
    """Visualize gradient influence across clients."""
    gradient_magnitudes = np.array([agent.gradient_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(gradient_magnitudes, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(gradient_magnitudes.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Gradient Magnitude Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with dynamic adjustment."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))  # Prevent stagnation
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure all clients receive meaningful quantum-modulated updates."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))  # Scale based on max quantum influence

    # Boost gradients for clients that have low magnitude updates
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

import seaborn as sns

def plot_gradient_heatmap(agents):
    """Visualize gradient influence across clients."""
    gradient_magnitudes = np.array([agent.gradient_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(gradient_magnitudes, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(gradient_magnitudes.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Gradient Magnitude Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with dynamic adjustment."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))  # Prevent stagnation
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure all clients receive meaningful quantum-modulated updates."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))  # Scale based on max quantum influence

    # Boost gradients for clients that have low magnitude updates
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

import seaborn as sns

def plot_gradient_heatmap(agents):
    """Visualize gradient influence across clients."""
    gradient_magnitudes = np.array([agent.gradient_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(gradient_magnitudes, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(gradient_magnitudes.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Gradient Magnitude Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with dynamic adaptation."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))  # Prevent stagnation
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Ensure all clients receive meaningful quantum-modulated updates."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))  # Scale based on max quantum influence

    # Boost gradients for clients that have low magnitude updates
    phase_adjusted_gradients = [
        np.real(classical_gradients[i] * np.abs(quantum_data[i]) / scaling_factor + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return phase_adjusted_gradients

     

import seaborn as sns

def plot_lr_heatmap(agents):
    """Visualize learning rate changes across clients."""
    lr_data = np.array([agent.lr_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(lr_data, cmap="coolwarm", annot=True, fmt=".3f", xticklabels=[f"Round {i+1}" for i in range(lr_data.shape[1])])
    plt.xlabel("Training Rounds")
    plt.ylabel("Clients")
    plt.title("Learning Rate Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with adaptive LR for all clients."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Boosts low-magnitude updates using quantum gradient amplification."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))

    boosted_gradients = [
        np.real(classical_gradients[i] * (np.abs(quantum_data[i]) / scaling_factor) + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return boosted_gradients

     

import seaborn as sns

def plot_lr_heatmap(agents):
    """Visualize learning rate changes for each client over rounds."""
    lr_data = np.array([agent.lr_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(lr_data, cmap="coolwarm", annot=True, fmt=".3f",
                xticklabels=[f"Round {i+1}" for i in range(lr_data.shape[1])])
    plt.xlabel("Rounds")
    plt.ylabel("Clients")
    plt.title("Learning Rate Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with adaptive LR for all clients."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Boosts slow-learning clients using quantum gradient amplification."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))

    boosted_gradients = [
        np.real(classical_gradients[i] * (np.abs(quantum_data[i]) / scaling_factor) + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return boosted_gradients

     

import seaborn as sns

def plot_lr_heatmap(agents):
    """Visualize learning rate evolution for each client over rounds."""
    lr_data = np.array([agent.lr_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(lr_data, cmap="coolwarm", annot=True, fmt=".3f",
                xticklabels=[f"Round {i+1}" for i in range(lr_data.shape[1])])
    plt.xlabel("Rounds")
    plt.ylabel("Clients")
    plt.title("Learning Rate Evolution Heatmap")
    plt.show()

     

def apply_gradient(self, gradient):
    """Apply computed gradient with adaptive LR for ALL clients."""
    adaptive_lr = max(0.02, min(0.2, np.linalg.norm(gradient) * 0.05))
    self.weights -= adaptive_lr * gradient.real
    self.weights_history.append(self.weights.copy())
    self.gradient_history.append(np.linalg.norm(gradient))
    self.lr_history.append(adaptive_lr)  # Track per-client LR evolution
    print(f"{self.name} updated weights: {self.weights}, LR: {adaptive_lr}")

     

@staticmethod
def synchronize_gradients(classical_gradients):
    """Boosts slow-learning clients using quantum scaling."""
    quantum_data = QuantumSynchronizer.entangle_qubits(len(classical_gradients))
    scaling_factor = np.max(np.abs(quantum_data))

    boosted_gradients = [
        np.real(classical_gradients[i] * (np.abs(quantum_data[i]) / scaling_factor) + 1e-2)
        for i in range(len(classical_gradients))
    ]
    return boosted_gradients

     

import seaborn as sns

def plot_lr_heatmap(agents):
    """Visualize learning rate evolution for each client over rounds."""
    lr_data = np.array([agent.lr_history for agent in agents])

    plt.figure(figsize=(10, 6))
    sns.heatmap(lr_data, cmap="coolwarm", annot=True, fmt=".3f",
                xticklabels=[f"Round {i+1}" for i in range(lr_data.shape[1])])
    plt.xlabel("Rounds")
    plt.ylabel("Clients")
    plt.title("Learning Rate Evolution Heatmap")
    plt.show()

     

class Agent:
    def __init__(self, name):
        self.name = name
        self.weights = np.random.randn(10)  # Initialize weights
        self.weights_history = []  # Track weight history
        self.gradient_history = []  # Track gradient magnitude
        self.lr_history = []  # Track learning rate evolution (NEW)
