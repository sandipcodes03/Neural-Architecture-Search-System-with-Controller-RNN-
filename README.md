# Neural Architecture Search via Reinforcement Learning: A Practical Implementation

This project embodies an implementation of Neural Architecture Search (NAS) utilizing the REINFORCE algorithm. It leverages a recurrent neural network (RNN) to generate model descriptions of neural networks and trains this RNN via reinforcement learning to optimize the anticipated accuracy of the generated architectures on a validation set. The effectiveness of this algorithm is evaluated on the CIFAR-10 dataset. The project draws inspiration from the seminal work "Neural Architecture Search with Reinforcement Learning" by Barret et al. from Google Brain.

## Architecture
![alt text](https://miro.medium.com/max/656/1*hIif88uJ7Te8MJEhm40rbw.png)
![alt text](https://i.ytimg.com/vi/CYUpDogeIL0/maxresdefault.jpg)

## Technical Deep Dive

**Core Concept**:

* **Controller (RNN)**: A recurrent neural network is employed to generate variable-length strings representing the architecture of a neural network. These strings encode hyperparameters such as the number of layers, type of layers (convolutional, pooling, etc.), filter sizes, and connectivity patterns.

* **Child Network**: The neural network architecture described by the controller is constructed and trained on the training set.

* **Reward Signal**: The accuracy achieved by the child network on a validation set serves as the reward signal.

* **Policy Gradient**: The REINFORCE algorithm is applied to update the parameters of the controller RNN, maximizing the expected reward (validation accuracy).

**Implementation Highlights**:

* **RNN Controller**: The controller is typically an LSTM or GRU network, generating a sequence of tokens that define the child network architecture.

* **Action Space**: The action space of the controller consists of choices for various hyperparameters (layer type, filter size, etc.).

* **Child Network Training**: The generated child network is trained for a fixed number of epochs on the training set.

* **Reward Calculation**: The validation accuracy of the child network is used as the reward signal.

* **Policy Gradient Update**: The parameters of the controller RNN are updated using the policy gradient, aiming to increase the probability of generating architectures that lead to higher validation accuracy.

**CIFAR-10 Experiment**:

* **Dataset**: The CIFAR-10 dataset, comprising 60,000 32x32 color images in 10 classes, is used for evaluation.

* **Search Space**: The search space includes various convolutional and pooling layers, with different filter sizes and numbers of filters.

* **Training**: The controller RNN and child networks are trained iteratively, with the controller progressively generating better-performing architectures.

**Considerations :**:

* **Exploration-Exploitation Tradeoff**: Balance the exploration of new architectures with the exploitation of known good ones.

* **Computational Cost**: NAS can be computationally expensive, requiring the training of numerous child networks. Consider strategies for efficient search, such as weight sharing or early stopping.

* **Search Space Design**: The choice of search space significantly impacts the quality of the discovered architectures. Carefully consider the types of layers and hyperparameters to include.

* **Beyond Accuracy**: Incorporate other metrics into the reward signal, such as model size or inference speed, to discover architectures that balance accuracy with efficiency.

This implementation serves as a foundational step in understanding and applying NAS using reinforcement learning. Senior ML engineers can leverage this knowledge to design and discover novel architectures tailored to specific tasks and constraints.

**Note**: The project is inspired by the work of Barret et al., but may differ in specific implementation details. Refer to the original paper for a comprehensive understanding of their approach.
