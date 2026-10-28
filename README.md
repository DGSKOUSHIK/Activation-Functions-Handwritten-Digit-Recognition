
# Comprehensive Analysis of Activation Functions in Handwritten Digit Recognition

## Project Overview

This research project investigates the influence of different activation functions on the performance of artificial neural networks for handwritten digit recognition.

Activation functions are fundamental components of neural networks because they introduce nonlinearity, allowing models to learn complex relationships between input features and output classes.

The selection of an activation function directly influences gradient propagation, convergence behavior, training stability, and classification accuracy.

This study compares four widely used activation functions:

1. Sigmoid
2. Rectified Linear Unit (ReLU)
3. Leaky ReLU
4. Mish

Additionally, the research proposes a **Hybrid ReLU → Mish Activation Scheduling Strategy**, which dynamically changes the activation function during the training process.

The proposed strategy uses ReLU during the initial training epochs to support fast learning and switches to Mish for the remaining epochs to support smoother gradient-based optimization.

The experiments were conducted using the MNIST handwritten digit dataset and a lightweight fully connected neural network implemented using Python and NumPy.

The experimental results show that Mish achieved the highest individual classification accuracy of **98.07%**, while the proposed Hybrid ReLU → Mish method achieved **97.63% accuracy**.

---

## Key Project Highlights

| Feature | Description |
|---|---|
| Project Domain | Deep Learning |
| Research Area | Neural Network Optimization |
| Application | Handwritten Digit Recognition |
| Dataset | MNIST |
| Dataset Samples Used | 42,000 |
| Image Resolution | 28 × 28 |
| Input Features | 784 |
| Hidden Layers | 1 |
| Hidden Neurons | 10 |
| Output Neurons | 10 |
| Output Activation | Softmax |
| Activation Functions | Sigmoid, ReLU, Leaky ReLU, Mish |
| Proposed Approach | Hybrid ReLU → Mish |
| Switching Epoch | After Epoch 5 |
| Best Individual Activation | Mish |
| Highest Reported Accuracy | 98.07% |
| Hybrid Model Accuracy | 97.63% |
| Programming Language | Python |
| Core Libraries | NumPy, Pandas, Matplotlib |

---

## Problem Statement

Handwritten digit recognition is a widely studied classification problem in machine learning.

Despite the availability of established datasets such as MNIST, accurately recognizing handwritten digits remains challenging due to variations in:

- Individual handwriting styles
- Digit shapes
- Stroke thickness
- Writing angles
- Orientation
- Spacing
- Visual similarity between digits

Neural networks learn to distinguish these patterns through repeated training and parameter optimization.

However, their learning performance depends heavily on the activation functions used within hidden layers.

Different activation functions exhibit different mathematical properties, which influence how gradients propagate through the network.

For example:

- Sigmoid can experience vanishing gradients.
- ReLU may produce inactive neurons.
- Leaky ReLU reduces zero-gradient behavior for negative inputs.
- Mish provides smooth and non-monotonic transformations.

Most conventional neural networks use a fixed activation function throughout training.

This research explores whether different activation functions can be used at different training stages to combine their respective advantages.

---

## Research Objectives

The primary objectives of this study are:

1. Analyze the mathematical properties of common neural network activation functions.
2. Compare Sigmoid, ReLU, Leaky ReLU, and Mish.
3. Study their influence on handwritten digit recognition accuracy.
4. Implement a lightweight fully connected neural network.
5. Evaluate activation functions using comparable architectural and training conditions.
6. Investigate the effect of activation functions on gradient propagation.
7. Propose a Hybrid ReLU → Mish activation scheduling strategy.
8. Use ReLU during initial training epochs for efficient early learning.
9. Switch to Mish during later epochs for smoother optimization.
10. Compare the proposed hybrid method against fixed activation functions.
11. Evaluate the influence of the Leaky ReLU leakage coefficient.
12. Identify limitations and future opportunities for dynamic activation scheduling.

---

## Literature Review

Activation functions have been studied extensively because of their influence on neural network learning.

### Sigmoid Activation

Early neural networks frequently used Sigmoid activation.

The function maps input values into the interval between 0 and 1.

However, Sigmoid can suffer from vanishing gradients when inputs become very large or very small.

This can reduce learning efficiency in deeper networks.

### ReLU Activation

ReLU became popular because of its simple mathematical formulation and computational efficiency.

It allows positive values to pass through while replacing negative values with zero.

Although ReLU often improves training efficiency, it can cause neurons to become permanently inactive.

### Leaky ReLU

Leaky ReLU was introduced to address the zero-gradient problem of ReLU.

Instead of outputting zero for negative inputs, it preserves a small negative slope.

The magnitude of this slope is controlled by a leakage coefficient.

### Mish Activation

Mish is a smooth and non-monotonic activation function.

Its mathematical properties allow smooth transitions between activation values and support gradient propagation.

### Hybrid Activation Strategies

Research into adaptive activation strategies suggests that a single activation function may not necessarily be optimal for every phase of training.

This motivates exploring different activation functions at different training stages.

The current project investigates a scheduled transition from ReLU to Mish.

---

## Research Gap

Existing research has explored individual activation functions extensively.

However, several issues motivate additional study.

### 1. Fixed Activation Functions

Many neural networks use the same activation function throughout the entire training process.

This approach does not explore whether different training phases benefit from different activation characteristics.

### 2. Limited Analysis of Small Networks

Much of the activation-function literature focuses on deep architectures.

Smaller fully connected networks can provide useful controlled environments for comparing activation behavior.

### 3. Inconsistent Experimental Conditions

Previous studies may use different datasets, optimization procedures, training durations, and architectures.

These differences can make direct comparisons difficult.

### 4. Limited Evaluation of Dynamic Activation

There is scope to investigate how changing activation functions during training influences convergence and classification performance.

### 5. Leaky ReLU Parameter Sensitivity

Leaky ReLU introduces a leakage coefficient that can influence training behavior.

This creates an additional parameter that may require careful selection.

---

## Proposed Solution

The research proposes a two-phase training strategy called:

**Hybrid ReLU → Mish Activation Scheduling**

The strategy combines ReLU and Mish within a single training process.

### Phase 1: Initial Learning

During the first five epochs, ReLU is used as the hidden-layer activation function.

The objective is to support efficient early-stage learning and rapid extraction of basic digit patterns.

### Phase 2: Training Refinement

After the fifth epoch, the activation function changes from ReLU to Mish.

Mish is used for the remaining training epochs.

The objective is to support smoother gradient propagation and refinement of learned representations.

### Hybrid Scheduling Rule

```text
Epochs 1–5:

    Activation = ReLU

Epochs 6 onward:

    Activation = Mish
```

### Research Hypothesis

A scheduled activation transition may combine the computational simplicity of ReLU with the smooth nonlinear behavior of Mish.

The study evaluates whether this combination produces competitive performance compared with fixed-activation models.

---

## Dataset Description

### MNIST Dataset

The experiments use the **Modified National Institute of Standards and Technology (MNIST)** handwritten digit dataset.

MNIST is a benchmark dataset for evaluating image classification and neural network algorithms.

The dataset contains grayscale images representing handwritten digits from 0 to 9.

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | MNIST |
| Classification Task | Handwritten Digit Recognition |
| Classes | 10 |
| Digit Labels | 0–9 |
| Image Dimensions | 28 × 28 |
| Color Format | Grayscale |
| Input Features | 784 |
| Samples Used in Research | 42,000 |
| Pixel Value Range | 0–255 |
| Normalized Pixel Range | 0–1 |

### Dataset Challenges

The handwritten digit samples exhibit variations in:

- Writing style
- Stroke width
- Digit shape
- Writing direction
- Orientation
- Spacing
- Visual similarity between digit classes

These variations make MNIST useful for evaluating classification algorithms.

---

## Data Preprocessing

Data preprocessing prepares the digit samples for neural network training.

### Step 1: Data Loading

The dataset is imported from CSV files.

Each data record contains:

- A digit class label
- 784 grayscale pixel values

### Step 2: Feature Preparation

Each 28 × 28 image is represented as a vector containing 784 input features.

### Step 3: Pixel Normalization

Pixel values are divided by 255.

The normalization equation is:

```text
X_normalized = X / 255
```

This scales pixel intensities from the original range of 0–255 into 0–1.

Normalization supports more stable numerical computation during optimization.

### Step 4: Dataset Shuffling

The samples are randomly shuffled to reduce potential ordering bias.

### Step 5: Dataset Partitioning

The processed data is separated into:

- Training data
- Validation data
- Testing data

The research paper describes these subsets but does not specify their exact numerical sizes.

### Step 6: One-Hot Encoding

Digit labels are converted into one-hot encoded vectors.

For example:

```text
Digit: 3

Encoded Label:

[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

The output representation contains 10 elements corresponding to the 10 digit categories.

---

## Neural Network Architecture

A lightweight fully connected neural network is used to study the influence of activation functions.

The network contains three primary stages:

1. Input layer
2. Hidden layer
3. Output layer

### Architecture Configuration

| Component | Configuration |
|---|---|
| Input Layer | 784 features |
| Hidden Layer | 10 neurons |
| Hidden Activation | Configurable |
| Output Layer | 10 neurons |
| Output Activation | Softmax |
| Classification Type | Multiclass Classification |
| Loss Function | Cross-Entropy |
| Optimization | Gradient Descent |

### Input Layer

The input layer accepts a flattened grayscale image containing 784 pixel values.

### Hidden Layer

The hidden layer contains 10 neurons.

Different activation functions are applied to this layer while maintaining the same general network architecture.

The study evaluates:

- Sigmoid
- ReLU
- Leaky ReLU
- Mish
- Hybrid ReLU → Mish

### Output Layer

The output layer contains 10 neurons representing the digits 0 through 9.

The Softmax activation converts the output scores into a probability distribution.

The class with the highest predicted probability is selected as the recognized digit.

---

## Activation Functions

Activation functions introduce nonlinearity into artificial neural networks.

Without nonlinear activation functions, repeated linear transformations can be represented by a single linear transformation.

Nonlinear activations enable neural networks to learn more complex relationships between input features and output labels.

This research evaluates four individual activation functions and one hybrid strategy.

---

## 1. Sigmoid Activation Function

Sigmoid is a smooth, S-shaped activation function.

### Mathematical Formula

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

### Characteristics

- Produces outputs between 0 and 1.
- Continuous and differentiable.
- Historically used in neural networks.
- Can experience saturation.
- May cause vanishing gradients.

### Advantages

- Smooth mathematical behavior.
- Bounded output range.
- Useful in probability-related applications.

### Limitations

When the input magnitude becomes large, the gradient approaches zero.

This can slow parameter updates during backpropagation.

### Experimental Accuracy

**97.55%**

---

## 2. Rectified Linear Unit (ReLU)

ReLU is a widely used activation function in neural networks.

### Mathematical Formula

```text
ReLU(x) = max(0, x)
```

### Behavior

For positive inputs:

```text
ReLU(x) = x
```

For zero or negative inputs:

```text
ReLU(x) = 0
```

### Advantages

- Simple computation.
- Efficient gradient propagation for positive inputs.
- Often supports faster convergence.
- Widely used in neural network architectures.

### Limitations

Negative inputs produce zero output and zero gradient.

Neurons that repeatedly receive negative inputs may become inactive.

This is commonly called the **Dying ReLU Problem**.

### Experimental Accuracy

**98.06%**

---

## 3. Leaky ReLU Activation Function

Leaky ReLU modifies ReLU by allowing a small nonzero slope for negative inputs.

### Mathematical Formula

```text
LeakyReLU(x) = x       if x >= 0

LeakyReLU(x) = αx      if x < 0
```

Where α represents the leakage coefficient.

A commonly used example value is:

```text
α = 0.01
```

### Advantages

- Reduces zero-gradient behavior for negative inputs.
- Helps maintain gradient flow.
- Addresses a limitation of standard ReLU.

### Limitations

- Requires selection of the leakage coefficient.
- Performance can depend on the chosen coefficient.
- Additional parameter tuning may be necessary.

### Experimental Accuracy

**97.57%**

---

## 4. Mish Activation Function

Mish is a smooth, continuous, and non-monotonic activation function.

### Mathematical Formula

```text
Mish(x) = x × tanh(ln(1 + e^x))
```

The function combines the input with a hyperbolic tangent transformation of Softplus.

### Characteristics

- Smooth nonlinear behavior.
- Continuous gradient transitions.
- Non-monotonic activation.
- Supports negative activation values.
- Does not require a separate leakage coefficient.

### Advantages

- Smooth gradient propagation.
- Flexible nonlinear representation.
- Competitive classification performance.
- Suitable for comparison with ReLU-based networks.

### Experimental Accuracy

**98.07%**

Mish achieved the highest reported accuracy among the individual activation functions tested.

---

## 5. Proposed Hybrid ReLU → Mish Activation

The proposed Hybrid Activation Scheduling strategy changes the activation function during model training.

### Mathematical Definition

```text
Hybrid(x, epoch) =

    ReLU(x),    epoch <= 5

    Mish(x),    epoch > 5
```

### Phase 1: ReLU

During epochs 1 through 5, ReLU is used.

The objective is to support rapid early-stage learning.

### Phase 2: Mish

After epoch 5, Mish replaces ReLU.

The objective is to support smoother parameter refinement during later training.

### Advantages of the Proposed Approach

- Uses different activation behavior at different training stages.
- Combines ReLU-based early learning with Mish-based refinement.
- Provides an alternative to fixed activation functions.
- Supports research into activation scheduling.

### Experimental Accuracy

**97.63%**

The hybrid model produced competitive results, although its reported accuracy was lower than Mish and ReLU.

---

## Implementation Methodology

The implementation uses Python and core scientific computing libraries.

The research avoids relying on a high-level neural network framework for the main mathematical operations.

This provides direct control over:

- Activation functions
- Neural network weights
- Bias values
- Forward propagation
- Backpropagation
- Gradient calculations
- Parameter updates
- Model evaluation

### Libraries Used

| Library | Purpose |
|---|---|
| Python | Implementation language |
| NumPy | Matrix operations and numerical computation |
| Pandas | Dataset loading and processing |
| Matplotlib | Training and performance visualization |

---

## Weight and Bias Initialization

Neural network parameters are initialized before training.

### Weight Initialization

Weights are initialized using random values from a uniform distribution.

```text
Weight Range:

-0.5 to +0.5
```

### Bias Initialization

Bias values are initialized to zero.

```text
Bias = 0
```

Random weight initialization helps break symmetry between neurons.

This allows different neurons to learn different feature representations.

---

## Forward Propagation

Forward propagation computes the predicted outputs of the neural network.

### Hidden Layer Linear Transformation

```text
Z1 = W1 × X + b1
```

Where:

- W1 represents hidden-layer weights.
- X represents input features.
- b1 represents hidden-layer biases.

### Hidden Layer Activation

```text
A1 = g(Z1)
```

Here, g represents the selected activation function.

### Output Layer Transformation

```text
Z2 = W2 × A1 + b2
```

### Softmax Output

```text
A2 = Softmax(Z2)
```

Softmax produces normalized probabilities for the ten digit categories.

The predicted class is selected from the highest output probability.

---

## Loss Function

The research uses cross-entropy loss to evaluate prediction errors.

Cross-entropy measures the difference between the predicted probability distribution and the true digit labels.

The loss value is used during backpropagation to update model parameters.

Minimizing cross-entropy helps improve classification performance during training.

---

## Backpropagation

Backpropagation computes the gradients required to update neural network parameters.

The process uses the chain rule of differentiation.

### Output Layer Error

```text
dZ2 = A2 - Y
```

Where:

- A2 represents output probabilities.
- Y represents the true labels.

### Output Layer Weight Gradient

```text
dW2 = (1/m) × dZ2 × A1.T
```

### Output Layer Bias Gradient

```text
db2 = (1/m) × sum(dZ2)
```

### Hidden Layer Error

```text
dZ1 = (W2.T × dZ2) * g'(Z1)
```

Here, g'(Z1) is the derivative of the selected hidden-layer activation function.

The activation derivative determines how gradients propagate through the hidden layer.

### Hidden Layer Weight Gradient

```text
dW1 = (1/m) × dZ1 × X.T
```

### Hidden Layer Bias Gradient

```text
db1 = (1/m) × sum(dZ1)
```

The gradients are used to update the network parameters during optimization.

---

## Gradient Descent Optimization

Gradient descent updates model parameters to reduce prediction error.

### Weight Update

```text
W = W - α × dW
```

### Bias Update

```text
b = b - α × db
```

Where:

- W represents model weights.
- b represents bias values.
- α represents the learning rate.
- dW represents weight gradients.
- db represents bias gradients.

The learning rate controls the size of parameter updates.

The optimization process is repeated over training epochs.

---

## Experimental Setup

To compare activation functions, the research uses the same basic neural network architecture and preprocessing procedure.

### Experimental Configurations

| Experiment | Hidden Activation |
|---|---|
| Experiment 1 | Sigmoid |
| Experiment 2 | ReLU |
| Experiment 3 | Leaky ReLU |
| Experiment 4 | Mish |
| Experiment 5 | Hybrid ReLU → Mish |

The output layer uses Softmax across the experiments.

The research focuses on measuring how the choice of activation function affects handwritten digit recognition accuracy.

---

## Experimental Results

The research reports classification accuracy for five activation configurations.

### Main Classification Results

| Activation Function | Accuracy |
|---|---:|
| Sigmoid | 97.55% |
| ReLU | 98.06% |
| Leaky ReLU | 97.57% |
| Mish | **98.07%** |
| Hybrid ReLU → Mish | 97.63% |

### Result Analysis

**Mish: 98.07%**

Mish achieved the highest reported test accuracy.

**ReLU: 98.06%**

ReLU achieved nearly the same accuracy as Mish, with a difference of only 0.01 percentage points.

**Hybrid ReLU → Mish: 97.63%**

The proposed hybrid schedule achieved competitive classification performance, but did not outperform the best fixed activation functions.

**Leaky ReLU: 97.57%**

Leaky ReLU achieved slightly higher accuracy than Sigmoid in the main comparison.

**Sigmoid: 97.55%**

Sigmoid achieved the lowest accuracy among the five configurations.

### Main Finding

The results indicate that activation-function selection influences classification performance.

The proposed hybrid approach demonstrates the feasibility of changing activation functions during training.

However, the reported results do not establish that hybrid scheduling is more accurate than the best fixed-activation models.

---

## Leaky ReLU Hyperparameter Analysis

An additional experiment investigated how the Leaky ReLU coefficient α influences classification accuracy.

### Reported Results

| Leakage Coefficient (α) | Accuracy |
|---|---:|
| 0.01 | 75.0% |
| 0.05 | 79.0% |
| 0.10 | 84.0% |
| 0.50 | 90.0% |
| 1.00 | 92.0% |

The reported values show increasing accuracy with larger leakage coefficients within this particular experiment.

These results are presented separately from the main comparison, where Leaky ReLU achieved 97.57%.

The paper does not fully explain why these two sets of reported results differ, so they should not be treated as directly interchangeable.

The results suggest that the leakage coefficient can influence training performance.

---

## Discussion

The experimental comparison provides several observations.

### Sigmoid

Sigmoid achieved lower accuracy than the other reported configurations.

Its saturation characteristics can restrict gradient propagation.

### ReLU

ReLU produced strong classification accuracy and provides computational simplicity.

However, its zero-gradient behavior for negative inputs can lead to inactive neurons.

### Leaky ReLU

Leaky ReLU addresses the negative-input gradient limitation of ReLU.

Its results depend on the selected leakage coefficient.

### Mish

Mish demonstrated the highest reported accuracy.

Its smooth and non-monotonic behavior makes it an interesting alternative to conventional activation functions.

### Hybrid ReLU → Mish

The hybrid scheduling strategy demonstrates that an activation function does not necessarily have to remain fixed throughout training.

Although the hybrid result was lower than Mish alone, the experiment provides a basis for exploring adaptive activation scheduling.

---

## Key Research Contributions

### 1. Controlled Activation Comparison

The research evaluates four individual activation functions using a common neural network architecture.

### 2. Hybrid Activation Scheduling

The project proposes switching from ReLU to Mish during training.

### 3. Two-Phase Training Strategy

The proposed schedule separates initial learning and later optimization refinement.

### 4. Lightweight Neural Network

The experiments use a small fully connected neural network with 784 input features, 10 hidden neurons, and 10 output neurons.

### 5. Manual Neural Network Implementation

The research uses Python and scientific computing libraries to implement neural network operations.

### 6. Leaky ReLU Coefficient Analysis

The study includes an investigation of how the leakage coefficient affects reported classification accuracy.

### 7. Empirical Evaluation

The research compares five configurations on handwritten digit recognition.

---

## Limitations

### 1. Single Dataset

The experiments focus on MNIST handwritten digit recognition.

Performance on other image classification datasets has not been established by these results.

### 2. Lightweight Architecture

The neural network contains only one hidden layer with 10 neurons.

Findings may differ in deeper architectures.

### 3. Fixed Switching Epoch

The hybrid strategy switches activation functions after five epochs.

Other switching points were not systematically evaluated in the reported results.

### 4. Hybrid Accuracy

The Hybrid ReLU → Mish approach achieved 97.63% accuracy, below Mish at 98.07% and ReLU at 98.06%.

### 5. Hyperparameter Sensitivity

The behavior of Leaky ReLU depends on the leakage coefficient.

### 6. Generalization

Additional experiments are required to determine whether the same activation behavior occurs across other datasets and network architectures.

### 7. Training Efficiency

The paper discusses convergence behavior, but does not provide a complete numerical benchmark of training time or computational cost for each configuration.

---

## Future Scope

Potential research extensions include:

1. Evaluating hybrid activation scheduling on deeper neural networks.
2. Testing the approach on convolutional neural networks.
3. Investigating adaptive activation switching.
4. Comparing different switching epochs.
5. Testing multiple activation transitions.
6. Evaluating performance on Fashion-MNIST and other image datasets.
7. Analyzing gradient distributions across epochs.
8. Comparing training time and computational requirements.
9. Investigating alternative activation combinations.
10. Studying activation schedules under different optimization algorithms.

These are proposed directions for future research.

---

## Potential Applications

The techniques explored in this research may be relevant to:

- Handwritten digit recognition
- Document digitization
- Optical character recognition
- Automated form processing
- Postal code recognition
- Educational deep learning experiments
- Neural network optimization research
- Activation-function benchmarking

The reported experiments focus specifically on MNIST digit classification.

---

## Repository Structure

The research repository can be organized as follows:

```text
Activation-Functions-Handwritten-Digit-Recognition/
|
|-- README.md
|
|-- Activation Functions.pdf
```

Additional source code, datasets, or results can be included separately when available.

---

## Research Paper

**Title:** Comprehensive Analysis of Activation Functions in Handwritten Digit Recognition

The research paper covers:

- Introduction
- Literature Survey
- Research Gaps
- Proposed Methodology
- Implementation Modules
- Activation Functions
- Forward and Backward Propagation
- Experimental Results
- Hybrid Activation Scheduling
- Conclusion
- References

---

## Research Authors

**Koushik Dgs**

School of Computer Science and Engineering  
Vellore Institute of Technology  
Vellore, Tamil Nadu, India

**Sayan Sikder**

School of Computer Science and Engineering  
Vellore Institute of Technology  
Vellore, Tamil Nadu, India

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Artificial Neural Networks
- Machine Learning
- Deep Learning
- Gradient Descent
- Backpropagation
- Softmax Classification
- Activation Function Optimization

---

## Conclusion

This research evaluates the performance of Sigmoid, ReLU, Leaky ReLU, Mish, and a proposed Hybrid ReLU → Mish activation scheduling strategy for handwritten digit recognition.

The experiments use a lightweight fully connected neural network trained on the MNIST dataset.

Among the individual activation functions, Mish achieved the highest reported accuracy of **98.07%**, closely followed by ReLU at **98.06%**.

The proposed Hybrid ReLU → Mish strategy achieved **97.63% accuracy**, demonstrating competitive classification performance while introducing an alternative approach to activation selection during training.

Although the hybrid method did not outperform the strongest fixed activation functions, the results support further investigation of phase-dependent activation scheduling.

The study contributes to understanding how activation functions influence neural network performance and provides a foundation for future work on adaptive training strategies.

---

## Contact

**Koushik Daggupati**

GitHub: https://github.com/DGSKOUSHIK

---

**Research Area:** Deep Learning | Neural Networks | Activation Functions | Handwritten Digit Recognition
