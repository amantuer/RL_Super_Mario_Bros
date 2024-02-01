# Super Mario Bros Reinforcement Learning Project

![Description](images/mario.png)

Embark on an exciting journey to create an AI that can master the classic game of Super Mario Bros! This project harnesses the power of the Double Deep Q Network (DDQN) Reinforcement Learning algorithm to train an AI agent. Our goal is to develop an intelligent agent capable of navigating the challenges of this iconic game, learning strategies, and making decisions just like a human player.

## Installation

**Step 1: Clone the Repository**
```bash
git clone https://github.com/amantuer/RL_Super_Mario_Bros.git
```

**Step 2: Create a Virtual Environment**

Use the following command for a conda environment (adapt for your preferred environment manager). Python version used is 3.10.

```bash
conda create --name your_env_name python=3.10
```

Activate the new environment:

```bash
conda activate your_env_name
```

**Step 3: Install PyTorch v2.1.1**

Installation varies based on GPU usage and CUDA version. For detailed instructions, visit [PyTorch's website](https://pytorch.org/get-started/locally/). Example for CUDA 12.1:


```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install Remaining Requirements**

```bash
pip install -r requirements.txt
```

**Adding New Libraries**

If you need to add new libraries in the future, follow these steps:

1.Add the library to requirements.txt with a specific version.

2.Install the new library using pip install <library-name>.

3.Update the repository to include the updated requirements.txt.

## License

This project is licensed under the [MIT](LICENSE.md).


