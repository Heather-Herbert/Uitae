# Uitae: Loom-Node

A Node.js implementation of a **Predictive Coding Neural Network** based on the "Loom-Node" specification. This project implements a recursive predictive mind that performs simultaneous inference and Hebbian learning.

## Features

- **Recursive Hierarchy:** A multi-layered architecture where higher layers predict the state of lower layers.
- **Simultaneous Learning:** Inference (energy minimization) and Hebbian weight updates occur within the same processing loop.
- **Dynamic Expansion:** The "Growth" module monitors error levels and automatically adds nodes to layers that exceed a confusion threshold.
- **Character-Level Learning:** Encodes UTF-8 bytes to interact with the user character-by-character.
- **Persistence:** Saves and loads the "Mind" state (weights and hierarchy) to/from a binary file (`mind.bin`).

## Prerequisites

- [Node.js](https://nodejs.org/) (v20 or later recommended)
- [npm](https://www.npmjs.com/)

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:Heather-Herbert/Uitae.git
   cd Uitae
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Project

### Chat Mode

To start the interactive chat loop with the predictive mind:

```bash
node index.js
```

### Scraping Mode

To gather training data from a website:

```bash
node scraper.js <url>
```

This will save the text of the main page and all linked pages (one level deep) into the `./data` folder. You can then run the trainer on this folder:

```bash
node train.js ./data
```

### Interaction

- Once running, you will see the prompt `>`.
- Type a message (e.g., "Hello Smudge.") and press Enter.
- The model will process each byte of your input, minimizing prediction error and updating its weights in real-time.
- The "Mind" will then generate a response based on its current internal state.
- The state of the mind is automatically saved to `mind.bin` after each interaction.

## Development

### Linting

This project uses ESLint to maintain code quality. A GitHub Action is configured to run linting on every push.

To run the linter locally:
```bash
npm run lint
```

## Architecture Notes

The core of the system is the `Mind` class, which manages an array of `Layer` objects. 

- **Inference Rule:** $X[l] += dt \times (-E[l] + (W[l-1]^T \times E[l-1]))$
- **Error Calculation:** $E[l] = X[l] - (W[l] \times X[l+1])$
- **Learning Rule:** $W[l] += lr \times (E[l] \otimes X[l+1])$ (Outer Product)

The model is designed to be "unclamped" during prediction, allowing it to free-run based on its internal temporal expectations.
