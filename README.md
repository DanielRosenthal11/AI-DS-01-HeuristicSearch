# 8-Puzzle Solver – A* Search

**Course:** Introduction to Artificial Intelligence and Data Science, WS 2026/27  
**Task:** Task 1 – Heuristic Search

## Overview

This project implements an A* search algorithm to solve the 8-puzzle in Python. It compares two heuristic functions, Hamming distance and Manhattan distance, to evaluate their impact on search performance.

The program generates random solvable start states, finds solutions, and measures the number of expanded nodes and execution time for each heuristic. It then calculates the mean and standard deviation of these measurements across 100 test cases.

## Requirements

- Python 3.8 or newer
- No external packages required

## Usage

Run the program with the default settings:

```
python puzzle8.py
```

This runs the experiment with 100 random solvable start states and uses seed 42 by default. The program also displays one example solution step by step.

### Optional arguments

Run 50 test cases with a specific random seed:

```
python puzzle8.py --n 50 --seed 7
```

Export the individual measurements to a CSV file:

```
python puzzle8.py --csv results.csv
```

## Goal State

The goal state used in this implementation is:

```text
0 1 2
3 4 5
6 7 8
```

The blank space is represented by `0`.


See the project documentation for the software architecture, design decisions, complexity analysis, and discussion of the experimental results.
