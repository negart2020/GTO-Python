# Gorilla Troops Optimizer (GTO) 
A Python implementation of the **Artificial Gorilla Troops Optimizer**, inspired by the collective behavior and intelligent movements of gorilla groups in nature.

---

## Introduction
The **Artificial Gorilla Troops Optimizer (GTO)** is a nature-inspired metaheuristic optimization algorithm introduced in 2021.  
GTO simulates:
- Group movement of gorillas
- Decision-making strategies of the silverback leader
- Exploration and exploitation balance through biological behaviors

This repository contains a simple and clean **Python version** of the algorithm.

---

## Reference — Original Research Paper
> A. Abdollahzadeh, H. Eslami, T. Mirjalili  
> “Artificial Gorilla Troops Optimizer: A new nature-inspired metaheuristic algorithm.”  
> *Engineering Applications of Artificial Intelligence*, Volume 94, 2021, 103830.  
> [https://doi.org/10.1016/j.engappai.2020.103830](https://www.researchgate.net/publication/353186350_Artificial_gorilla_troops_optimizer_A_new_nature-inspired_metaheuristic_algorithm_for_global_optimization_problems)

 Original MATLAB implementation provided by authors:
https://github.com/Benyamin-abdollahzadeh/Artificial-gorilla-troops-optimizer

---

## Algorithm Idea (Short Summary)
In GTO:
- Each gorilla represents a candidate solution
- The **silverback** represents the best solution found so far
- **Exploration** happens when gorillas wander or investigate new areas
- **Exploitation** happens when gorillas move toward the silverback

GTO maintains a strong balance between:
- Searching globally (avoiding local minima)
- Converging fast toward optimal region

---

##  Requirements
Install Python dependencies:

```sh
pip install numpy
