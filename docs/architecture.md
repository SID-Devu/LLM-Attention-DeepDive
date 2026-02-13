# Attention Mechanism Architecture

## Overview

This document explains the attention mechanism implementations in this repository.

## Standard Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Memory Complexity
- Standard: O(n²) for sequence length n
- Flash Attention: O(n) with tiling

## Implementation Comparison

| Kernel | Memory Pattern | Complexity | Best For |
|--------|----------------|------------|----------|
| Naive | Global only | O(n²) | Small sequences |
| Shared | LDS tiled | O(n²) | Medium sequences |
| Flash | Block-wise | O(n) | Long sequences |

## GPU Memory Layout

```
┌─────────────────────────────────────────┐
│           Global Memory (HBM)           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │  Q  │ │  K  │ │  V  │ │ Out │       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──▲──┘       │
│     │       │       │       │           │
│     └───────┴───────┴───────┘           │
│              Tile Load                   │
│                 │                        │
│     ┌───────────▼───────────┐           │
│     │    Shared Memory      │           │
│     │  Q_tile  K_tile       │           │
│     └───────────┬───────────┘           │
│                 │ Compute                │
│     ┌───────────▼───────────┐           │
│     │      Registers        │           │
│     │   Partial Softmax     │           │
│     └───────────────────────┘           │
└─────────────────────────────────────────┘
```
