# Physics-Informed Neural Network for Option Pricing

[Live demo](https://theobradbury-pinn-black-scholes.streamlit.app)

I wanted to see whether a neural network could solve the Black-Scholes PDE without ever seeing real option prices. This is a PyTorch model that learns to price European calls by being penalised whenever it breaks the PDE or its boundary conditions, so there's no training data at all.

It treats the option value as a function of five inputs:

$$V = V(S, t, K, r, \sigma)$$

| Input | Range |
| :--- | :--- |
| Stock price $S$ | 0 to 200 |
| Time $t$ | 0 to 1 (maturity $T = 1$) |
| Strike $K$ | 70 to 130 |
| Interest rate $r$ | 0.01 to 0.08 |
| Volatility $\sigma$ | 0.10 to 0.35 |

One trained network can price calls across all of these without retraining.

## How it works

The network has to satisfy the Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

along with the payoff at maturity, $V = \max(S-K, 0)$, and $V = 0$ at $S = 0$. PyTorch's autograd gives me the derivatives, and the loss is how badly the network breaks each of these conditions.

**Architecture:** a simple MLP with 3 hidden layers of 128 neurons and tanh activations. Inputs are normalised, since $S \sim 100$ but $r \sim 0.05$, and the output is scaled back up by 100.

**Training:** 5,000 epochs with Adam, resampling random points each epoch (10,000 interior, 10,000 terminal, 400 boundary). The terminal loss weight is adjusted every 100 epochs so it doesn't swamp the PDE loss.

## Greeks

Since the network is smooth, Delta, Gamma, Theta and Vega come straight from autograd, with no finite differences needed.

## Results

Tested against the analytical formula at $S = K = 100$, $t = 0$, $r = 0.05$, $\sigma = 0.20$:

| | PINN | Analytical |
| :--- | :--- | :--- |
| Call price | \$10.46 | \$10.45 |
| Delta | 0.6368 | 0.6368 |

That's under 0.1% error on price, and inference takes under 1 ms.

## Streamlit app

`app.py` lets you play with pricing and Greeks, compare against the closed-form solution, and view a 3D price surface in Plotly.

## What I learned from

- 3Blue1Brown's [neural networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Steve Brunton on [PINNs](https://www.youtube.com/watch?v=-zrY7P2dVC4&t=1215s)
- MIT OCW on [Black-Scholes](https://www.youtube.com/watch?v=TnS8kI_KuJc)
- quantpie's [PDE derivation](https://www.youtube.com/watch?v=-qa2B_sCpZQ)
