import matplotlib.pyplot as plt
import numpy as np
from utils.flow_utils import DirichletConditionalFlow




def plot_flow_map(x0, n_steps, alpha_max = 8, K = 4):
    h = (alpha_max - 1) / n_steps
    t = 1
    xs = [x0]
    ts = [1]
    x = xs[0]
    for i in range(n_steps):
        u = x ** t
        t = t+h
        x = u ** (1/t)
        xs.append(x)
        ts.append(t)

    xss = [x0]
    tss = [1]

    x = xs[0]
    t=1
    condflow = DirichletConditionalFlow(K=K, alpha_spacing=h, alpha_max=alpha_max)
    for i in range(n_steps-20):
        c = condflow.c_factor(x, t)
        vector_flow = (1 - x) * c * h
        x = x + vector_flow
        u = x ** t
        t = t+h
        x = u ** (1/t)
        xss.append(x)
        tss.append(t)

    print(xss, tss)
    plt.plot(np.asarray(xs)*1.5, ts)
    plt.plot(xss, tss)
    plt.show()

plot_flow_map(0.05, 100, K = 20)