import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Определение системы Лоренца и бифуркационной диаграммы

def lorenz(state, t, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Определение системы Росслера

def rossler(state, t, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]

# Определение системы Sprott A

def sprott_a(state, t, a=1.0):
    x, y, z = state
    dx = y
    dy = -x - y * z
    dz = y**2 - a
    return [dx, dy, dz]

# вычисление пересечений сечения Пуанкаре при z = z0 (пересечение вверх)
def poincare_section(sol, z0=0.0):
    crossings = []
    for i in range(len(sol) - 1):
        z1, z2 = sol[i,2], sol[i+1,2]
        if z1 < z0 and z2 >= z0:
            # линейная интерполяция
            alpha = (z0 - z1) / (z2 - z1)
            pt = sol[i,:] + alpha * (sol[i+1,:] - sol[i,:])
            crossings.append(pt)
    arr = np.array(crossings)
    # гарантируем двумерный массив с тремя колонками
    if arr.size == 0:
        return arr.reshape((0, 3))
    return arr

# Построение бифуркационной диаграммы Лоренца

def plot_lorenz_bifurcation(rho_vals, sigma=10.0, beta=8/3, transient=100, t_max=200, dt=0.01):
    t = np.arange(0, t_max, dt)
    fig, ax = plt.subplots(figsize=(8,6))
    for rho in rho_vals:
        sol = odeint(lorenz, [1,1,1], t, args=(sigma, rho, beta))
        # отбрасываем переходный процесс
        z = sol[int(transient/dt):,2]
        # поиск локальных максимумов (простой тест по соседним значениям)
        for i in range(1, len(z)-1):
            if z[i-1] < z[i] > z[i+1]:
                ax.plot(rho, z[i], ',k')
    ax.set_xlabel('rho')
    ax.set_ylabel('z maxima')
    ax.set_title('Lorenz bifurcation diagram')
    plt.tight_layout()
    plt.show()

# использование и построения графиков
if __name__ == '__main__':
    # Бифуркация системы Лоренца
    rhos = np.linspace(0, 100, 401)
    plot_lorenz_bifurcation(rhos)

    # Аттрактор Ресслера
    t = np.linspace(0, 200, 20000)
    ross_sol = odeint(rossler, [0.1, 0.0, 0.0], t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ross_sol[:,0], ross_sol[:,1], ross_sol[:,2], lw=0.5)
    ax.set_title('Rеssler attractor')
    plt.show()

    # Сечение Пуанкаре для Ресслера (плоскость z=0)
    ross_sec = poincare_section(ross_sol, z0=5.0)
    plt.figure()
    plt.scatter(ross_sec[:,0], ross_sec[:,1], s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rössler Poincaré section (z=5)')
    plt.tight_layout()
    plt.show()

    # Аттрактор Sprott A и его сечение Пуанкаре
    sprott_sol = odeint(sprott_a, [0.0, 5.0, 0.0], t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sprott_sol[:,0], sprott_sol[:,1], sprott_sol[:,2], lw=0.5)
    ax.set_title('Sprott A attractor')
    plt.show()

    sprott_sec = poincare_section(sprott_sol, z0=0.0)
    plt.figure()
    plt.scatter(sprott_sec[:,0], sprott_sec[:,1], s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sprott A Poincarе section (z=0)')
    plt.tight_layout()
    plt.show()

