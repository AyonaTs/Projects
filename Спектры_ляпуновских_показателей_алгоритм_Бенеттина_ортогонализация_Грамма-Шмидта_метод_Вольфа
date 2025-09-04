import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры
sigma = 10.0
beta  = 8/3
rho   = 28.0

def lorenz_rhs(t, state):
    x, y, z = state
    return np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])

def lorenz_jac(state):
    x, y, z = state
    return np.array([
        [-sigma,    sigma,      0],
        [ rho - z,   -1.0,    -x],
        [    y,       x,    -beta]
    ])

def lyap_spectrum_benettin(rhs, jac, state0, dt, t_trans, t_max, dim):
    # t_trans — время переходного процесса, t_max — итоговое время (всего)
    # dim — размерность системы (для Лоренца dim=3)
    # инициализация
    t = 0.0
    state = state0.copy()
    # вариационная матрица Φ (изначально единичная)
    Phi = np.eye(dim)
    # накопитель сумм ln|r_jj|
    sums = np.zeros(dim)

    # фаза переходного процесса
    while t < t_trans:
        # один шаг RK4 для (state, Φ)
        k1_s = rhs(t, state)
        k1_P = jac(state).dot(Phi)

        k2_s = rhs(t+dt/2, state + dt*k1_s/2)
        k2_P = jac(state + dt*k1_s/2).dot(Phi + dt*k1_P/2)

        k3_s = rhs(t+dt/2, state + dt*k2_s/2)
        k3_P = jac(state + dt*k2_s/2).dot(Phi + dt*k2_P/2)

        k4_s = rhs(t+   dt, state + dt*k3_s)
        k4_P = jac(state + dt*k3_s).dot(Phi + dt*k3_P)

        state += dt*(k1_s + 2*k2_s + 2*k3_s + k4_s)/6
        Phi   += dt*(k1_P + 2*k2_P + 2*k3_P + k4_P)/6

        t += dt

    # основная фаза с QR-нормировками
    n_steps = int((t_max - t_trans)/dt)
    for _ in range(n_steps):
        # RK4 для state и для Φ
        k1_s = rhs(t, state)
        k1_P = jac(state).dot(Phi)

        k2_s = rhs(t+dt/2, state + dt*k1_s/2)
        k2_P = jac(state + dt*k1_s/2).dot(Phi + dt*k1_P/2)

        k3_s = rhs(t+dt/2, state + dt*k2_s/2)
        k3_P = jac(state + dt*k2_s/2).dot(Phi + dt*k2_P/2)

        k4_s = rhs(t+   dt, state + dt*k3_s)
        k4_P = jac(state + dt*k3_s).dot(Phi + dt*k3_P)

        state += dt*(k1_s + 2*k2_s + 2*k3_s + k4_s)/6
        Phi   += dt*(k1_P + 2*k2_P + 2*k3_P + k4_P)/6

        # QR-разложение Φ = Q·R
        Q, R = np.linalg.qr(Phi)
        Phi = Q
        # аккумулируем ln|r_jj|
        sums += np.log(np.abs(np.diag(R)) + 1e-16)

        t += dt

    # нормируем на общее время интегрирования в основной фазе
    total_time = t_max - t_trans
    return sums / total_time

# Параметры для Лоренца (пример)
state0 = np.array([1.,1.,1.])
dt      = 0.01
t_trans = 10.0
t_max   = 110.0
dim     = 3

spec = lyap_spectrum_benettin(lorenz_rhs, lorenz_jac, state0, dt, t_trans, t_max, dim)
print("Corrected Lyapunov spectrum:", spec)

def lyap_wolf(rhs, state0, delta0, dt, t_trans, t_max, M=20):

    # Метод Вольфа для наибольшего λ с нормировкой каждые M шагов.
    # rhs       – функция правой части (t, state) -> ndarray
    # state0    – начальное состояние (ndarray)
    # delta0    – начальное разделение (float)
    # dt        – шаг интегрирования
    # t_trans   – время переходного процесса (float)
    # t_max     – общее время (float)
    # M         – число шагов между нормировками

    # 1. Инициализация
    x1 = state0.copy()
    x2 = state0 + delta0
    sum_log = 0.0
    n = 0
    t = 0.0

    # 2. Переходный процесс (без учёта)
    while t < t_trans:
        # интегрируем обе траектории шагом dt
        for x in (x1, x2):
            k1 = dt * rhs(t,        x)
            k2 = dt * rhs(t + dt/2, x + k1/2)
            k3 = dt * rhs(t + dt/2, x + k2/2)
            k4 = dt * rhs(t +    dt, x + k3)
            x += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt

    # 3. Основной цикл с нормировкой каждые M шагов
    while t < t_max:
        # 3.1. Интегрируем M шагов RK4
        for _ in range(M):
            for x in (x1, x2):
                k1 = dt * rhs(t,        x)
                k2 = dt * rhs(t + dt/2, x + k1/2)
                k3 = dt * rhs(t + dt/2, x + k2/2)
                k4 = dt * rhs(t +    dt, x + k3)
                x += (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt  # учитываем M·dt времени

        # 3.2. Измеряем разделение и накапливаем логарифм
        delta = x2 - x1
        dist  = np.linalg.norm(delta)
        sum_log += np.log(dist / delta0)
        n += 1

        # 3.3. Нормируем разделение обратно к delta0
        x2 = x1 + (delta0 / dist) * delta

    # 4. Возвращаем средний λ
    return sum_log / n


# Параметры
delta0 = 1e-8
t_trans_w = 10.0
t_max_w   = 100.0

λ1 = lyap_wolf(lorenz_rhs, np.array([1.0,1.0,1.0]),
                     delta0=1e-8, dt=0.01,
                     t_trans=10.0, t_max=100.0, M=20)
print("Largest Lyapunov λ ≈", λ1)

λ1 = lyap_wolf_fixed(
    lorenz_rhs,
    state0    = np.array([1.0,1.0,1.0]),
    delta0    = 1e-8,
    dt        = 0.001,
    t_trans   = 50.0,
    t_max     = 500.0,
    M         = 100
)
print("Wolf финальный λ1 =", λ1)

# Выводим результаты
print("Сравнение методов")
print(f"Benettin spectrum: λ1={spec[0]:.4f}, λ2={spec[1]:.4f}, λ3={spec[2]:.4f}")
print(f"Wolf largest:     λ1={λ1:.4f}")

# Проверка для Лоренца при (σ,ρ,β)=(10,28,8/3):
# – ожидаемые: λ1≈0.905, λ2≈0.0…, λ3≈−14.6
