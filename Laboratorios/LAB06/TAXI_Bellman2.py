import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Inicialización del entorno
env = gym.make('Taxi-v3')

# Parámetros
gamma = 0.95  # Factor de descuento para recompensas futuras
theta = 1e-4  # Umbral de convergencia

# Paso 1: Inicializar la función de valor de estado V en ceros
V = np.zeros(env.observation_space.n)

# Función de actualización de Bellman para los valores de estado
def bellman_update_state_value(env, V, gamma):
    delta = 0
    # Acceso al entorno base con env.unwrapped
    for state in range(env.observation_space.n):
        v = V[state]
        V[state] = max([sum([prob * (reward + gamma * V[next_state])
                             for prob, next_state, reward, done in env.unwrapped.P[state][action]])
                        for action in range(env.action_space.n)])
        delta = max(delta, abs(v - V[state]))
    return delta

# Paso 2: Iteración de valores para calcular V(s)
delta_history = []  # Almacena el delta en cada iteración para graficar la convergencia
iteration = 0
while True:
    delta = bellman_update_state_value(env, V, gamma)
    delta_history.append(delta)
    if delta < theta:
        break
    iteration += 1
    print(f"Iteración {iteration}, Delta = {delta}")

print("\nValores de estado (V) después de la convergencia:")
print(V)

# Paso 3: Graficar la convergencia de los valores de estado
plt.plot(delta_history)
plt.xlabel('Iteraciones')
plt.ylabel('Delta')
plt.title('Convergencia de la Iteración de Valores')
plt.show()

# Paso 4: Visualización de la política óptima
def decode_taxi_state(state):
    """Decodifica un estado en sus componentes: fila, columna, ubicación del pasajero y destino."""
    taxi_row, taxi_col, passenger_location, destination = env.unwrapped.decode(state)
    return {'taxi_row': taxi_row, 'taxi_col': taxi_col, 
            'passenger_location': passenger_location, 'destination': destination}

# Reiniciar el entorno y configurar para visualización
state = env.reset()[0]
env = gym.make('Taxi-v3', render_mode='human')  # Modo con renderizado
env.reset()
total_reward = 0
done = False
step_count = 0

print("\nEjecutando la política óptima en un episodio:")
while not done:
    # Selecciona la acción óptima en el estado actual según los valores de estado V
    action = np.argmax([sum([prob * (reward + gamma * V[next_state]) 
                             for prob, next_state, reward, done in env.unwrapped.P[state][action]])
                        for action in range(env.action_space.n)])
    # Ejecuta la acción en el entorno
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    step_count += 1

    # Renderiza el entorno y muestra el estado decodificado
    env.render()
    decoded_state = decode_taxi_state(state)
    print(f"Paso {step_count}: Estado = {decoded_state}, Acción = {action}, Recompensa = {reward}")
    if truncated:
        break

print(f"\nEpisodio completado en {step_count} pasos con una recompensa total de {total_reward}.")
env.close()
