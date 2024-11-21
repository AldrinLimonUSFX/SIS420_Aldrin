import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Inicialización del entorno y configuración de parámetros
env = gym.make('Taxi-v3')  # Eliminamos el render_mode para evitar renderizado

# Parámetros de entrenamiento
episodes = 5000             # Número total de episodios de entrenamiento
learning_rate = 0.1                 # Tasa de aprendizaje
discount_factor = 0.99               # Factor de descuento
epsilon = 1.0               # Valor inicial de epsilon para exploración
epsilon_decay = 0.0005       # Factor de decaimiento de epsilon
min_epsilon = 0          # Valor mínimo de epsilon ajustado

# Inicializar la tabla Q
Q = np.zeros([env.observation_space.n, env.action_space.n])  # Tabla Q inicializada en ceros

# Lista para almacenar las recompensas totales por episodio
rewards_per_episode = []

# Bucle de entrenamiento
for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not done and not truncated:
        # Selección de acción con epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state])        # Explotación

        # Ejecuta la acción y observa el nuevo estado y recompensa
        next_state, reward, done, truncated, _ = env.step(action)

        # Penalización si el agente se queda en el mismo estado por más de 5 pasos
    
        # Actualización de la ecuación de Bellman
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])

        # Acumula la recompensa total del episodio
        total_reward += reward 
        state = next_state

    # Almacenar la recompensa total del episodio
    rewards_per_episode.append(total_reward)

    # Reducir epsilon gradualmente hasta un valor mínimo (epsilon decay)
    epsilon = max(min_epsilon, epsilon - epsilon_decay)

    # Imprime el progreso cada 100 episodios
    if episode % 100 == 0:
        print(f"Episodio {episode}: Recompensa total = {total_reward}, Epsilon = {epsilon:.4f}")

env.close()

# Graficar las recompensas acumuladas por episodio
plt.plot(rewards_per_episode)
plt.xlabel('Episodios')
plt.ylabel('Recompensa total')
plt.title('Recompensas Totales por Episodio (Epsilon-Greedy)')
plt.show()

# Verificación y visualización de la política aprendida
listaEstados = []
done = False
truncated = False
G, reward = 0, 0
state, _ = env.reset()
firstState = state
pasos = 0
listaEstados.append(state)
print("First State: ", firstState)

while reward != 20 and not truncated:
    action = np.argmax(Q[state])  # Selecciona la mejor acción conocida
    state2, reward, done, truncated, _ = env.step(action)
    pasos += 1
    print('State: ', state)
    print('pasos: ', pasos)
    G += reward
    state = state2
    listaEstados.append(state)  


# Inicializar el entorno
env = gym.make('Taxi-v3', render_mode='human')
estado, _ = env.reset()
env.unwrapped.s = estado  # Estado inicial definido

# Función para decodificar el estado del Taxi
def decode_taxi_state(state):
    taxi_row, taxi_col, passenger_location, destination = env.unwrapped.decode(state)
    return {'taxi_row': taxi_row, 'taxi_col': taxi_col, 'passenger_location': passenger_location, 'destination': destination}

# Mostrar el estado inicial
decoded_state = decode_taxi_state(estado)
env.render()
print("Estado inicial:", decoded_state)

# Bucle para seguir la política del agente hasta completar el episodio
done = False
state = estado
pasos = 0
total_reward = 0

import time  # Para agregar una pausa entre pasos
while not done:
    action = np.argmax(Q[state])  # Selecciona la acción óptima basada en la tabla Q
    state, reward, done, truncated, _ = env.step(action)  # Ejecuta la acción
    total_reward += reward
    pasos += 1
    
    # Renderiza el entorno en cada paso y muestra el estado decodificado
    env.render()
    decoded_state = decode_taxi_state(state)
    print(f"Paso {pasos}: Estado = {decoded_state}, Acción = {action}, Recompensa = {reward}")

    # Pausa de 0.5 segundos entre pasos para evitar sobrecargar el renderizado
    time.sleep(0.5)

print(f"Juego completado en {pasos} pasos con una recompensa total de {total_reward}.")
env.close()

print(Q)