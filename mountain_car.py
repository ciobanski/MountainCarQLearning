import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

class QLAgent:
    def __init__(self, env, num_states, alpha, gamma, epsilon, Q=None) -> None:
        # Inițializează agentul Q-learning cu mediul specificat și parametrii corespunzători
        self.env = env
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = Q if Q is not None else np.random.uniform(low=-2, high=0,
                                                           size=(num_states, num_states, env.action_space.n))
        # Definește intervalele discrete ale spațiului stărilor
        min_p, min_v = self.env.observation_space.low
        max_p, max_v = self.env.observation_space.high
        self.p_bins = np.linspace(start=min_p - 0.1, stop=max_p, num=self.num_states)
        self.v_bins = np.linspace(start=min_v - 0.1, stop=max_v, num=self.num_states)

    def get_d_state(self, state) -> tuple:
        # Convertește starea continuă într-o stare discretă
        p, v = state
        return (np.digitize(p, self.p_bins, right=True) - 1, np.digitize(v, self.v_bins, right=True) - 1)

    def choose_action(self, state) -> int:
        # Selectează acțiunea utilizând politica epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            p, v = self.get_d_state(state)
            return np.argmax(self.Q[p, v])

    def update_Q(self, state, action, reward, next_state) -> None:
        # Actualizează valoarea Q utilizând ecuația Bellman
        p, v = self.get_d_state(state)
        next_p, next_v = self.get_d_state(next_state)
        max_Q = np.max(self.Q[next_p, next_v])
        sample = reward + self.gamma * max_Q
        self.Q[p, v][action] = (1 - self.alpha) * self.Q[p, v][action] + self.alpha * sample

    def run_episode(self, epsilon_decay=0.99) -> tuple:
        # Rulează un singur episod al agentului Q-learning
        episode_rewards = []
        steps_list = []  # Colectează informații despre numărul de pași

        state, *_ = self.env.reset()
        state = tuple(state)
        done = False
        steps = 0  # Inițializează numărul de pași

        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.env.step(action)
            self.update_Q(state, action, reward, tuple(next_state))
            state = next_state

            episode_rewards.append(reward)
            steps += 1  # Incrementarea numărului de pași

            # Redare explicită după fiecare pas
            self.env.render()

        # Scăderea valorii epsilon la sfârșitul fiecărui episod
        self.epsilon *= epsilon_decay

        # Calculează și returnează recompensa medie și numărul de pași
        avg_reward = np.mean(np.array(episode_rewards))
        steps_list.append(steps)

        return self.Q, episode_rewards, avg_reward, steps

    def plot_statistics(self, episode_rewards, epsilon_values, avg_rewards, steps_list):
        # Afișează diferitele statistici după antrenament
        plt.figure(figsize=(15, 10))

        # Afișează tendințele valorilor Q
        q_values = np.array(self.Q).mean(axis=2)  # Medie peste acțiuni
        plt.subplot(121)
        plt.imshow(q_values, cmap='viridis', origin='lower', extent=[0, self.num_states, 0, self.num_states])
        plt.title('Valorile medii Q')
        plt.xlabel('Index poziție')
        plt.ylabel('Index viteză')
        plt.colorbar()

        # Afișează numărul de pași per episod
        plt.subplot(122)
        plt.plot(steps_list, label='Număr de pași per episod')
        plt.title('Număr de pași per episod')
        plt.xlabel('Episod')
        plt.ylabel('Pași')
        plt.legend()

        # Afișează graficele
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Definește opțiunile pentru episoade și mediul înconjurător
    num_episodes = 200
    display_interval = num_episodes  # Afișează doar la sfârșitul tuturor episoadelor
    num_states = 20  # Dimensiunea spațiului discret al stărilor (crescut pentru învățare mai bună)
    alpha = 0.2  # Rata de învățare
    gamma = 0.99  # Factorul de reducere
    epsilon = 0.1  # Parametrul de explorare

    # Rulează episoadele
    Q = None
    episode_rewards = []
    epsilon_values = []
    avg_rewards = []
    steps_list = []  # Colectează informații despre numărul de pași

    total_start_time = time.time()  # Înregistrează momentul de start pentru toate episoadele

    for episode in range(1, num_episodes + 1):
        # Afișează doar dacă este un episod de afișare
        display = episode % display_interval == 0
        env = gym.make('MountainCar-v0', render_mode='human' if display else None)
        agent = QLAgent(env, num_states, alpha, gamma, epsilon, Q)

        # Rulează episodul
        print(f'Învățând momentan, sunt la episodul {episode}...')
        start_time = time.time()
        Q, rewards, avg_reward, steps = agent.run_episode()
        episode_rewards.extend(rewards)
        epsilon_values.append(agent.epsilon)
        avg_rewards.append(avg_reward)
        steps_list.append(steps)

        end_time = time.time()
        delta = end_time - start_time
        print(f'Obiectiv atins în {delta:.4f} secunde')
        env.close()

    # După toate episoadele, afișează statisticile
    agent.plot_statistics(episode_rewards, epsilon_values, avg_rewards, steps_list)
    print('Toate episoadele complete')
