import random
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import gym
import copy

class NN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()
        self.dense1 = nn.Linear(state_size, 20)
        self.dense2 = nn.Linear(20, action_size)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class DQN:
        def __init__(self, state_size, action_size,pathname):

            self.model = NN(state_size, action_size)
            self.model2 = NN(state_size, action_size)
            self.TARGET_REPLACE_ITER = 10
            self.learn_step_counter = 0

            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=10000)
            self.GAMMA = 0.9
            self.epsilon = 0.1
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.criterion = nn.MSELoss()
            self.pathname = pathname

            if os.path.getsize(self.pathname) > 0:
                self.model = torch.load(self.pathname)
                print("the size of the model is {}".format(os.path.getsize(self.pathname)))


        def select_action(self, state):

            sample = random.random()

            if random.random() <= self.epsilon:
                return np.random.choice(list(range(self.action_size)))
            else:
                s = Variable(torch.FloatTensor(state.reshape(1, self.state_size)), volatile=True)
                out = self.model(s).max(1)[1].data[0]
                return out

        def replay(self, BATCH_SIZE):
            if len(self.memory) < BATCH_SIZE:
                return

            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.model2.load_state_dict(self.model.state_dict())
            self.learn_step_counter += 1

            batch = random.sample(self.memory, BATCH_SIZE)
            [states, actions, rewards, next_states, dones] = zip(*batch)

            state_batch = Variable(torch.FloatTensor(np.array(states)))
            action_batch = Variable(torch.LongTensor(np.array(actions)))
            reward_batch = Variable(torch.FloatTensor(rewards))
            next_states_batch = Variable(torch.FloatTensor(np.array(next_states)))


            state_action_values = self.model(state_batch).gather(1, action_batch.view(-1, 1))


            next_states_batch.volatile = True
            next_state_values = self.model2(next_states_batch).max(1)[0]
            for i in range(BATCH_SIZE):
                if dones[i]:
                    next_state_values.data[i] = 0

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
            expected_state_action_values.volatile = False

            loss = self.criterion(state_action_values, expected_state_action_values)
           # print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print("##",loss.data)
            loss_mean = np.mean(loss.data.numpy())
            return loss_mean

        def save_model(self):
            torch.save(self.model, self.pathname)

class Gym_model:
    def __init__(self,model_name,pathname):
        self.pathname = pathname
        self.sample_batch_size = 32
        self.episodes          = 20
        self.env = gym.make(model_name)
        self.env._max_episode_steps = 2000
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = DQN(self.state_size, self.action_size,pathname)
        self.reward = []

    def run(self):
        loss_list = []
        reward_list = []
        times_list = []
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                #state = np.reshape(state, [1, self.state_size])
                index = 0
                sum_reward =0
                loss =0
                done = False
                while not done:
                    self.env.render()
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.memory.append([state, action, reward, next_state, done])
                    state = next_state
                    index += 1
                    sum_reward += reward
                    state = next_state
                    if len(self.agent.memory) >= self.sample_batch_size:
                        loss += self.agent.replay(self.sample_batch_size)
                print("Episode {}# Score: {}".format(index_episode, index ))
                self.reward.append(sum_reward)
                loss_list.append([index_episode, loss/index])
                reward_list.append([index_episode, sum_reward])
                times_list.append([index_episode, index])

        finally:
            self.agent.save_model()
            #np.savetxt('list_data/3loss_list.csv', loss_list, delimiter=',')
            #np.savetxt('list_data/3reward_list.csv', reward_list, delimiter=',')
            #np.savetxt('list_data/3times_list.csv', times_list, delimiter=',')

    def get_mean_variance(self):
        variance = np.std(self.reward, ddof=1)
        mean = np.mean(self.reward)
        return mean, variance
if __name__ == "__main__":
    #model_name = 'MountainCar-v0'
    #model = "data/MountainCar_IMP_DQN.model"
    ##model = "data/CartPole_IMP_DQN.model"
    model_name = 'Acrobot-v1'
    model = "data/Acrobot_IMP_DQN.model"
    cartpole = Gym_model(model_name,model)
    cartpole.run()
    print(cartpole.get_mean_variance())


