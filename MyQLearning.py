import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd
import os
class QLearningTable:
    #0.8 0.9 0.9   train 3000
    #0.1 0.9 0.9   train 100
    #0.05 0.9 0.98   train 100
    def __init__(self, actions, learning_rate=0.95, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, state):

        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    np.random.randn(len(self.actions)),
                    #[0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )

            )
            print("pd", np.random.randn(len(self.actions)) )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            #state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.values.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self,s,a,r,s_,done):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s,a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def save_qtable(self,pathname):
        self.q_table.to_csv(pathname, index=True)

    def load_q_table(self,pathname):
        if os.path.getsize(pathname):
            self.q_table = pd.read_csv(pathname,index_col=0,dtype={0:str})
            #print(self.q_table)

class Gym_model:
    def __init__(self,pathname,model_name):
        self.pathname = pathname
        self.episodes          = 20
        #self.env.unwarpped
        self.env               = gym.make(model_name)
        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.env._max_episode_steps = 20000
        self.agent             =  QLearningTable(actions=list(range(self.action_size)))
        self.agent.load_q_table(pathname)
        self.reward=[]

    def change_state(self,state):
        state = state.reshape((self.state_size ,))
        deta = 0.01*(cartpole.env.observation_space.high-cartpole.env.observation_space.low)
        baseline = np.array(cartpole.env.observation_space.low)
        new_state = np.zeros(self.state_size)
        for i in range(self.state_size):
            for j in range(100):
                if baseline[i]+j*deta[i]<= state[i] and state[i]  <=  baseline[i]+(j+1)*deta[i]:
                    new_state[i] = j
                    break

        return new_state.reshape((1,self.state_size ))


    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                state = self.change_state(state)
                done = False
                index = 0
                count_reward=0
                while not done:
                    self.env.render()

                    action = self.agent.choose_action(str(state))

                    next_state, reward, done, _ = self.env.step(action)
                    count_reward+=reward
                    next_state = np.reshape(next_state, [1, self.state_size])
                    next_state = self.change_state(next_state)
                    self.agent.learn(str(state),action,reward,str(next_state),done)#  if close it,the training will be end
                    state = next_state
                    index += 1

                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.reward.append(count_reward)
        finally:
            self.agent.save_qtable(self.pathname)

    def get_mean_variance(self):
        variance = np.std(self.reward, ddof = 1)
        mean = np.mean(self.reward)
        return mean,variance

if __name__ == "__main__":
    model_name = 'MountainCar-v0'
    model = "data/MountainCar.txt"
    for i in range(300):
       cartpole = Gym_model(model,model_name)
       cartpole.run()
       print("----------CartPole--------- : ", i ," ",cartpole.get_mean_variance())


