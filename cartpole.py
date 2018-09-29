import gym
import random
import sys
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from collections import deque
from keras.layers import Dense


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import xlsxwriter


#Agent that usees qnn with replays and target networks to figure out best action for cartpole
class DeepAgent:
    def __init__(self, stateSize, actionSize):
        
        # get size of state and action
        self.stateSize = stateSize
        self.actionSize = actionSize

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.epsilon = 0.05
        self.batchSize = 50
        
        # create replay memory using deque
        self.memory = deque(maxlen=1000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()


    #Send in 4 inputs and get 2 outputs for which direction to move
    def build_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.stateSize, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.actionSize, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    #Choose action using epsilon greedy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    #Save to memory for replay use later
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        

    #Randomly sample from memory or pick most recent for no replay
    def train_model(self):
        if len(self.memory) < self.batchSize:
            return
        
        #get a sample
        batch = random.sample(self.memory, self.batchSize)
        
        #take the most recent
        #batch = [self.memory[len(self.memory)-1]]
        
        newInput = np.zeros((self.batchSize, self.stateSize))
        newTarget = np.zeros((self.batchSize, self.stateSize))
        action = []
        done = []
        reward = []

        for i in range(self.batchSize):
            newInput[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            newTarget[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.model.predict(newInput)
        targetVal = self.target_model.predict(newTarget)

        for i in range(self.batchSize):
            # Q Learning: if done then just return reward as no future runs
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(targetVal[i]))

        #fit based on batch size, set to 1 for no replay
        self.model.fit(newInput, target, batch_size=self.batchSize, epochs=1, verbose=0)

    #Update target model, after every 2nd episode
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())



runs = 1000

if __name__ == "__main__":
    #CartPole v1 max is 500, CartPole v0 max is 200
    env = gym.make('CartPole-v1')
    
    # get size of state and action from environment
    stateSize = env.observation_space.shape[0]
    actionSize = env.action_space.n

    agent = DeepAgent(stateSize, actionSize)

    scores = []
    runList = []
    
    for epi in range(runs):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, stateSize])
        

        while not done:
            

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, stateSize])

            # save the sample <s, a, r, s'> to the replay
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every second episode update the target model
                if epi%2 == 0:
                    agent.update_target_model()
                
                #add to lists to use later
                scores.append(score)
                runList.append(epi)
                
                if score >= 200:
                    print("Episode:", epi," Score: ", score)
                

    workbook = xlsxwriter.Workbook('Scores.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for data in (scores):
        worksheet.write(row, col, data)
        row = row + 1
    
    workbook.close()
