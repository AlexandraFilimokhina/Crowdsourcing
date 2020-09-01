import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from gym import spaces

NUM_PICTURES = 10
PICTURES_DIR = "/Users/alexa.filimokhina/PycharmProjects/VALOR_A2C_PICTURES/image"
CSV_DIR = os.path.join(PICTURES_DIR, "data.csv")
os.chdir(PICTURES_DIR)


class CroudsorsingEnv:
    def __init__(self, render=False):

        self.render = render
        self.max_steps = NUM_PICTURES
        self.action_space = spaces.Discrete(3)
        self.__read_picture_csv()

    def reset(self):
        self.level, self.steps, self.fails = 0, 0, 0
        return self.__get_picture(level=self.level)

    def step(self, action):
        self.reward = -1
        if self.answer == action:
            self.fails = 0
            self.reward = 1
            self.level += 1
        else:
            self.fails +=1
            self.reward = -1

        #if agent fails 2 times, he goes 1 level back
        if self.fails == 2 and self.level > 1:
            self.level -= 1
        # if agent fails 3 times, he goes on the 1-st level
        elif self.fails == 3 and self.level > 1:
            self.level = 1

        self.steps += 1
        self.done = bool(self.steps == 10)

        if self.level == 10:
            next_state = self.__get_picture(level=9)
        else:
            next_state = self.__get_picture(self.level)
        self.information = {}
        return next_state, self.reward, self.done, self.information

    def __read_picture_csv(self):
        with open(CSV_DIR) as f:
            self.df = pd.read_csv(f)

    def __get_picture(self, level):
        i = np.random.randint(NUM_PICTURES)
        self.answer = self.df[f'level_{level + 1}'][i]
        state = torch.tensor(np.array(Image.open(f'image_{level + 1}_#_{i}/FINAL.png'))).float()
        state = state.unsqueeze(0).permute(0, 3, 1, 2)
        return state

if __name__ == '__main__':
    env = CroudsorsingEnv()
    state = env.reset()