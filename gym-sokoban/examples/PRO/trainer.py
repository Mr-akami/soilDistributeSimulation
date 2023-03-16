# 必要なライブラリのインポート．
from abc import ABC, abstractmethod
import os
import glob
from collections import deque
from time import time
from time import sleep
from datetime import timedelta
import pickle
from base64 import b64encode
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from matplotlib import animation

# Gymの警告を一部無視する．
gym.logger.set_level(40)

def wrap_monitor(env):
    """ Gymの環境をmp4に保存するために，環境をラップする関数． """
    return gym.wrappers.Monitor(env, '/tmp/monitor', video_callable=lambda x: True, force=True)

# def play_mp4():
#     """ 保存したmp4をHTMLに埋め込み再生する関数． """
#     path = glob.glob(os.path.join('/tmp/monitor', '*.mp4'))[0]
#     mp4 = open(path, 'rb').read()
#     url = "data:video/mp4;base64," + b64encode(mp4).decode()
#     return HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % url)

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=3):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        # 環境の乱数シードを設定する．
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}

        # データ収集を行うステップ数．
        self.num_steps = num_steps
        # 評価の間のステップ数(インターバル)．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes
        self.writer = SummaryWriter('./logs')

    def train(self):
        """ num_stepsステップの間，データ収集・学習・評価を繰り返す． """

        # 学習開始の時間
        self.start_time = time()
        # エピソードのステップ数．
        t = 0

        # 環境を初期化する．
        state = self.env.reset()
        
        total_reward = 0

        for steps in range(1, self.num_steps + 1):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t, reward = self.algo.step(self.env, state, t, steps)
            total_reward += reward

            # アルゴリズムが準備できていれば，1回学習を行う．
            if self.algo.is_update(steps):
                self.algo.update()

            # 一定のインターバルで評価する．
            if steps % self.eval_interval == 0:
                self.evaluate(steps)
        self.writer.close()

    def evaluate(self, steps):
        """ 複数エピソード環境を動かし，平均収益を記録する． """

        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0

            while (not done):
                action = self.algo.exploit(state)
                # print('action:',action)
                action = np.argmax(action)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')
        self.writer.add_scalar('return', mean_return, steps)

    def visualize(self):
        # """ 1エピソード環境を動かし，mp4を再生する． """
        env = wrap_monitor(gym.make(self.env.unwrapped.spec.id))
        state = env.reset()
        done = False
        total_reward = 0
        frames = []

        while (not done):
            sleep(0.001)
            env.render(mode='human')
            frames.append(env.render(mode='rgb_array'))
            action = self.algo.exploit(state)
            action = np.argmax(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        print('Total reward:', total_reward)
        del env
        # return play_mp4()
        
        plt.figure(figsize=(8, 8), dpi=50)
        # patch = plt.imshow(frames[0], cmap='gray')
        patch = plt.imshow(frames[0])
        plt.axis('off')
    
        def animate(i):
            patch.set_data(frames[i])
    
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1000)
        # display(HTML(anim.to_jshtml(default_mode='once')))
        anim.save("output.gif", writer="imagemagick")
        plt.show()
        plt.close()
        print('open file directly')

    def plot(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))