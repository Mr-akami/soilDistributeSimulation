import random

# import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import gym
import gym_sokoban
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def make_env(noop_max=30, skip=4, width=84, height=84, grayscale=True):
    env = gym.make('Sokoban-v0')
    # env = NoopResetEnv(env, noop_max=noop_max)
    # env = MaxAndSkipEnv(env, skip=skip)
    # env = EpisodicLifeEnv(env)
    # env = FireResetEnv(env)
    # env = WarpFrame(env, width=width, height=height, grayscale=grayscale)
    # env = ClipRewardEnv(env)
    # env = TorchFrame(env)
    return env

env = make_env()

"""
   Prioritized Experience Replayを実現するためのメモリクラス.
"""
class PrioritizedReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0
    
    def __len__(self):
        return len(self.buffer)

    # 経験をリプレイバッファに保存する． 経験は(obs, action, reward, next_obs, done)の5つ組を想定    
    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        # 優先度は最初は大きな値で初期化しておき, 後でサンプルされた時に更新する
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size
    
    def sample(self, batch_size, alpha=0.6, beta=0.4):
        # 現在経験が入っている部分に対応する優先度を取り出し, サンプルする確率を計算
        priorities = self.priorities[: self.buffer_size if len(self.buffer) == self.buffer_size else self.index]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        # 上で計算した確率に従ってリプレイバッファ中のインデックスをサンプルする
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)

        # 学習の方向性を補正するための重みを計算
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        # 上でサンプルしたインデックスに基づいて経験をサンプルし, (obs, action, reward, next_obs, done)に分ける
        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])

        # あとで計算しやすいようにtorch.Tensorに変換して(obs, action, reward, next_obs, done, indices, weights)の7つ組を返す
        return (torch.stack(obs),
                torch.as_tensor(action), 
                torch.as_tensor(reward, dtype=torch.float32),
                torch.stack(next_obs), 
                torch.as_tensor(done, dtype=torch.uint8),
                indices,
                torch.as_tensor(weights, dtype=torch.float32))

    # 優先度を更新する. 優先度が極端に小さくなって経験が全く選ばれないということがないように, 微小値を加算しておく.
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-4
        

"""
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します. 
"""
class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action
        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3, stride=1),  # 1x84x84 -> 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 32x20x20 -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),  # 64x9x9 -> 64x7x7
            nn.ReLU()
        )

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(9088, 512), # conv_layersから出力されたfeatのshapeに合わせる
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(9088, 512), # conv_layersから出力されたfeatのshapeに合わせる
            nn.ReLU(),
            nn.Linear(512, n_action)
        )
    
    def forward(self, obs):
        feature = self.conv_layers(obs)
        feature = feature.view(feature.size(0), -1)  #　Flatten. 64x7x7　-> 3136 # 32 x 142 -> 4544

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return action_values

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します. 
    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action
    
    
"""
    リプレイバッファの宣言
"""
buffer_size = 100000  #　リプレイバッファに入る経験の最大数
initial_buffer_size = 10000  # 学習を開始する最低限の経験の数
replay_buffer = PrioritizedReplayBuffer(buffer_size)


"""
    ネットワークの宣言
"""
net = CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(device)
# net = CNNQNetwork((144,144,3), n_action=env.action_space.n).to(device)
target_net = CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(device)
# target_net = CNNQNetwork((144,144,3), n_action=env.action_space.n).to(device)
target_update_interval = 2000  # 学習安定化のために用いるターゲットネットワークの同期間隔


"""
    オプティマイザとロス関数の宣言
"""
optimizer = optim.Adam(net.parameters(), lr=1e-4)  # オプティマイザはAdam
loss_func = nn.SmoothL1Loss(reduction='none')  # ロスはSmoothL1loss（別名Huber loss）


"""
    Prioritized Experience Replayのためのパラメータβ
"""
beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000
# beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))


"""
    探索のためのパラメータε
"""
epsilon_begin = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
# epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))


"""
    その他のハイパーパラメータ
"""
gamma = 0.99  #　割引率
batch_size = 32
n_episodes = 500  # 学習を行うエピソード数


def update(batch_size, beta):
    obs, action, reward, next_obs, done, indices, weights = replay_buffer.sample(batch_size, beta)
    obs, action, reward, next_obs, done, weights \
        = obs.float().to(device), action.to(device), reward.to(device), next_obs.float().to(device), done.to(device), weights.to(device)

    #　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    
    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN. 
        # ① 現在のQ関数でgreedyに行動を選択し, 
        greedy_action_next = torch.argmax(net(next_obs), dim=1)
        # ②　対応する価値はターゲットネットワークのものを参照します.
        q_values_next = target_net(next_obs).gather(1, greedy_action_next.unsqueeze(1)).squeeze(1)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    #　TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    replay_buffer.update_priorities(indices, (target_q_values - q_values).abs().detach().cpu().numpy())

    return loss.item()


# TensorBoardをColab内に起動. リアルタイムに学習経過が更新されます
writer = SummaryWriter('./logs')
# %tensorboard --logdir="logs"
# ctr + shift + p でコマンドパレットを開き, "TensorBoard: Launch TensorBoard"を選択してください.

# 学習開始
step = 0
for episode in range(n_episodes):
    obs = env.reset()
    # print(obs)
    done = False
    total_reward = 0

    while not done:
        # ε-greedyで行動を選択
        action = net.act(obs.float().to(device), epsilon_func(step)) # obsにfloat()がないっぽい、env.resetについてみるか
        # 環境中で実際に行動
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        # リプレイバッファに経験を蓄積
        replay_buffer.push([obs, action, reward, next_obs, done])
        obs = next_obs
        
        # ネットワークを更新            
        if len(replay_buffer) > initial_buffer_size:
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())
        
        step += 1

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
    writer.add_scalar('Reward', total_reward, episode)

writer.close()

# 学習結果の確認
frames = []
obs = env.reset()
total_reward = 0
done = False


def display_video(frames):
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


while not done:
    # frames.append(env.render(mode='rgb_array')[:, :, 0]) # DQN本来のコード
    frames.append(env.render(mode='rgb_array'))
    # frames.append(env.get_image('human', 1))
    action = net.act(obs.float().to(device), epsilon=0.0)
    next_obs, reward, done, _ = env.step(action)
    
    total_reward += reward
    obs = next_obs
    

print('Reward: ', total_reward)
display_video(frames)