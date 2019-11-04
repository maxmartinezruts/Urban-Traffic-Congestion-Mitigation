import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
import controller as ct
import car as cr
import visual as vs
# %matplotlib inline



use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

num_envs = 16
env_name = "Pendulum-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk
#

envs = [make_env() for i in range(num_envs)]
# envs.step([0,1,1,1,1])
print(envs)
# envs = SubprocVecEnv(envs)

env = gym.make(env_name)
print(env)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        # print('x',x)
        value = self.critic(x)
        mu = self.actor(x)

        std = self.log_std.exp().expand_as(mu)
        # print('mu',mu, 'std',std, 'log',self.log_std)
        dist = Normal(mu, std)
        return dist, value


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def test_env(model,controller,vis=False):
    state = controller.reset()
    # if vis: env.render()
    done = False
    total_reward = 0
    i = 100
    for k in range(2000):
        controller.cars.append(cr.Car(controller))

    while not done and i<5:
        i+=1
        # print(state,'1')
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print(state,'2')

        dist, _ = model(state)
        # print(dist.sample,dist.sample().cpu().numpy())
        # next_state, reward, done, _ = controller.step(dist.sample().cpu().numpy()[0])
        next_state, reward, done, _ = controller.step(dist.sample().cpu().numpy())

        if i % 10 == 0:
            controller.cars.append(cr.Car(controller))
        if i % 500 == 0:

            vs.draw(controller)

        state = next_state
        # if vis: env.render()
        total_reward += reward[0]
    print('Total reward', total_reward)

    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    # print('rewards', len(rewards))
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        # print('----------')
        # print(delta,'rewardsstep',rewards[step], 'valuesstep', values[step+1], 'maskt',masks[step], 'val', values[step])
        returns.insert(0, gae + values[step])
        # print('Gae', gae, values[step])
        # print('mask',masks[step]*gae)
        # print(gae,values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer,clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            # print('ppo_update',state, model.actor[0].weight)
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            # print('Ratio',ratio, value, return_)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            # print('Loss',loss, critic_loss, actor_loss, entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Actor weight',model.actor[0].weight)
            # print('Gradients',model.actor[0].weight.grad)

# num_inputs  = envs.observation_space.shape[0]
# num_outputs = envs.action_space.shape[0]


# Train your model

def train():
    num_inputs = 16*9+4*9
    num_outputs = 1

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    num_steps = 5
    mini_batch_size = 5
    ppo_epochs = 4
    threshold_reward = 1000000000000

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    controller = ct.Controller()

    # print(model.log_std)
    max_frames = 1500000
    frame_idx = 0
    test_rewards = []

    # state = envs.reset()
    state = controller.reset()
    early_stop = False

    while frame_idx < max_frames and not early_stop:

        # Probability log distribution
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        controller.reset()

        for _ in range(num_steps):

            if _ == 0:
                for k in range(800):

                    controller.cars.append(cr.Car(controller))


                print(frame_idx)
            state = torch.FloatTensor([controller.get_state()]).to(device)
            dist, value = model(state)
            action = dist.sample()
            # print('Input:',state)
            # print('Output:',action)
            next_state, reward, done, _ = controller.step(action)
            # next_state, reward, done, _ = envs.step(action)
            next_state =  torch.FloatTensor([next_state]).to(device)
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            print('Reward;', reward)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(device))

            # masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

            if frame_idx % 100 == 0:
                print('start')
                controller.reset()

                test_reward = np.mean([test_env(model,controller) for _ in range(5)])
                test_rewards.append(test_reward)
                # plot(frame_idx, test_rewards)
                print(test_rewards)
                if test_reward > threshold_reward: early_stop = True
                controller.reset()


        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, model, optimizer)
    # print(model.actor[0].weight)
    # print(model.critic[0].weight)

    return model, controller

print('fsa')
#
# from itertools import count
#
# max_expert_num = 50000
# num_steps = 0
# expert_traj = []
#
#
# # Test your model
# for i_episode in count():
#     state = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         state = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist, _ = model(state)
#         action = dist.sample().cpu().numpy()[0]
#         next_state, reward, done, _ = env.step(action)
#         state = next_state
#         total_reward += reward
#         expert_traj.append(np.hstack([state, action]))
#         num_steps += 1
#
#     print("episode:", i_episode, "reward:", total_reward)
#
#     if num_steps >= max_expert_num:
#         break
#
# expert_traj = np.stack(expert_traj)
# print()
# print(expert_traj.shape)
# print()
# np.save("expert_traj.npy", expert_traj)