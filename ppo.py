import numpy as np
import utils
import torch
from torch import nn
import variable as v
from torch.distributions import Categorical
from copy import deepcopy
from collections import namedtuple
from PIL import Image


class PPO:
    """
    PPO Agent
    """
    def __init__(self, ppo_init):
        torch.manual_seed(ppo_init['seed'])
        self.discount_factor = ppo_init["discount_factor"]
        self.num_action = ppo_init["num_action"]
        self.epsilon = ppo_init["epsilon"]
        self.input_image = ppo_init["input_as_image"]

        self.num_epoch = ppo_init["num_epoch"]
        self.mini_batch_size = ppo_init["mini_batch_size"]
        self.experience = namedtuple('Experience', field_names=["state", "action", "action_prob", "reward", "done"])
        self.memory = list()

        self.actor = PPOActor(ppo_init["actor"]).to(v.device)
        self.actor_old = PPOActor(ppo_init["actor"]).to(v.device)
        self.critic = PPOCritic(ppo_init["critic"]).to(v.device)

        self.random_generator = np.random.RandomState(seed=ppo_init['seed'])
        self.last_state = None
        self.last_action = None
        self.last_action_prob = None

        self.init_optimizers(critic_optimizer=ppo_init['critic']['optimizer'],
                             actor_optimizer=ppo_init['actor']['optimizer'])
        self.update_target_weights()

    def init_optimizers(self, critic_optimizer={}, actor_optimizer={}):
        self.critic.init_optimizer(critic_optimizer)
        self.actor.init_optimizer(actor_optimizer)

    def update_target_weights(self):
        self.actor_old.load_state_dict(self.actor.state_dict())

    def set_test(self):
        self.actor.eval()

    def set_train(self):
        self.actor.train()

    @staticmethod
    def preprocess_image(state_image, im_size=84):
        state_image = state_image[34:194]
        state_image = np.round(np.dot(state_image, [0.2989, 0.587, 0.114])).astype(np.float)
        state_image = np.array(Image.fromarray(state_image).resize((im_size, im_size)))
        return np.expand_dims(state_image, axis=0).astype(np.float)

    def append_experience(self, state, action, action_prob, reward, done):
        episode = self.experience(state, action, action_prob, reward, done)
        self.memory.append(episode)

    def policy(self, state):
        state = self.preprocess_image(state) if self.input_image else state
        state_tensor = utils.to_tensor(state).view((1, ) + state.shape)
        with torch.no_grad():
            action_probs = self.actor_old.predict(state_tensor).squeeze(0).cpu().numpy()
        action = self.random_generator.choice(self.num_action, p=action_probs)

        self.last_state = state
        self.last_action = action
        self.last_action_prob = action_probs[action]

        return action

    def compute_discounted_rewards(self, rewards, dones):
        discounted_reward = 0.0
        discounted_rewards = list()
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.discount_factor * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def flush_memory(self):
        old_states, old_actions, old_action_probs, rewards, dones = list(zip(*self.memory))
        discounted_rewards = self.compute_discounted_rewards(rewards, dones)

        old_states = utils.to_tensor(old_states).to(v.device)
        old_actions = utils.to_tensor(old_actions).to(v.device)
        old_action_probs = utils.to_tensor(old_action_probs).to(v.device)
        discounted_rewards = utils.to_tensor(discounted_rewards).to(v.device)

        self.memory = list()

        return old_states, old_actions, old_action_probs, discounted_rewards

    def update(self, reward, done):
        self.append_experience(self.last_state, self.last_action, self.last_action_prob, reward, done)

        if len(self.memory) < self.mini_batch_size:
            return

        old_states, old_actions, old_action_probs, discounted_rewards = self.flush_memory()
        for _ in range(self.num_epoch):
            new_action_probs, entropy = self.actor.retrieve_action_probs(old_states, old_actions)

            states_values = self.critic.predict(old_states)
            r = new_action_probs / old_action_probs
            advantages = discounted_rewards - states_values

            self.actor.update(advantages, r, entropy, self.epsilon)
            self.critic.update(states_values, discounted_rewards)

        self.update_target_weights()


class PPOActor(torch.nn.Module):
    """
    Actor using Proximal Policy Optimization with Adaptive KL penalty along with SGD to optimize its policy
    """
    def __init__(self, actor_init):
        super(PPOActor, self).__init__()
        net = actor_init['network_init']
        self.entropy_learning_rate = actor_init['entropy_learning_rate']
        self.optimizer = None
        self.loss_history = list()
        self.input_image = actor_init["input_as_image"]

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=net["cl1_channels"],
                      out_channels=net["cl2_channels"],
                      kernel_size=net["cl1_kernel_size"],
                      stride=net["cl1_stride"],
                      padding=net["cl1_padding"]
                      ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=net["cl2_channels"],
                      out_channels=net["out_channels"],
                      kernel_size=net["cl2_kernel_size"],
                      stride=net["cl2_stride"],
                      padding=net["cl2_padding"]
                      )
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.l1 = nn.Linear(net["l1_shape"], net["l2_shape"])
        self.l2 = nn.Linear(net["l2_shape"], net["l3_shape"])
        self.o = nn.Linear(net["l3_shape"], net["o_shape"])

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        if self.input_image:
            x = self.conv_layer(x)
            x = x.view(x.shape[0], -1)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.softmax(self.o(x))

        return x

    def predict(self, state):  # return action's distribution
        return self.forward(state)

    def retrieve_action_probs(self, states, actions):  # return probability for each action
        action_probs = self.forward(states)
        distributions = Categorical(action_probs)

        action_logprobs = distributions.log_prob(actions)
        dist_entropy = distributions.entropy()

        return torch.exp(action_logprobs), dist_entropy

    def update(self, r, advantages, entropy, epsilon):
        loss = -torch.min(
            r * advantages,
            torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantages
        ).mean() - self.entropy_learning_rate * torch.sum(entropy)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.loss_history.append(loss.item())


class PPOCritic(torch.nn.Module):
    """
    Value based Critic used to estimate V(s) in order to compute A(s).
    """
    def __init__(self, critic_init):
        super(PPOCritic, self).__init__()
        net = critic_init['network_init']
        self.loss = torch.nn.MSELoss()
        self.optimizer = None
        self.loss_history = list()
        self.input_image = critic_init["input_as_image"]

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=net["cl1_channels"],
                      out_channels=net["cl2_channels"],
                      kernel_size=net["cl1_kernel_size"],
                      stride=net["cl1_stride"],
                      padding=net["cl1_padding"]
                      ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=net["cl2_channels"],
                      out_channels=net["out_channels"],
                      kernel_size=net["cl2_kernel_size"],
                      stride=net["cl2_stride"],
                      padding=net["cl2_padding"]
                      )
        )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.l1 = nn.Linear(net["l1_shape"], net["l2_shape"])
        self.l2 = nn.Linear(net["l2_shape"], net["l3_shape"])
        self.o = nn.Linear(net["l3_shape"], 1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        if self.input_image:
            x = self.conv_layer(x)
            x = x.view(x.shape[0], -1)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.o(x)

        return x

    def predict(self, states):
        return self.forward(states)

    def update(self, states_values, discounted_rewards):
        loss = self.loss(states_values.squeeze(), discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())