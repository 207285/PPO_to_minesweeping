import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Action1(nn.Module):
    def __init__(self,input_shape=[10,10]):
        super(Action1,self).__init__()
        self.input_dim=input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.conv_layers(x).view(x.shape[0],-1)
        out = self.softmax(x)
        return out

class Action2(nn.Module):
    def __init__(self,input_shape=[10,10]):
        super(Action2,self).__init__()
        self.input_dim=input_shape[0]*input_shape[1]
        self.output_dim=(input_shape[0]+6)*(input_shape[1]+6)
        self.liner=nn.Linear(self.input_dim,512)
        self.liner2=nn.Linear(512,self.output_dim)
        self.liner3 = nn.Linear(self.output_dim,self.input_dim)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x=self.relu(self.liner(x))
        x=self.relu(self.liner2(x))
        out=self.softmax(self.liner3(x))
        return out

class Bvalue(nn.Module):
    def __init__(self):
        super(Bvalue,self).__init__()
        self.relu = nn.ReLU()
        self.liner=nn.Linear(200,256)
        self.liner2=nn.Linear(256,512)
        self.liner3 = nn.Linear(512,1)

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x=self.relu(self.liner(x))
        x=self.relu(self.liner2(x))
        out = self.liner3(x)
        return out

class PPO():
    def __init__(self,input_shape=[10,10],up_time=10,batch_size=32,a_lr=1e-5,b_lr=1e-5,gama=0.9,epsilon=0.1):
        self.up_time=up_time
        self.batch_size=batch_size
        self.gama=gama
        self.epsilon=epsilon
        self.suffer = []
        self.action = Action1(input_shape)
        self.action.to(device)
        self.bvalue = Bvalue()
        self.bvalue.to(device)
        self.acoptim = optim.Adam(self.action.parameters(), lr=a_lr)
        self.boptim = optim.Adam(self.bvalue.parameters(), lr=b_lr)
        self.loss = nn.MSELoss().to(device)
        self.old_prob = []

    def appdend(self, buffer):
        self.suffer.append(buffer)

    def load_net(self,path):
        self.action=torch.load(path)

    def get_action(self, x):
        x = x.unsqueeze(dim=0).to(device)
        ac_prob = self.action(x)

        a = Categorical(ac_prob).sample()[0]  # 按概率采样

        # values, indices = ac_prob.topk(k=15,dim=1)
        # a = Categorical(values).sample()[0]  # 按topk15概率采样
        # a = indices[0,a]

        ac_pro = ac_prob[0][a]
        return [a.item()], [ac_pro.item()]

    def update(self):
        states = torch.stack([t.state for t in self.suffer],dim=0).to(device)
        actions = torch.tensor([t.ac for t in self.suffer], dtype=torch.int).to(device)
        rewards = [t.reward for t in self.suffer]
        done=[t.done for t in self.suffer]
        old_probs = torch.tensor([t.ac_prob for t in self.suffer], dtype=torch.float32).to(device)  # .detach()

        false_indexes = [i+1 for i, val in enumerate(done) if not val]
        if len(false_indexes)>=0:
            idx,reward_all=0,[]
            for i in false_indexes:
                reward=rewards[idx:i]
                R = 0
                Rs = []
                reward.reverse()
                for r in reward:
                    R = r + R * self.gama
                    Rs.append(R)
                Rs.reverse()
                reward_all.extend(Rs)
                idx=i
        else:
            R = 0
            reward_all = []
            rewards.reverse()
            for r in rewards:
                R = r + R * self.gama
                reward_all.append(R)
            reward_all.reverse()
        Rs = torch.tensor(reward_all, dtype=torch.float32).to(device)
        for _ in range(self.up_time):
            self.action.train()
            self.bvalue.train()
            for n in range(max(10, int(10 * len(self.suffer) / self.batch_size))):
                index = torch.tensor(random.sample(range(len(self.suffer)), self.batch_size), dtype=torch.int64).to(device)
                v_target = torch.index_select(Rs, dim=0, index=index).unsqueeze(dim=1)
                v = self.bvalue(torch.index_select(states, 0, index))
                adta = v_target - v
                adta = adta.detach()
                probs = self.action(torch.index_select(states, 0, index))
                pro_index = torch.index_select(actions,0,index).to(torch.int64)

                probs_a = torch.gather(probs, 1, pro_index)
                ratio = probs_a / torch.index_select(old_probs, 0, index).to(device)
                surr1 = ratio * adta
                surr2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * adta.to(device)
                action_loss = -torch.mean(torch.minimum(surr1, surr2))
                self.acoptim.zero_grad()
                action_loss.backward(retain_graph=True)
                self.acoptim.step()
                bvalue_loss = self.loss(v_target, v)
                self.boptim.zero_grad()
                bvalue_loss.backward()
                self.boptim.step()
        self.suffer = []
