import torch
from torch.distributions.categorical import Categorical
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from pyecharts.charts import Line
from env import Minesweeper
from ppo import PPO
import time


def get_a(a,x_idx,y_idx):
    if x_idx > y_idx:
        x = a // x_idx
        y = a % x_idx
    else:
        x = a // y_idx
        y = a % y_idx
    return [x, y]

def test_get_action(x1, net,x_idx,y_idx):
    x = x1.unsqueeze(dim=0)
    ac_prob = net(x)
    values, indices = ac_prob.topk(k=15,dim=1)
    a = Categorical(values).sample()[0]  # 按概率采样
    a = indices[0,a].item()
    ac_prob=ac_prob.detach().numpy()
    ac_prob=ac_prob.reshape([1,10,10])

    if x_idx > y_idx:
        x = a // x_idx
        y = a % x_idx
    else:
        x = a // y_idx
        y = a % y_idx
    return [x, y],ac_prob

def mian(times,x,y,mine_num):
    env=Minesweeper(grid_width=x,grid_height=y,mine_count=mine_num,window=False)
    net=PPO(input_shape=[x,y],up_time=up_time,batch_size=batch_size,a_lr=a_lr,b_lr=b_lr,gama=gama,epsilon=epsilon)
    # path='net_model1.pt'
    # net.load_net(path)
    Rs=[]
    for i in range(times):
        with tqdm(total=epoch, desc='Iteration %d' % i) as pbar:
            for e in range(epoch):
                env.reset()
                s=torch.tensor(env.get_status(),dtype=torch.float32)
                while env.condition and env.t<51:
                    a,a_p=net.get_action(s)
                    at=get_a(a[0],x,y)
                    [s_t,r,d]=env.update(at)
                    buffer=Transition(s,a,a_p,r,d)
                    net.appdend(buffer)
                    s=s_t
                R=np.array(env.R).sum()
                Rs.append(R)
                if len(net.suffer)>batch_size:
                    net.update()
                pbar.set_postfix({'return': '%.2f' % R})
                pbar.update(1)

    torch.save(net.action,'net_model.pt')
    Re=[]
    for i in range(int(len(Rs)/50)):
        idx=i*50
        Re.append(sum(Rs[idx:idx+50])/50)
    x=[str(i) for i in range(len(Re))]
    line=Line()
    line.add_xaxis(xaxis_data=x)
    line.add_yaxis(y_axis=Re,series_name='Recall')
    line.render('result.html')


def test(path,x=10,y=10,mine_num=10):
    env = Minesweeper(grid_width=x, grid_height=y, mine_count=mine_num)
    net = torch.load(path)
    device = torch.device("cpu")
    net = net.to(device)
    s = torch.tensor(env.get_status(), dtype=torch.float32)
    a_p = 0
    for i in range(10):
        while env.condition:
            a, a_p = test_get_action(s, net, x_idx=x, y_idx=y)
            [s_t, r, d] = env.agengt_run(a)
            time.sleep(1.)
            s = s_t
        env.reset()

batch_size=32
a_lr=0.0001
b_lr=0.002
gama=0.995
epsilon=0.2
up_time=10
epoch=50

Transition = namedtuple('Transition', ['state', 'ac', 'ac_prob', 'reward', 'done'])

if __name__=='__main__':
    mian(times=100,x=10,y=10,mine_num=10)

    # path='net_model_3.pt'
    # test(path=path, x=10, y=10, mine_num=10)



