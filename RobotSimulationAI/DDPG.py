# %%
import json
import websockets, asyncio
import threading

class WaitableQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        self.event = threading.Event()

    def put(self, item):
        super().put_nowait(item)
        self.event.set()

    def get(self,timeout=3):
        if self.event.wait(timeout):
            res = super().get_nowait()
            if super().empty():
                self.event.clear()
            return res
        else:
            raise TimeoutError("Environement is not responding.")

# the server
class Server:
    def __init__(self):
        self.inQueue = WaitableQueue()
        self.outQueue = WaitableQueue()
        self.debug = True
        self.ws = None

    def start(self):
        threading.Thread(target=self.message_sender_loop).start()
        asyncio.run(self.main())

    async def main(self):
        try:
            async with websockets.serve(self.echo, "localhost", 8765):
                await asyncio.Future()  # run forever
        except websockets.exceptions.ConnectionClosedError as e: print(e)

    async def echo(self,websocket):
        self.ws = websocket
        print('connect')
        #asyncio.create_task(self.message_sender_loop())
        async for message in websocket:
            try:
                self.recv(json.loads(message))
            except json.decoder.JSONDecodeError:
                self.recv(message)

    def recv(self,message):
        self.inQueue.put(message)
        if self.debug:
            print("recv: ",message)
    
    def send(self,command:str, content):
        self.outQueue.put({'command':command,'content':content})

    def message_sender_loop(self):
        while True:
            try:
                message = self.outQueue.get(None)
                asyncio.run(self.ws.send(json.dumps(message, indent=4)))
            except websockets.exceptions.ConnectionClosedError:
                print("Connection closed")
                break
            except Exception as e:
                print(e)
                break
                
# start the server in a separate thread to avoid blocking
import threading
server = Server()
t=threading.Thread(target=server.start)
t.start()

# the interface to the server
class WSManager:
    def __init__(self,server:Server):
        self.debug = False
        self.server = server

#server.send("action",{"voltage":[1,0,0,0,100,200,100,100]})

# %%
import numpy as np
import torch
from utils import Stat
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list(list_of_lists)
    if hasattr(list_of_lists[0], '__iter__'):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list(list_of_lists[:1]) + flatten(list_of_lists[1:])
def decomposeCosSin(angle):
    return [np.cos(angle), np.sin(angle)]
def processFeature(state:dict,targetPos):
    feature = []
    feature.append(state['baseLinkPos']['x']-targetPos[0].item())
    feature.append(state['baseLinkPos']['y']-targetPos[1].item())
    feature.append(decomposeCosSin(state['baseLinkOrientation']))
    feature.append(state['baseLinkVelocity']['x'])
    feature.append(state['baseLinkVelocity']['y'])
    feature.append(state['baseLinkAngularVelocity'])
    feature.append(decomposeCosSin(state['wheelBaseOrientation']))
    feature.append(state['wheelSpeed'])
    feature = flatten(feature)
    return feature

# %%
from torch import nn
class Q(nn.Module):
    def __init__(self,state_size,action_size,hidden_size):
        super(Q, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc = nn.Sequential(
        nn.Linear(state_size+action_size,hidden_size),
        nn.LeakyReLU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size,hidden_size),
        nn.LeakyReLU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size,1)
        )

    def forward(self,state,action):
        return self.fc(torch.cat([state,action],dim=1))

class Policy(nn.Module):
    def __init__(self,state_size,action_size,hidden_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc = nn.Sequential(
        nn.Linear(state_size,hidden_size),
        nn.LeakyReLU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size,hidden_size),
        nn.LeakyReLU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size,action_size)
        )

    def forward(self,state):
        return self.fc(state)

import random, ou
class Environment:
    def __init__(self,ws_server : Server,device = 'cpu'):
        self.ws = ws_server
        self.replayBuffer = []
        self.t = 0
        self.t_episode = 0
        self.device = device
        self.prevState = None
        self.prevAction = None
        self.pos = None
        self.targetPos = None
        self.ouNoise = ou.ND_OUNoise(8,0, 0.1, 1, 0, 100)
        self.noiseIntensity = 0.5
        self.targetRelPos = torch.tensor([0.,-3.])

    def restartEpisode(self):
        self.pos = torch.tensor([0.,0.])
        self.targetPos = self.pos + self.targetRelPos
        self.t_episode = 0
        self.prevState = None
        self.ws.send("new target",{"pos":{'x':self.targetPos[0].item(),'y':0, 'z':self.targetPos[1].item()}})
        self.ws.send("pos",{'x':0,'y':0, 'z':0})

    def calculateReward(self,pos,targetPos):
        return -torch.dist(pos,targetPos)

    def terminateCondition(self,pos,targetPos):
        return torch.dist(pos,targetPos)<0.5 or torch.dist(pos,targetPos)>10

    def getPos(self,state):
        return torch.tensor([state['baseLinkPos']['x'],state['baseLinkPos']['y']],dtype=torch.float32)

    def update(self, policy: torch.nn.Module):
        raw_state = None
        reward = None
        while not server.inQueue.empty():
            message = server.inQueue.get()
            if message['command'] == 'state':
                raw_state = message['content']
        if raw_state:
            # If the environment returns a state, the step is finnished.
            self.pos = self.getPos(raw_state)
            if self.t_episode > 100 or self.t == 0 or self.terminateCondition(self.pos,self.targetPos):
                self.restartEpisode()
                self.ws.send("require state",None)
                self.t+=1
                return
            state = torch.tensor(processFeature(raw_state,self.targetPos),dtype=torch.float32).to(self.device)
                
            # Add the experience to the replay buffer.
            if self.t_episode > 0: # Skip the first step.
                reward = self.calculateReward(self.pos,self.targetPos) - self.calculateReward(self.prevPos,self.targetPos)#-(torch.max(torch.zeros_like(self.prevAction),(torch.abs(self.prevAction)-2000))**2).mean()*0.001
                self.replayBuffer.append((state,self.prevAction,reward,self.prevState))
                if len(self.replayBuffer) > 5000:
                    self.replayBuffer.pop(random.randint(0,len(self.replayBuffer)-1))
            
            # Give the new action to enable the environment to continue on the next step.
            with torch.no_grad():
                policy.eval()
                action = policy(state).detach().cpu()
                action += self.ouNoise.__next__()*self.noiseIntensity
                #action[5]=action[6]=action[7]=action[4]
                action = torch.clamp(action,-2000,2000)
            self.ws.send("action",{"voltage":list(action.detach().numpy().tolist())})

            
            self.t+=1
            self.t_episode += 1
            self.prevState = state
            self.prevAction = action
            self.prevPos = self.getPos(raw_state)

        
        return reward

    def sampleExperience(self,batch_size):
        ns,a,r,s = zip(*random.sample(self.replayBuffer,batch_size))
        return torch.stack(ns),torch.stack(a),torch.stack(r),torch.stack(s)

#env = Environment(server,device)
from torch.nn import functional as F
def soft_update_target(target:nn.Module, source:nn.Module,tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(
            (1. - tau) * t.data + tau * s.data)

# %%
stat = Stat(1000)

# %%
device = 'cuda'
env = Environment(server,device)
tau = 1-0.5**(1/1000)
gamma = 0.5**(1/50)
batch_size = 128
q = Q(state_size=19,action_size=8,hidden_size=512)
q_target = Q(state_size=19,action_size=8,hidden_size=512)
policy = Policy(state_size=19,action_size=8,hidden_size=512)
policy_target = Policy(state_size=19,action_size=8,hidden_size=512)
soft_update_target(q_target,q,1)
soft_update_target(policy_target,policy,1)
t=0


# %%
optimQ = torch.optim.Adam(q.parameters(),lr=0.001)
optimPolicy = torch.optim.Adam(policy.parameters(),lr=0.0001)

# %%
env.targetRelPos = torch.tensor([-2,-2],dtype=torch.float32)
env.noiseIntensity = 0.1
gamma = 0.5**(1/50)

# %%


server.debug = False
q.train()
policy.train()
q_target.eval()
policy_target.eval()

q.to(device)
policy.to(device)
q_target.to(device)
policy_target.to(device)

policy_loss = torch.tensor(torch.nan)

import time
# Fill the replay buffer with random experiences.
while len(env.replayBuffer) < batch_size+1:
    reward = env.update(policy)

# Training.
while(True):
    reward = env.update(policy)
    if reward is not None:
        stat.add('reward',reward.mean().item())
        #print(env.t_episode, reward)

    new_state, action, reward, old_state = env.sampleExperience(batch_size)
    new_state = new_state.to(device)
    old_state = old_state.to(device)
    action = action.to(device)
    reward = reward.to(device)

    q_target.eval()
    policy_target.eval()
    
    with torch.no_grad():
        action_ = policy_target(new_state)
        action_ = torch.clamp(action_,-2000,2000)
        new_value = q_target(new_state,action_).detach()
        
        gamma_normalizer = 1/(1-gamma) 
        target_value = (reward.unsqueeze(1) + gamma*new_value)/gamma_normalizer
    
    #target_value = reward.unsqueeze(1)
    q.train()
    policy.train()

    # Update the Q network.
    q_loss = F.mse_loss(q(old_state,action),target_value)
    optimQ.zero_grad()
    q_loss.backward()
    optimQ.step()
    
    if q_loss.item()<50:
        # Update the policy network.
        q.eval()
        action = policy(old_state)
        voltage_penalty = (torch.max(torch.zeros_like(action),(torch.abs(action)-2000))**2).mean()*1
        policy_loss = -q(old_state,action).mean() + voltage_penalty
        optimPolicy.zero_grad()
        policy_loss.backward()
        optimPolicy.step()
    
    # Update the target networks.
    soft_update_target(q_target,q,tau)
    soft_update_target(policy_target,policy,tau)
    
    stat.set_epoch(t)
    stat.add('q_loss',q_loss.item())
    stat.add('policy_loss',policy_loss.item())

    t+=1


# %%
with torch.no_grad():
    action_ = policy_target(new_state)
    action_ = torch.clamp(action_,-4000,4000)
    new_value = q_target(new_state,action_).detach()
    target_value = reward.unsqueeze(1) + gamma*new_value

#target_value = reward.unsqueeze(1)
q.train()
policy.train()

# Update the Q network.
q_loss = F.mse_loss(q(old_state,action),target_value)

# %%
new_state[11].tolist()


# %% [markdown]
# 

# %%
old_state[0]

# %%
action[0]

# %%
reward[0]

# %%
server.outQueue

# %%



