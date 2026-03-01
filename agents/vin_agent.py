import torch
import torch.optim as optim
import numpy as np
from agents.actor_critic import ActorCritic
from agents.base_agent import BaseAgent
import copy


class RLAgent(BaseAgent):
    def __init__(self, player_id, board_size=9):
        super().__init__(player_id)
        self.board_size = board_size

        self.model = ActorCritic(board_size)
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.001)  # 0.01
        
        self.gamma = 0.99    
    
    def preprocess(self, state, player_id=None):
        # convert board to tensor
        pid = player_id if player_id is not None else self.player_id
        
        state = state * pid
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return state 

    def mask(self, state):
        mask = (state==0).astype(np.float32)  # identify legal moves and convert boolean to float
        mask = torch.FloatTensor(mask.flatten()).unsqueeze(0)
        return mask 
    

    def predict(self, state):

        state_tensor = self.preprocess(state)
        mask = self.mask(state)
        
        with torch.no_grad():
            probs, _ = self.model(state_tensor, mask)

        
        probs = probs.numpy().flatten()
        # renormalise
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()

        action_ind = np.random.choice(len(probs), p=probs)

        row = action_ind // self.board_size
        col = action_ind % self.board_size

        return (row, col)

        
    # monte carlo 
    def learn(self, trajectory, episode=0, max_episodes=10000):
        if not trajectory:
            return

        # Compute discounted returns from the end of the episode backwards
        returns = []
        G = 0
        for _, _, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Normalise returns to reduce variance
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        entropy_coeff = max(0.01 * (1 - episode / max_episodes), 0.001)
        total_loss = 0

        for (state, action, _), G in zip(trajectory, returns):
            state_tens = self.preprocess(state, player_id=1)
            mask = self.mask(state)

            probs, value = self.model(state_tens, mask)
            action_ind = action[0] * self.board_size + action[1]

            log_prob = torch.log(probs[0, action_ind] + 1e-8)
            advantage = G - value.squeeze()

            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage ** 2
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))

            total_loss += actor_loss + critic_loss - entropy_coeff * entropy

        self.optimiser.zero_grad()
        total_loss.backward()

        # clip gradients so they don't explode
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

    def learn_td(self, state, action, reward, next_state, done, episode=0, max_episodes=10000):
        state_tens = self.preprocess(state, player_id=1)
        mask = self.mask(state)

        probs, value = self.model(state_tens, mask)
        action_ind = action[0] * self.board_size + action[1]
        log_prob = torch.log(probs[0, action_ind] + 1e-8)   # log (pi(s|a))

        if done:
            target_value = torch.tensor([[float(reward)]])
        else:
            # Next state is opponent's turn 
            next_tens = self.preprocess(next_state, player_id=-1)
            next_mask = self.mask(next_state)
            with torch.no_grad():
                _, next_value = self.model(next_tens, next_mask)
            target_value = reward - self.gamma * next_value.detach()

        advantage = target_value - value

        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage ** 2

        entropy_coeff = max(0.01 * (1 - episode / max_episodes), 0.001)    # exploration probability - decays
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        loss = actor_loss + critic_loss - entropy_coeff * entropy

        self.optimiser.zero_grad()
        loss.backward()

        # prevent gradients exploding
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

    def get_frozen_copy(self):
        
        # returns copy of agent tht doesn't update
        frozen = RLAgent(player_id=-1, board_size=self.board_size)
        frozen.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        frozen.model.eval()
        return frozen

    def save(self, path):
        torch.save({
        "model": self.model.state_dict(),
        "optimiser": self.optimiser.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.model.train()
        

