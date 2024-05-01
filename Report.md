# <p style="text-align: center;">[S24] Reinforcement Learning and Intelligent Agents</p>
## <p style="text-align: center;">Project report (Atari Bank Heist)</p>
### <p style="text-align: center;">Team introduction</p>
#### Team 13
#### Team members:
* Aleksandr Vashchenko.
* Grigoriy Nesterov.

### <p style="text-align: center;">About the game</p>
<p style="text-align:center;"><img src="https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/bank_heist.gif" alt="drawing" width="300"/></p>
* Bank heist is a game where the player controls a car. His task is to collect bags of money in the maze and move to the next level.
* The player has 4 lives at the beginning of the game. After the player picks up a bag, a police car appears in its place. The player needs to avoid the policemen, otherwise he will be caught and his life will be spent.
* The player can spawn dynamite to blow up the police officers. But the player needs to be careful because it is possible to blow himself up.
* The player's car also has fuel. If it runs out of fuel, it is the end of the game.
* The player can turn up, down, right, left, shoot and do nothing.
* The number of points is the amount of money the player has collected.

### <p style="text-align: center;">Preprocessing</p>
<p style="text-align:center;"><img src="https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/output.png" alt="drawing" width="700"/></p>

#### <p style="text-align: center;"> Left image - unprocessed image, right image - preprocessed matrix of states.</p>
From environment, we get an observation of the current step (grayscale image of the game 210x160 pixels represented by numpy array with uint8 data type).
We cut this image to only playable field of the game, and then it convert to observation matrix of size 68x68 with encoding of each element.<br>
This greatly increased speed and performance of our model.

### <p style="text-align: center;">DDQN and model inside it</p>
For this game we decided use Double Q-learning (DDQN) where we soft update our target network with next formula: 
<p style="text-align: center;">$\Theta_{\text{target}} \leftarrow \tau \cdot \Theta_{\text{net}} + (1 - \tau) \cdot \Theta_{\text{target}}$</p>
Thank to preprocessing we can use pretty simple neural network for our game.

```
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(1 * 68 * 68, 2048) 
        self.layer2 = nn.Linear(2048, 512)
        self.layer3 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), 1)
```

### <p style="text-align: center;">Graphs and results</p>
<p style="text-align:center;"><img src="https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/duration_plot.png" alt="drawing" width="700"/></p>

#### <p style="text-align: center;">Graph of reward/episode</p>
<p style="text-align:center;"><img src="https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/reward_plot.png" alt="drawing" width="700"/></p>

#### <p style="text-align: center;">Graph of duration/episode</p>

##### <p style="text-align: center;">Video demonstration of the model</p>
[![Youtube link](https://img.youtube.com/vi/apy8TOZutRQ/0.jpg)](https://www.youtube.com/watch?v=apy8TOZutRQ)