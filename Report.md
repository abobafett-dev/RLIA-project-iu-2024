# <center>[S24] Reinforcement Learning and Intelligent Agents
## <center>Project report (Atari Bank Heist)
### <center>Team introduction
#### Team 13
<b>Team members:</b>

* Aleksandr Vashchenko.
* Grigoriy Nesterov.

### <center>About the game
# <center>![](https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/bank_heist.gif)
* Bank heist is a game where the player controls a car. His task is to collect bags of money in the maze and move to the next level.
* The player has 4 lives at the beginning of the game. After the player picks up a bag, a police car appears in its place. The player needs to avoid the policemen, otherwise he will be caught and his life will be spent.
* The player can spawn dynamite to blow up the police officers. But the player needs to be careful because it is possible to blow himself up.
* The player's car also has fuel. If it runs out of fuel, it is the end of the game.
* The player can turn up, down, right, left, shoot and do nothing.
* The number of points is the amount of money the player has collected.

### <center>Preprocessing
# <center>![](https://raw.githubusercontent.com/abobafett-dev/RLIA-project-iu-2024/main/Report%20media/output.png)
#### <center> Left image - unprocessed image, right image - preprocessed matrix of states.
From environment, we get an observation of the current step (grayscale image of the game 210x160 pixels represented by numpy array with uint8 data type).
We cut this image to only playable field of the game, and then it convert to observation matrix of size 68x68 with encoding of each element.<br>
This greatly increased speed and performance of our model.

### <center>DDQN and model inside it
For this game we decided use Double Q-learning (DDQN) where we soft update our target network with by formula 
$\Theta_{\text{target}} \leftarrow \tau \cdot \Theta_{\text{local}} + (1 - \tau) \cdot \Theta_{\text{target}}$




##### <center>Video demonstration of the model
[![Youtube link](https://img.youtube.com/vi/apy8TOZutRQ/0.jpg)](https://www.youtube.com/watch?v=apy8TOZutRQ)
