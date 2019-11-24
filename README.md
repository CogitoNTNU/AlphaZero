# AlphaZero
AlphaZero is a computer program developed by artificial intelligence research company DeepMind. The program is generalized to work on all two-player complete information games such as tic tac toe, four in a row, chess and go. It is only given the rules of the game, and learns to master the game solely by playing against itself.

## About us
We are a project group within the student organization Cogito at NTNU (Norwegian University of Science and Technology). The group of eight has worked on this through the fall of 2019.

## Motivation
This project was created to achieve a greater understanding of the workings of one of the hottest reinforcement learning algorithms, as well as to have fun in the process. Optimization and introduction of parallel training was motivated by the desire of being able to train a network in reasonable time. This is espacially important for games with large action- and state spaces, as convergence will take a long time. The visualization was fueled by the quest to see how the network is thinking.

## Installation
Clone the repository
```
git clone https://github.com/CogitoNTNU/AlphaZero.git
```
Navigate into the project folder
```
cd AlphaZero
```
Install the dependencies
```
pip3 install -r requirements.txt
```
If everything went well, you can run the code
```
python3 Main.py
```
And you should then get a Tic Tac Toe game up 
<p align="center">
<img src="https://user-images.githubusercontent.com/45593399/68744963-ff811f00-05f5-11ea-8fd4-180ab7e3651f.png" width="200" height="225" />
</p>

## Visualisation
To visualise the tree-search, install graphviz from this external link: [Graphviz](https://graphviz.gitlab.io/download/)  
If you are using Windows, remember to add the graphviz/bin directory to PATH.  
You should now see something like this: 
<p align="center"><img src="https://tinyurl.com/yyk9vfpg" width="600" height="300" /></p>


## How to play
To play against AlphaZero write:
```
python3 play.py

```
### Tweak parameters
There are command line arguments, that simplify changing between games, number of searches and opponent. Currently supported games are Tic Tac Toe and Four In A Row. Default values are TicTacToe, 500 and 10_3_3.h5 respectively.
```
python3 play.py --game FourInARow --numSearch 1000
```

### Speeding up training
Even though AlphaZero is an effective algorithm and has achieved impressive feats, it is infamous for its computation costs required for training. Speeding up training is essential for several reasons. First of all, faster training allows us to tune hyperparameters considerably faster; we calculated that it went from taking weeks to only taking a few days. Secondly, AlphaZero are now able to complete training in a couple of days instead of weeks.<br><br>
There were two concepts that significantly sped up the training; parallelization and batching. We also used caching, which caused a minor speed improvement. 
We had eight processes that generated games in parallel. Each of which played 400 games simultaneously, bathing up the Resnet predictions. Thus a total of 4000 games were played in parallel each epoch. All of this resulted in a speedup of about 16 times.

### Benchmark generations
To play a version of AlphaZero against all others run
```
python3 play.py --game FourInARow --numSearch 1000
```

## Results
Having played only 3000 games of Tic Tac Toe againts itself, AZ was able to master the game. By only jusing raw predictrions from the resnet, it plays perfectly. 
After 100 000 self-played games on Four In A Row, the network is able to play at a decent level with 500 searches per move.
__more results coming soon__


## Add your own game
Create a folder for your game and implement the functions in Config (Contains information about board dimensions, and can convert between game actions and action numbers) and Gamelogic (Contains board and functions like execute a move, undo, reset, get legal moves, and see if someone has won).

## Additions for future projects
* Pondering - Make the network think while it is the players turn.  
* Virtual loss - Spread out threads on different lines while multithreading, to avoid duplicate work.  
* Chess type clock - Add ability to play with a timer so that the AI can think as long as the human, as to simulate a real match.  
* Transfer learning - Use a pre-trained resnet and only train the first and last few layers. This shuold speed up convergence significantly. The weights can for instance be taken from Leela Chess Zero.
* Match temperature - A temperature argument as to make the network not behave deterministicly when playing against a human (can be extended easily by the same approach used in training).

## Acknowledgments
* The [AlphaZero-](https://deepmind.com/documents/260/alphazero_preprint.pdf "AlphaZero paper by D. Silver et al.") and [AphaGo Zero paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf "AlphaGo Zero paper by D. Silver et al.") are essential to read to achieve a thorough understanding of the algorithm. 
* For a brief walkthrough of the algorithm and a more "hands on approach", I recommend reading through [this article](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191 "Lessons From Implementing AlphaZero") on Medium about an implementation of the AlpaZero algorithm.
* To get a gentle introduction to the algorithm, [this video](https://www.youtube.com/watch?v=2ciR6rA85tg "AlphaZero: DeepMind's New Chess AI ") by Two Minute Papers might be a nice place to start.
* David Silver also [explains AlphaZero](https://www.youtube.com/watch?v=Wujy7OzvdJk=0s "Deepmind AlphaZero - Mastering Games Without Human Knowledge") himself.


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
