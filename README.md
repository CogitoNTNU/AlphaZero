# AlphaZero
AlphaZero is a computer program developed by artificial intelligence research company DeepMind. The program is generalized to work on all two-player complete information games such as tic tac toe, four in a row, chess and go. It only requires the rules of the game, and will with training try to learn which moves are the best for a given board state through

## Who we are
We are a project group within the student organization Cogito at NTNU (Norwegian University of Science and Technology). The group of seven has worked on this through the fall semester 2019

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

<img src="https://user-images.githubusercontent.com/45593399/68744963-ff811f00-05f5-11ea-8fd4-180ab7e3651f.png" width="200" height="225" />


## Visualisation
If you wish to visualise the tree-search, you must install graphviz from this external link: [Graphviz](https://graphviz.gitlab.io/download/)  
If you are using Windows, you must also add the graphviz/bin directory to PATH.  
If done correctly, you should see something like this: <img src="https://tinyurl.com/yyk9vfpg" width="600" height="300" />

## Tweak playing parameters
There are command line arguments so that you can play one of the currently supported games, Tic Tac Toe or Four In A Row. And you can choose how many nodes you want the machine to search. (default set to TicTacToe and 500 searches)
```
python3 play.py --game FourInARow --numSearch 1000
```

## Results
After 3000 self-played games on Tic Tac Toe, the network was able to rarely loose, even on 10 search per move in match conditions against a human. With 500 searches per move  
After 200 000 self-played games on Four In A Row, the network is able to play at an adequate level with 1000 searches per move. 

## How to train
_Coming soon_

## Add your own game
_Coming soon_

## Additions to future projects
Pondering, Virtual loss, Chess type clock, Pretrained net, command line to play against different versions, a little temperature in match conditions to get variation

## Other resources
* The [AlphaZero-](https://deepmind.com/documents/260/alphazero_preprint.pdf "AlphaZero paper by D. Silver et al.") and [AphaGo Zero paper](https://deepmind.com/documents/119/agz_unformatted_nature.pdf "AlphaGo Zero paper by D. Silver et al.") are essential to read to achieve a thorough understanding of the algorithm. 
* For a brief walkthrough of the algorithm and a more "hands on approach", I recommend reading through [this article](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191 "Lessons From Implementing AlphaZero") on Medium about an implementation of the AlpaZero algorithm.
* To get a gentle introduction to the algorithm, [this video](https://www.youtube.com/watch?v=2ciR6rA85tg "AlphaZero: DeepMind's New Chess AI ") by Two Minute Papers might be a nice place to start.
* David Silver also [explains AlphaZero](https://www.youtube.com/watch?v=Wujy7OzvdJk=0s "Deepmind AlphaZero - Mastering Games Without Human Knowledge") himself.


## Motivation
_Coming soon_

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

