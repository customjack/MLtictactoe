# MLtictactoe
Machine Learning tic tac toe using linear neural network model.

REQUIRED PACKAGES:

- numpy

- tensorflow (can be downloaded with by downloading keras)

- keras

- matplotlib


You can train a model that can be used by a bot to choose moves with trainModel.py. 
The game can be an arbitrary number of rows and columns, configurable by changing N_ROWS and N_COLS.
N_GAMES is the number of games two bots will play (with each move decided randomly) to generate training data for the keras model.
After training, the model will automatically be saved as 'tictactoe_{N_ROWS}x{N_COLS}.model'

You can play a bot that uses the trained model to decide it's moves by running tictactoePlayGame.py.
You can change which bot you are playing by editting BOT_MODEL_NAME.

tictactoe.py contains a class that handles the tic tac toe game and stores game data.

MLbot.py contains the code the bot uses to decide it's moves.
