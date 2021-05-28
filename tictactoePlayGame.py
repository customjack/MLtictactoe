# -*- coding: utf-8 -*-
"""
Final Project:
    NN ML tic-tac-toe
    The "Play" Part of the code

@author: Jack Carlton
"""

from tictactoe import TicTacToe
from MLbot import determineMove
from MLbot import getAllPossibleMoves
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense

BOT_MODEL_NAME = 'tictactoe_3x3.model'


def playGame(model,n_rows=3,n_cols=3,botPlayer=1):
    totalMoves = getAllPossibleMoves(n_rows,n_cols)
    game = TicTacToe(n_rows, n_cols)
    game.drawBoard()
    possibleMoves = totalMoves.copy()
    playerToMove = -1
    while not game.gameOver: #Random Move
        if playerToMove != botPlayer:
            selectedMove = getMoveFromHuman(possibleMoves, n_rows, n_cols)
            possibleMoves.remove(selectedMove)
            game.playTurn(playerToMove, selectedMove[0], selectedMove[1])
        else:
            selectedMove = determineMove(model,botPlayer,possibleMoves,game)
            possibleMoves.remove(selectedMove)
            game.playTurn(playerToMove, selectedMove[0], selectedMove[1])
        playerToMove*=-1
        game.drawBoard()
    if (game.winner != 0):
        winnerSymbol = game.getSymbol(game.winner)
        print(winnerSymbol + " won!")
    else:
        print("Draw!")

def getMoveFromHuman(possibleMoves,n_rows,n_cols):
    move = []
    while move not in possibleMoves:
        row = -1
        while row < 0 or row >= n_rows:
            try:
                row=int(input("What row do you want:"))
            except ValueError:
                print("Pick an integer between 0 and " + str(n_rows-1))
            if (row < 0 or row >= n_rows):
                print("Pick an integer between 0 and " + str(n_rows-1))
        col = -1
        while col < 0 or col >= n_cols:
            try:
                col=int(input("What col do you want:"))
            except ValueError:
                print("Pick an integer between 0 and " + str(n_cols-1))
            if (col < 0 or col >= n_cols):
                print("Pick an integer between 0 and " + str(n_rows-1))
        move = [row,col]
        if move not in possibleMoves:
            print("The space (" + str(row) + ',' + str(col) + ") is already occupied")
    return move

def split(word):
    return [char for char in word]

#%% Actual Script
model = keras.models.load_model(BOT_MODEL_NAME)
dimensions = []
for word in split(BOT_MODEL_NAME):
   if word.isdigit():
      dimensions.append(int(word))
n_rows = dimensions[0]
n_cols = dimensions[1]
playerChoice = ''
while (playerChoice != 'X' and playerChoice != 'x' and playerChoice!='O' and playerChoice!='o'):
    playerChoice = input("Do you want to be X's (first) or O's (second)?: ")
botPlayer = -1
if playerChoice == 'X' or playerChoice == 'x':
    botPlayer = 1
playGame(model,n_rows,n_cols,botPlayer=botPlayer)
