# -*- coding: utf-8 -*-
"""
Final Project:
    NN ML tic-tac-toe
    The ML bot's move selection algorithm

@author: Jack Carlton
"""
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense

def determineMove(model,botPlayer,possibleMoves,game):
    bestMoveIndex = 0
    bestMoveScore = 0
    possibleMoveBoards = []
    for i in range(len(possibleMoves)):
        possibleMoveBoard = createFutureGameBoard(game.board, possibleMoves[i], botPlayer)
        possibleMoveBoards.append(possibleMoveBoard)
    possibleMoveBoards = np.array(possibleMoveBoards)
    predictions = model.predict(possibleMoveBoards)
    scores = calcMoveScores(predictions,botPlayer)
    for i in range(len(scores)):
        score = scores[i]
        if (score - bestMoveScore >= 0.001):
            bestMoveScore = score
            bestMoveIndex = i
        elif (abs(score - bestMoveScore) < 0.001): #We throw a little randomness if move choices are very similiar expected value
            if (random.random() > 0.5):
                bestMoveScore = score
                bestMoveIndex = i
    return possibleMoves[bestMoveIndex]
        
        
def calcMoveScores(moveVectors,botPlayer):
    moveScores = []
    for moveVector in moveVectors:
        moveScores.append(botPlayer*(1*moveVector[1]+(-1)*moveVector[2]))
    return moveScores

def createFutureGameBoard(board,move,player):
    newboard = board.copy()
    newboard[move[0]][move[1]] = player
    return newboard.flatten()

def getAllPossibleMoves(n_rows,n_cols):
    possibleMoves = []
    for i in range(0,n_rows):
        for j in range(0,n_cols):
            possibleMoves.append([i,j])
    return possibleMoves