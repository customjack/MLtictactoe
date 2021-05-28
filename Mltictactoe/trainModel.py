# -*- coding: utf-8 -*-
"""
Final Project:
    NN ML tic-tac-toe
    Training the NN


@author: Jack Carlton
"""
import numpy as np
import random
from tictactoe import TicTacToe
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from MLbot import determineMove
from MLbot import getAllPossibleMoves


EPOCHS = 16
BATCH_SIZE = 64
BAR = '-----------------------------'
N_ROWS, N_COLS = 3,3
N_GAMES = 20000

def getRandomData(N_games=20000,n_rows=3,n_cols=3):
    totalMoves = getAllPossibleMoves(n_rows,n_cols)
    gameBoards = []
    gameBoardResults = []
    results = []
    for i in range(N_games):
        game = TicTacToe(n_rows, n_cols)
        possibleMoves = totalMoves.copy()
        playerToMove = -1
        while not game.gameOver:
            selectedMove = random.choice(possibleMoves)
            possibleMoves.remove(selectedMove)
            game.playTurn(playerToMove, selectedMove[0], selectedMove[1])
            playerToMove*=-1
        for j in game.history:
            gameBoards.append(j.flatten())
            gameBoardResults.append(game.winner)
        results.append(game.winner) #Just to see how the games go randomly
        if i % 1000 == 0:
            print("Generating Games Data... " + str(i) +"/"+str(N_games) + " (" + str(round(i*100/float(N_games),3)) + "%)" )
    print("Generating Games Data... " + str(N_games) +"/"+str(N_games) + " (100%) [COMPLETE]" )
    return np.array(gameBoards),np.array(gameBoardResults),np.array(results)


def createModel(n_rows=3,n_cols=3):
    model = Sequential()
    model.add(Dense(n_rows*n_cols*8, activation='relu', input_shape=(n_rows*n_cols, )))
    model.add(Dense(n_rows*n_cols*4, activation='relu'))
    model.add(Dense(n_rows*n_cols*2, activation='relu'))
    model.add(Dense(n_rows*n_cols*1, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def trainModel(model, train_in, train_out,test_in,test_out):
    history = model.fit (train_in, train_out, epochs=EPOCHS, batch_size=BATCH_SIZE)
    scores = model.evaluate (test_in, test_out)
    print ()
    print('Test score:', scores[0])
    print ('%s: %.2f \n' % (model.metrics_names [1], scores [1]))
    return history,scores
    
def getTrainAndTestdata(gameBoards,winners,trainPercent=0.8):
    N = len(gameBoards)
    s = round(N*trainPercent)
    train_in = gameBoards[:s]
    train_out= tf.keras.utils.to_categorical(winners[:s],num_classes=3)
    test_in  = gameBoards[s:]
    test_out = tf.keras.utils.to_categorical(winners[s:],num_classes=3)
    return train_in,train_out,test_in,test_out

def plotHistoriesData(historyData):
    '''
    Plots the loss and accuracy functions from the history data
    '''
    plt.figure()
    if (isinstance(historyData,list)):
        histories = historyData
    else:
        histories = [historyData]
    #plot Loss
    plt.subplot(2, 1, 1)
    for history in histories:
        plt.plot(history.history['loss'], color = 'blue')
    plt.title('Cross Entropy Loss')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.grid()
         
    # plot accuracy
    plt.subplot(2, 1, 2)
    for history in histories:
        plt.plot(history.history['accuracy'],color = 'blue')
    plt.title('Classification Accuracy')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('loss-tic-tac-toe.png')

def testTrainedBot(model,N_games=20000,n_rows=3,n_cols=3,botPlayer=1):
    totalMoves = getAllPossibleMoves(n_rows,n_cols)
    results = []
    for i in range(N_games):
        game = TicTacToe(n_rows, n_cols)
        possibleMoves = totalMoves.copy()
        playerToMove = -1
        while not game.gameOver: #Random Move
            if playerToMove != botPlayer:
                selectedMove = random.choice(possibleMoves)
                possibleMoves.remove(selectedMove)
                game.playTurn(playerToMove, selectedMove[0], selectedMove[1])
            else:
                selectedMove = determineMove(model,botPlayer,possibleMoves,game)
                possibleMoves.remove(selectedMove)
                game.playTurn(playerToMove, selectedMove[0], selectedMove[1])
            playerToMove*=-1
        results.append(game.winner) #Just to see how the games go randomly
        if i % 1000 == 0:
            print("Testing Trained Bot... " + str(i) +"/"+str(N_games) + " (" + str(round(i*100/float(N_games),3)) + "%)" )
    print("Testing Trained Bot... " + str(N_games) +"/"+str(N_games) + " (100%) [COMPLETE]" )
    return np.array(results)

#%% Actual Script
gameBoards, winners, results = getRandomData(N_games=N_GAMES,n_rows=N_ROWS,n_cols=N_COLS)
train_in,train_out,test_in,test_out = getTrainAndTestdata(gameBoards, winners)
O_win_percentage = (results == 1).sum()/float(len(results)) 
X_win_percentage = (results == -1).sum()/float(len(results)) 
Draw_percentage  = (results == 0).sum()/float(len(results))
print()
print("Random Bot (X) vs. Random Bot (O)")
print(BAR)
print("O wins: " + str(round(O_win_percentage*100,3)) + '%')
print("X wins: " + str(round(X_win_percentage*100,3)) + '%')
print("Draw  : " + str(round(Draw_percentage*100,3)) + '%')

model = createModel(n_rows=N_ROWS,n_cols=N_COLS)
history, scores = trainModel(model,train_in,train_out,test_in,test_out)
plotHistoriesData(history)
model.save('tictactoe_'+str(N_ROWS)+'x'+str(N_COLS)+'.model')

results = testTrainedBot(model,n_rows=N_ROWS,n_cols=N_COLS)
O_win_percentage = (results == 1).sum()/float(len(results)) 
X_win_percentage = (results == -1).sum()/float(len(results)) 
Draw_percentage  = (results == 0).sum()/float(len(results))
print()
print("Random Bot (X) vs. ML Bot (O)")
print(BAR)
print("O wins: " + str(round(O_win_percentage*100,3)) + '%')
print("X wins: " + str(round(X_win_percentage*100,3)) + '%')
print("Draw  : " + str(round(Draw_percentage*100,3)) + '%')

results = testTrainedBot(model,botPlayer=-1,n_rows=N_ROWS,n_cols=N_COLS)
O_win_percentage = (results == 1).sum()/float(len(results)) 
X_win_percentage = (results == -1).sum()/float(len(results)) 
Draw_percentage  = (results == 0).sum()/float(len(results))
print()
print("ML Bot (X) vs. Random Bot (O)")
print(BAR)
print("O wins: " + str(round(O_win_percentage*100,3)) + '%')
print("X wins: " + str(round(X_win_percentage*100,3)) + '%')
print("Draw  : " + str(round(Draw_percentage*100,3)) + '%')