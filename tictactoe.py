# -*- coding: utf-8 -*-
"""
Final Project:
    NN ML tic-tac-toe
    Just the Tic tac toe code

@author: Jack Carlton
"""
import numpy as np
import random

class TicTacToe:
    
    def __init__(self,n_rows,n_cols):
        self.board = np.zeros((n_rows,n_cols))
        self.history = []
        self.rows = n_rows
        self.cols = n_cols
        self.gameOver = False
        self.winner = 0
        self.numMoves = 0
        
    def drawBoard(self):
        for i in range(len(self.board)):
            boardRow = self.board[i]
            if i != 0: #Draw the divider first
                divider = '-'
                for j in range(len(boardRow)-1):
                    divider += '+-'
                print(divider)
            rowString = '' #Now we print the row string
            for j in range(len(boardRow)): #construct the row string
                symbol = self.getSymbol(boardRow[j])
                if j != len(boardRow)-1:
                    rowString += symbol + '|'
                else:
                     rowString += symbol
            print(rowString)
        print()
        
        
    def getSymbol(self,boardInt):
        if boardInt == 0:
            return ' '
        elif boardInt == 1:
            return 'O'
        elif boardInt == -1:
            return 'X'
        
    def playTurn(self,player,row,column):
        if not self.gameOver:
            self.numMoves += 1
            self.board[row][column] = player
            self.history.append(self.board.copy())
            self.gameOver = self.checkForWin()
            if self.gameOver:
                self.winner = player
            elif self.numMoves == self.rows*self.cols: #Maximum moves reached
                self.gameOver = True #No winner added, game ended
        
        
    def checkForWin(self):
        #Check rows for line
        for i in self.board:
            if (all(elem == i[0] and elem != 0 for elem in i)):
                return True
                
        #Check columns for line
        board_T = np.transpose(self.board)
        for i in board_T:
            if (all(elem == i[0] and elem != 0 for elem in i)):
                return True
                
        
        #Check Diagonals for line
        if self.rows == self.cols: #No diagonals otherwise
            diagonal1 = self.board.diagonal()
            diagonal2 = np.fliplr(self.board).diagonal()
            if (all(elem == diagonal1[0] and elem != 0 for elem in diagonal1)):
                return True
            if (all(elem == diagonal2[0] and elem != 0 for elem in diagonal2)):
                return True
        
        return False #If all tests fail, return false