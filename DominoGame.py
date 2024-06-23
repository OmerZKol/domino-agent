import numpy as np
import json
from utils import utils
from random import Random

TOTAL_HAND_SIZE = 14
#generate domSet
domSet = []
for i in range(0,7):
    for j in range(0,i+1):
        domSet.append((i,j))

#select random domino
SEED = 1
random = Random()
random.seed(SEED)
randomSample = random.sample(domSet,5)
shuffled = random.sample(domSet, len(domSet))

#class to represent the board, d1 is leftmost domino, d2 is rightmost domino and history is the current games' play history
#and contains the domino played, player that played it and the turn it was played on for each play done this game
class Board():
    def __init__(self, d1 , d2 , history):
        self.d1 = d1
        self.d2 = d2
        self.history = history
    
    def update(self, d1, d2 , history):
        self.d1 = d1
        self.d2 = d2
        self.history = history

def flip(domino):
    return (domino[1], domino[0])

def flipIfNeeded(domino):
    flipped = domino
    if (domino[1] > domino[0]):
        flipped = flip(domino)
    return flipped

#tests if a domino is in hand
def domInHand(domino, hand):
    flipped = flipIfNeeded(domino)
    return flipped in hand

def symmetricalDom(domino):
    return domino[0] == domino[1]

def bothDomSymmetrical(d1, d2):
    return symmetricalDom(d1) & symmetricalDom(d2)

#get turn number of that last(previous) turn
def lastTurnNum(history):
    if(history[0][2] > history[len(history)-1][2]):
        return history[0][2]
    return history[len(history)-1][2]

def calcPipScore(d1, d2, history):
    if(lastTurnNum(history) == 1):
        return d1[0]+d1[1]
    if(bothDomSymmetrical(d1, d2)):
        return d1[0]+d1[1]+d2[0]+d2[1]
    if(symmetricalDom(d1)):
        return d1[0]+d1[1]+d2[1]
    if(symmetricalDom(d2)):
        return d1[0]+d2[0]+d2[1]
    return d1[0]+d2[1]

def calcScore(board):
    score = calcPipScore(board.d1, board.d2, board.history)
    if(score%15 == 0):
        return int((score/3) + (score/5))
    if(score%5 == 0):
        return int((score/5))
    if(score%3 == 0):
        return int((score/3))
    return 0

def scoreBoard(board, lastDom):
    if(len(board.history) == 0):
        return 0
    score = calcScore(board)
    if(lastDom):
        return score+1
    return score

def canPlay(dom, end, board):
    #if end is left
    if(end == "L"):
        return ((dom[0] == board.d1[0]) | (dom[1] == board.d1[0]))
    #otherwise is right
    return ((dom[0] == board.d2[1]) | (dom[1] == board.d2[1]))

#get all possible plays for current hand (used for empty board state ie first move)
def allPlays(hand):
    possiblePlays = []
    for i in hand:
        possiblePlays.append((i, "L"))
    for i in hand:
        possiblePlays.append((i, "R"))
    return possiblePlays

def possPlays(hand, board):
    possiblePlays = ([],[])
    for i in hand:
        if(canPlay(i, "L", board)):
            possiblePlays[0].append(i)
    for i in hand:
        if(canPlay(i, "R", board)):
            possiblePlays[1].append(i)
    return possiblePlays

def possPlays2(hand, board):
    possiblePlays = []
    for i in hand:
        if(canPlay(i, "L", board)):
            possiblePlays.append((i, "L"))
    for i in hand:
        if(canPlay(i, "R", board)):
            possiblePlays.append((i, "R"))
    return possiblePlays

#possplays2, but the dominos are flipped if necessary for that play
def possPlays2Flipped(hand, board):
    possiblePlays = []
    for i in hand:
        if(canPlay(i, "L", board)):
            if(board.d1[0] != i[1]):
                possiblePlays.append((flip(i), "L"))
            else:
                possiblePlays.append((i, "L"))
    for i in hand:
        if(canPlay(i, "R", board)):
            if(board.d2[1] != i[0]):
                possiblePlays.append((flip(i), "R"))
            else:
                possiblePlays.append((i, "R"))
    return possiblePlays

def blocked(hand, board):
    if(board.d1 == 0):
        return False
    possiblePlays = possPlays(hand, board)
    #true if no possible plays, false if there are possible plays
    return ((len(possiblePlays[0]) == 0) & (len(possiblePlays[1]) == 0))

def flipIfNeededBoard(domino, end, board):
    if(((end == "L") & (domino[0] == board.d1[0])) | ((end == "R") & (domino[1] == board.d2[1]))):
        return flip(domino)
    return domino

#returns true if round over
def roundDone(game_score, target_score, p1_blocked, p2_blocked):
    return not ((game_score[0] != target_score) & (game_score[1] != target_score)) & (not(p1_blocked & p2_blocked))

#returns true if a player has won the game
def gameDone(game_score, target_score):
    return not ((game_score[0] != target_score) & (game_score[1] != target_score))

#returns are needed to break out of function
def playDom(player, domino, board, end):
    if(board.d1 == 0):
        return board.update(domino, domino, [(domino, player, 1)])
    playedDom = flipIfNeededBoard(domino, end, board)
    thisTurnNum = lastTurnNum(board.history)+1
    if(end == "L"):
        board.d1 = playedDom
        return board.history.insert(0, (playedDom, player, thisTurnNum))
    else:
        board.d2 = playedDom
        return board.history.append((playedDom, player, thisTurnNum))

def makeMove(player, player_turn, hand, board, game_score):
    move = player(hand, board, player_turn, game_score)
    #check if chosen move is valid and play if it is
    if(domInHand(move[0], hand)):
        playDom(player_turn, move[0], board, move[1])
        hand.remove(flipIfNeeded(move[0]))
        score = scoreBoard(board, (bool(hand)))
    else: 
        print("invalid move!!!")
    return board, hand, score

def playDomsRound(hand_size, target_score, p1, p2, player_turn, game_score, seed):
    board = Board(0, 0, [])
    p1_blocked = False
    p2_blocked = False
    
    shuffled = random.sample(domSet, len(domSet))
    hand1 = shuffled[:hand_size]
    hand2 = shuffled[hand_size:hand_size*2]
    
    hands = []
    ###p_hands_dict = {1:hand1.copy(), 2:hand2.copy()}
    #ok so board starts from first play, which means hands cannot record hand before the play as there is no 
    #0, 0 initial boardstate recorded/saved
    p_hands_dict = {1:[], 2:[]}
    hands.append(p_hands_dict)
    
    ##print(game_score)
    #continues game until player reaches score or both players blocked
    while(not roundDone(game_score, target_score, p1_blocked, p2_blocked)):
        hands.append({1:hand1.copy(), 2:hand2.copy()})
        p1_blocked = blocked(hand1, board)
        p2_blocked = blocked(hand2, board)
        
        #if either player blocked turn becomes next players
        if ((player_turn == 1) & p1_blocked):
            player_turn = 2
            continue
        if ((player_turn == 2) & p2_blocked):
            player_turn = 1
            continue
            
        #get chosen move by player (domino, end) and play it
        if (player_turn == 1):
            board, hand1, score = makeMove(p1, player_turn, hand1, board, game_score)
            if((game_score[0] + score) <= target_score):
                game_score[0] += score
        else: 
            board, hand2, score = makeMove(p2, player_turn, hand2, board, game_score)
            if((game_score[1] + score) <= target_score):
                game_score[1] += score
        player_turn = (player_turn % 2) + 1
        #p_hands_dict = {1:hand1, 2:hand2}
        ###hands.append({1:hand1.copy(), 2:hand2.copy()})
        ##print(game_score)
        #check if chosen domino in hand
        
    #print(board.history, "\n")
    #print(hands, "\n")
    return game_score, board, hands

#1 means p1 won 2 means p2 won
def playGame(hand_size, target_score, p1, p2, first_player, seed):
    board_hand_dict = {"board":[], "hands":[]}
    score = [0,0]
    while((score[0] != target_score) & (score[1] != target_score)):
        random.seed(seed)
        seed = random.randint(0,100000)
        score, board, hands = playDomsRound(hand_size, target_score, p1, p2, first_player, score, seed)
        
        board_hand_dict["board"].append(board)
        board_hand_dict["hands"].append(hands)
        
        first_player = (first_player % 2) + 1
        
    board_hand_dict["score"] = score
    if(score[0] == target_score):
        return 1, board_hand_dict
    return 2, board_hand_dict

#number of games, number of dominos, target score, func to determine player moves, seed
def domsMatch(number_games, hand_size, target_score, p1, p2, seed):
    all_b_h_dict = {1:[], 2:[]}
    final_score = [0,0]
    #first_player = 1
    while(number_games > 0):
        random.seed(seed)
        seed = random.randint(0,1000000)
        if((number_games%2) == 1):
            first_player = 1
        else:
            first_player = 2
            
        winner, board_hand_d = playGame(hand_size, target_score, p1, p2, first_player, seed)
        all_b_h_dict[winner].append(board_hand_d)
        if (winner == 1):
            final_score[0] = final_score[0] + 1
        else:
            final_score[1] = final_score[1] + 1
        number_games -= 1
    return final_score, all_b_h_dict

def simplePlayer(hand, board, player, scores):
    if(len(board.history) == 0):
        return (hand[0], "L")
    possiblePlays = possPlays(hand, board)
    if(len(possiblePlays[0]) == 0):
        return (possiblePlays[1][0], "R")
    return (possiblePlays[0][0], "L")

random_p = Random()
random_p.seed()

def randomMove(possiblePlays, random_obj):
    random_move = random_obj.choice(possiblePlays)
    return random_move

def randomPlayer(hand, board, player, scores):
    if(len(board.history) == 0):
        move = randomMove(hand, random_p)
        return (move, "L")
    possiblePlays = possPlays2(hand, board)
    move = randomMove(possiblePlays, random_p)
    return move

def main():
    target_score = 15
    result, game_data = domsMatch(1000, 7, target_score, simplePlayer, simplePlayer, 4)
    print(result)

if __name__ == "__main__":
    main()