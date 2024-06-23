import DominoGame
import numpy as np
from random import Random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def conv_hand_for_agent(board, hand):
    if(board.history != []):
        possiblePlays = DominoGame.possPlays2Flipped(hand, board)
    else:
        possiblePlays = DominoGame.allPlays(hand)
    converted_hand = np.full((1,42), 0)[0]
    for i in range(len(possiblePlays)):
        converted_hand[i*3] = possiblePlays[i][0][0]+1
        converted_hand[i*3 + 1] = possiblePlays[i][0][1]+1
        if(possiblePlays[i][1] == "L"):
            converted_hand[i*3+2] = 0
        if(possiblePlays[i][1] == "R"):
            converted_hand[i*3+2] = 1
    converted_hand = converted_hand.astype("int8")
    return converted_hand

#right now just converts the 2 edge dominos
def conv_board_for_agent(board):
    if(board.history == []):
        return [0, 0], [0, 0] #if empty board
    l_dom = [board.d1[0]+1, board.d1[1]+1]
    r_dom = [board.d2[0]+1, board.d2[1]+1] 
    return l_dom, r_dom

from gymnasium.wrappers import FlattenObservation # type: ignore
class DominoEnv(gym.Env):
    def __init__(self):
        super(DominoEnv, self).__init__()
        self.random = Random()

        l_r_side = 2
#         self.action_space = spaces.MultiDiscrete([n_actions, l_r_side])

        #agent chooses domino to play and it's side
        self.action_space = spaces.MultiDiscrete([7,7, l_r_side])

        #for actionspace possplays
        #self.action_space = spaces.Discrete(14)
        
        #FOR NOW JUST SHOW EDGE DOMINOS!!! (easy temp solution)
        
        self.observation_space = spaces.MultiDiscrete(
            [8,8,8,8,# the board l/r dominos - extra state for empty board
             8,8,2,8,8,2,8,8,2,8,8,2,8,8,2,8,8,2,8,8,2,
             8,8,2,8,8,2,8,8,2,8,8,2,8,8,2,8,8,2,8,8,2])#the possible plays - extra state for empty possplays
        
        #self.observation_space = spaces.Dict({"l_dom":l_dom, "r_dom":r_dom, "p_hand":player_hand})
        
#         # The observation will be the game the agent can see
#         # this can be described both by Discrete and Box space
#         #domino, player that played it, turn played and also hand of player
#         #28 possible dominos(49 possible permutations)(use binary to encode flip),
#         #14 possible played locations/turns, 2 players, 
#         #28 possible dominos, 7 in hand
#         player_hand = spaces.Box(low=1, high=28, shape=(1, 7), dtype=np.int32)
#         #board = spaces
#         self.observation_space = spaces.Tuple((spaces.Discrete(28), spaces.Discrete(14), spaces.Discrete(7)))

#         def get_info(self):
#             return {"history": self.history, "player": self.player, "winner": self.winner}

    def conv_agent_action_to_dom(self, hand, action):
        #convert action to corresponding domino
        # #CONVERSION FOR POSSPLAYS AS ACTIONSPACE
        # if(self.board.history != []):
        #     possiblePlays = DominoGame.possPlays2Flipped(hand, self.board)
        # else:
        #     possiblePlays = DominoGame.allPlays(hand)
        # if(action < len(possiblePlays)):
        #     return possiblePlays[action]
        # else:
        #     return "invalid"

        domino = (action[0],action[1])
        #check domino in hand
        if(not (DominoGame.flipIfNeeded(domino) in hand)):
            return "invalid"
        side_num = action[2]
        if(side_num == 0):
            side = "L"
        else:
            side = "R"
        return (domino, side)
    
    #returns mask for the action space based on possible plays
    def valid_action_mask(self):
        if(self.board.history != []):
            possiblePlays = DominoGame.possPlays2Flipped(self.hand1, self.board)
        else:
            possiblePlays = DominoGame.allPlays(self.hand1)
        
        # #MASK FOR POSSPLAYS AS ACTIONSPACE
        # masked_array = np.zeros((1,14))[0]
        # masked_array[:len(possiblePlays)] = 1

        #for when agent returned a domino+side action
        masked_array = np.zeros((1,16))[0]
        for i in possiblePlays:
            l_side = i[0][0]
            r_side = i[0][1]
            if(i[1] == "L"):
                side = 0
            else:
                side = 1
            masked_array[l_side] = 1
            masked_array[r_side+7] = 1
            masked_array[side+14] = 1
        return masked_array

    #if the round is over, but not the game (no winner yet)
    def next_round(self):
        self.board = DominoGame.Board(0, 0, [])
        self.round_done = False
        self.invalid_num = 0

        self.player_turn = self.random.randrange(2)+1
        self.p1_blocked = False
        self.p2_blocked = False

        shuffled = self.random.sample(DominoGame.domSet, len(DominoGame.domSet))
        self.hand1 = shuffled[:self.hand_size]
        self.hand2 = shuffled[self.hand_size:self.hand_size*2]
        
    def reset(self, seed=None, options=None):
        self.random.seed()
        #super.reset(seed=seed, options=options)
        self.prev_reward = 0
        self.round_done = False
        self.done = False
        self.invalid_num = 0
        
        self.hand_size = 7
        self.target_score = 10
        self.board = DominoGame.Board(0, 0, [])
        self.game_score = [0,0]
        self.history = []
        self.player = 1
        self.player_turn = self.random.randrange(2)+1
        self.played = False

        self.p1_blocked = False
        self.p2_blocked = False

        shuffled = self.random.sample(DominoGame.domSet, len(DominoGame.domSet))
        self.hand1 = shuffled[:self.hand_size]
        self.hand2 = shuffled[self.hand_size:self.hand_size*2]

        info = {}
        #if agent isnt first to play, then make other player do its play, so that reward isnt messed up (trust me)
        if(self.player_turn != self.player):
            self.step(self.action_space.sample())

        agent_hand = conv_hand_for_agent(self.board, self.hand1)
        agent_board_l, agent_board_r = conv_board_for_agent(self.board)
        
        observation = np.concatenate((agent_board_l, agent_board_r, agent_hand))
        #print(observation)
        #observation = np.array(observation)

        return observation, {}

    def step(self, action):
        score = 0
        reward = 0
        #if either player blocked turn becomes next players'
        if ((self.player_turn == 1) & self.p1_blocked):
            reward -= 1
            self.player_turn = 2
        if ((self.player_turn == 2) & self.p2_blocked):
            reward += 1
            self.player_turn = 1

        #get chosen move by player (domino, end) and play it
        if (self.player_turn == self.player):
            self.played = True
            move = self.conv_agent_action_to_dom(self.hand1, action)
            #tries doing a move for a domino that is not in hand
            if(move == "invalid"):
                self.invalid_num += 1
                reward -= 2
                ##print("invalid move!!!")
            else:
                if(self.board.history != []):
                    possiblePlays = DominoGame.possPlays2Flipped(self.hand1, self.board)
                else:
                    possiblePlays = DominoGame.allPlays(self.hand1)
                #check if chosen move is valid and play if it is (dom in hand and in possiblePlays)
                if((DominoGame.domInHand(DominoGame.flipIfNeeded(move[0]), self.hand1)) and (move in possiblePlays)):
                    DominoGame.playDom(self.player_turn, move[0], self.board, move[1])
                    self.hand1.remove(DominoGame.flipIfNeeded(move[0]))
                    score = DominoGame.scoreBoard(self.board, (bool(self.hand1)))
                    if((self.game_score[0] + score) <= self.target_score):
                        self.game_score[0] += score
                        reward += score
                        self.prev_reward = reward
        
                    self.player_turn = (self.player_turn % 2) + 1
                    self.p2_blocked = DominoGame.blocked(self.hand2, self.board)
                    if (DominoGame.roundDone(self.game_score, self.target_score, self.p1_blocked, self.p2_blocked)):
                        self.round_done = True
                        if(DominoGame.gameDone(self.game_score, self.target_score)):
                            self.done = True
                    
                    #after p1 valid move, p2 does move if not blocked and game/round not finished
                    if(not self.p2_blocked and not self.round_done):
                        self.board, self.hand2, score = DominoGame.makeMove(DominoGame.randomPlayer, self.player_turn, self.hand2, self.board, self.game_score)
                        if((self.game_score[1] + score) <= self.target_score):
                            self.game_score[1] += score
                        self.player_turn = (self.player_turn % 2) + 1
                #rl agent tries doing a move that cannot be done with that domino
                else:
                    self.invalid_num += 1
                    reward -= 1
        #basically now only called at start of game if turn isnt agents'
        else: 
            self.board, self.hand2, score = DominoGame.makeMove(DominoGame.randomPlayer, self.player_turn, self.hand2, self.board, self.game_score)
            if((self.game_score[1] + score) <= self.target_score):
                self.game_score[1] += score
            self.player_turn = (self.player_turn % 2) + 1

        self.p1_blocked = DominoGame.blocked(self.hand1, self.board)
        self.p2_blocked = DominoGame.blocked(self.hand2, self.board)

        score_difference = self.game_score[0] - self.game_score[1]
        reward += score_difference

        if (DominoGame.roundDone(self.game_score, self.target_score, self.p1_blocked, self.p2_blocked)):
            self.round_done = True
            if(DominoGame.gameDone(self.game_score, self.target_score)):
                self.done = True
        
        if(self.done):
            if(self.game_score[0] < self.game_score[1]):
                reward -= 30
            else:
                reward += 30
        
        #too many invalid moves will exit game
        if(self.invalid_num > 16):
            # print("hand: ", self.hand1)
            # print("action: ", action)
            # print("actionMask: ", self.valid_action_mask())
            # if(self.board.history != []):
            #     possiblePlays = DominoGame.possPlays2Flipped(self.hand1, self.board)
            # else:
            #     possiblePlays = DominoGame.allPlays(self.hand1)
            # print("possPlays: ", possiblePlays)
            # print(self.board.history)
            # print(self.conv_agent_action_to_dom(self.hand1, action))
            self.done = True

        #if round is over but not game, reset board and hands for next round
        if(self.round_done & (not self.done)):
            self.next_round()
        
        #if p1 is blocked and the game is not done, call step and it is p2 turn, this is so p1 doesnt needlessly generate
        #actions that it cannot do while p2 makes its move(s)
        if ((self.player_turn == 1) and self.p1_blocked and (not self.round_done) and (not self.p2_blocked)):
            reward -= 1
            self.player_turn = 2
            observation, reward, self.done, truncated, info = self.step(action)
        
        if((not self.played) and self.done):
            self.reset()

        info = {"score":self.game_score, "history":self.board.history}
        truncated = False
        reward

        agent_hand = conv_hand_for_agent(self.board, self.hand1)
        agent_board_l, agent_board_r = conv_board_for_agent(self.board)
        
        observation = np.concatenate((agent_board_l, agent_board_r, agent_hand))

        return (observation, reward, self.done, truncated, info)
