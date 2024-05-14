from random import randint, choice


class Player:
    def __init__(self, name="Player"):
        self.name = name

    def move(self, game, time_left):
        pass

    def get_name(self):
        return self.name


class RandomPlayer(Player):
    """Player that chooses a move randomly."""
    def __init__(self, name="RandomPlayer"):
        super().__init__(name)

    def move(self, game, time_left):
        if not game.get_player_moves(self):
            return None
        else:
            return choice(game.get_player_moves(self))

    def get_name(self):
        return self.name


class HumanPlayer(Player):
    """
    Player that chooses a move according to user's input. 
    (Useful if you play in the terminal)
    """
    def __init__(self, name="HumanPlayer"):
        super().__init__(name)

    def move(self, game, time_left):
        legal_moves = game.get_player_moves(self)
        my_choice = {}

        if not len(legal_moves):
            print("No more moves left.")
            return None, None

        counter = 1
        for move in legal_moves:
            my_choice.update({counter: move})
            print('\t'.join(['[%d] (%d,%d)' % (counter, move[0], move[1])]))
            counter += 1

        print("-------------------------")
        print(game.print_board(legal_moves))
        print("-------------------------")
        print(">< - impossible, o - valid move")
        print("-------------------------")

        valid_choice = False
        index = 0

        while not valid_choice:
            try:
                index = int(input('Select move index [1-' + str(len(legal_moves)) + ']:'))
                valid_choice = 1 <= index <= len(legal_moves)

                if not valid_choice:
                    print('Illegal move of queen! Try again.')
            except Exception:
                print('Invalid entry! Try again.')

        return my_choice[index]

    def get_name(self):
        return self.name


def get_trap_moves():
    # Where to go (especially when skidding) to set a trap
    trap_moves = (
        ((1, 0), (1, 2), (1, 4), (1, 6)),  # 0
        ((5, 0), (5, 2), (5, 4), (5, 6)),  # 1
        ((0, 1), (2, 1), (4, 1), (6, 1)),  # 2
        ((0, 5), (2, 5), (4, 5), (6, 5)),  # 3
        ((2, 0), (2, 2), (2, 4), (2, 6)),  # 4
        ((4, 0), (4, 2), (4, 4), (4, 6)),  # 5
        ((0, 2), (2, 2), (4, 2), (6, 2)),  # 6
        ((0, 4), (2, 4), (4, 4), (6, 4))  # 7
    )
    return trap_moves


def get_trappable_positions():
    # Wherein the opponent can be trapped
    trappable_positions = (
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)),  # 0
        ((6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)),  # 1
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)),  # 2
        ((0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)),  # 3
        ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)),  # 4
        ((5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)),  # 5
        ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)),  # 6
        ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5))  # 7
    )
    return trappable_positions


def get_trap_lines():
    # The optimals may actually be suboptimal, and vice versa, in which case, must move stuff around
    traps_lines = (
        ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)),  # 0: optimal top
        ((5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)),  # 1: optimal bottom
        ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)),  # 2: optimal left
        ((0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)),  # 3: optimal right
        ((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)),  # 4: suboptimal top
        ((4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)),  # 5: suboptimal bottom
        ((0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)),  # 6: suboptimal left
        ((0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4))  # 7: suboptimal right
    )
    return traps_lines


def get_mirrorables():
    # Mirroring may or may not be a winning strategy in *skidding* isolation
    mirrorables = {
        (0, 0): (6, 6),
        (1, 1): (5, 5),
        (2, 2): (4, 4),
        (4, 4): (2, 2),
        (5, 5): (1, 1),
        (6, 6): (0, 0),
        (0, 6): (6, 0),
        (1, 5): (5, 1),
        (2, 4): (4, 2),
        (4, 2): (2, 4),
        (5, 1): (1, 5),
        (6, 0): (0, 6),
        (3, 0): (3, 6),
        (3, 6): (3, 0),
        (3, 1): (3, 5),
        (3, 5): (3, 1),
        (3, 2): (3, 4),
        (3, 4): (3, 2),
        (0, 3): (6, 3),
        (6, 3): (0, 3),
        (1, 3): (5, 3),
        (2, 3): (4, 3),
        (4, 3): (2, 3)
    }
    return mirrorables


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, my_player=None):
        """Score the current game state.

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args:
            game (Board): The board and game state.
            my_player (Player object): This specifies which player you are.

        Returns:
            float: The current state's score, based on your own heuristic.
        """

        if my_player is None:
            return 0
        my_moves = game.get_player_moves(my_player)
        opp_moves = game.get_opponent_moves(my_player)
        board_size = game.width * game.height
        board_remaining = len(game.get_active_moves()) / board_size

        if len(my_moves) > 0 and len(opp_moves) == 0:
            # Seek win
            return float('inf')
        elif len(my_moves) == 0 and len(opp_moves) > 0:
            # Avoid loss
            return float('-inf')
        elif board_remaining > 0.5:
            # Early in the game, play offensive
            return (2 * len(my_moves)) - len(opp_moves)
        else:
            # Late in the game, play defensive
            return len(my_moves) - (2 * len(opp_moves))


class CustomPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=CustomEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.trap_line_pursuing = -1  # a.k.a trappable_positions_index
        self.trap_move_list = ()
        self.trap_move_index = -1
        self.trappable_positions = ()

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """

        # First, try mirroring opponent
        # print("Getting mirror move")
        mirror_move = self.get_mirror_move(game)
        if mirror_move is not None and mirror_move in game.get_player_moves(self):
            # print("Trying mirror move")
            return mirror_move

        # Next, try trapping
        # print("Getting trap move")
        #trap_move = self.get_trap_move(game)
        #if trap_move is not None and trap_move in game.get_player_moves(self):
            # print("Trying trap move")
        #    return trap_move

        #if time_left() > alphabeta_limit:
        #    best_move, utility = minimax(self, game, time_left, depth=self.search_depth)
        #else:
        best_move, utility = alphabeta(self, game, time_left, depth=self.search_depth, alpha=float("-inf"),
                                           beta=float("inf"), my_turn=True)

        my_moves = game.get_player_moves(self)
        if best_move not in my_moves:
            best_move = self.random_move(my_moves)
        if best_move is None:
            best_move = (-1, -1)

        return best_move

        """
    0|1|2|3|4|5|6
    1|-|-|-|-|-|-
    2|-|-|-|-|-|-
    3|-|-|-|-|-|-
    4|-|-|-|-|-|-
    5|-|-|-|-|-|-
    6|-|-|-|-|-|-
    """

    def get_mirror_move(self, game):
        opp_pos = game.get_opponent_position(self)
        # Maybe opponent hasn't gone yet
        if not game.move_is_in_board(opp_pos[0], opp_pos[1]):
            return None

        mirrorables = get_mirrorables()

        """for mirrorable in mirrorables.keys():
            if opp_pos == mirrorable:
                #print("Opp pos (" + str(opp_pos[0]) + "," + str(opp_pos[1]) + ") in mirrorables")
                if game.is_spot_open(mirrorables.get(mirrorable)[0], mirrorables.get(mirrorable)[1]):
                    return mirrorables.get(mirrorable)[0], mirrorables.get(mirrorable)[1]
                else:
                    return None
            #else:
            #    print("Opp pos (" + str(opp_pos[0]) + "," + str(opp_pos[1]) + ") not in mirrorables")
        return None"""
        # Simplified:
        if opp_pos in mirrorables.keys() and game.is_spot_open(mirrorables.get(opp_pos)[0], mirrorables.get(opp_pos)[1]):
            return mirrorables.get(opp_pos)
        else:
            return None


    def get_trap_move(self, game):
        # traps_lines = get_trap_lines()
        trap_moves = get_trap_moves()
        opp_pos = game.get_opponent_position(self)
        self.trappable_positions = get_trappable_positions()

        # Maybe opponent hasn't gone yet
        if not game.move_is_in_board(opp_pos[0], opp_pos[1]):
            return None

        if self.trap_line_pursuing != -1:
            return self.get_trap_move_from_trap_move_list(game, opp_pos)
        else:
            line_index = 0
            found_in_trappable_positions = False
            for line in self.trappable_positions:
                if opp_pos in line:
                    found_in_trappable_positions = True
                    break
                else:
                    line_index += 1
            if found_in_trappable_positions:
                self.trap_line_pursuing = line_index
                self.trap_move_list = trap_moves[self.trap_line_pursuing]
                self.trap_move_index = 0
            else:
                self.trap_line_pursuing = -1
                self.trap_move_list = ()
                self.trap_move_index = -1
        if len(self.trap_move_list) > 0:
            return self.get_trap_move_from_trap_move_list(game, opp_pos)
        return None

    def get_trap_move_from_trap_move_list(self, game, opp_pos):
        my_moves = game.get_player_moves(self)
        if self.trap_move_index == -1:
            move = self.trap_move_list[self.trap_line_pursuing][0]
            # if possible to go to first move and opp is trapped, then set index to 0 and return first move,
            if move in my_moves and opp_pos in self.trappable_positions[self.trap_line_pursuing]:
                self.trap_move_index = 0
                return move
            else:
                # else, set self.trap_line_pursuing = -1, self.trap_move_index = -1, and return none
                self.trap_line_pursuing = -1
                self.trap_move_index = -1
                return None
        else:
            self.trap_move_index += 1
            if self.trap_move_index >= len(self.trap_move_list):
                self.trap_line_pursuing = -1
                self.trap_move_index = -1
                return None
            move = self.trap_move_list[self.trap_move_index]
            # if possible to go to next move and opp is trapped, then increment index and return next move,
            if move in my_moves and opp_pos in self.trappable_positions[self.trap_line_pursuing]:
                return move
            else:
                # else, set self.trap_line_pursuing = -1, self.trap_move_index = -1, and return none
                self.trap_line_pursuing = -1
                self.trap_move_index = -1
                return None

    def random_move(self, my_moves):
        num_moves = len(my_moves)
        if num_moves == 0:
            return None
        return my_moves[randint(0, num_moves-1)]

    def utility(self, game, my_turn):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)


iterative_limit = 40
minimax_limit = 50
alphabeta_limit = 270


def minimax_iterative(player, game, time_left, depth, my_turn=True):
    my_moves = game.get_player_moves(player)
    opp_moves = game.get_opponent_moves(player)
    if depth == 0 or time_left() < iterative_limit or (my_turn and len(my_moves) == 0) or (not my_turn and len(opp_moves) == 0):
        best_move = player.random_move(my_moves)
        if best_move is None:
            best_move = (-1, -1)
        val = player.utility(game, my_turn)
        return best_move, val
    elif my_turn:
        best_val = float("-inf")
        best_move = None
        for my_move in my_moves:
            temp_game, is_over, winner = game.forecast_move(my_move)
            temp_move, temp_val = minimax_iterative(player, temp_game, time_left, depth-1, not my_turn)
            if temp_val > best_val:
                best_val = temp_val
                best_move = my_move
        return best_move, best_val
    else:
        best_val = float("inf")
        best_move = None
        for opp_move in opp_moves:
            temp_game, is_over, winner = game.forecast_move(opp_move)
            temp_move, temp_val = minimax_iterative(player, temp_game, time_left, depth-1, not my_turn)
            if temp_val < best_val:
                best_val = temp_val
                best_move = opp_move
        return best_move, best_val


def minimax(player, game, time_left, depth, my_turn=True):
    """Implementation of the minimax algorithm.
    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    best_move, best_val = minimax_iterative(player, game, time_left, depth, my_turn)
    last_3 = list()
    while time_left() > minimax_limit:
        depth += 1
        temp_move, temp_val = minimax_iterative(player, game, time_left, depth, my_turn)
        if len(last_3) < 3:
            last_3.append(temp_move)
        else:
            last_3[0] = last_3[1]
            last_3[1] = last_3[2]
            last_3[2] = temp_move
            if last_3[0] == last_3[1] and last_3[1] == last_3[2]:  # and last_3[0] == best_move:
                return best_move, best_val
        if temp_val > best_val:
            best_val = temp_val
            best_move = temp_move
    # print("Searched to depth " + str(depth))
    return best_move, best_val


def alphabeta_iterative(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    my_moves = game.get_player_moves(player)
    opp_moves = game.get_opponent_moves(player)

    if depth == 0 or time_left() < iterative_limit or (my_turn and len(my_moves) == 0) or (not my_turn and len(opp_moves) == 0):
        best_move = player.random_move(my_moves)
        if best_move is None:
            best_move = (-1, -1)
        val = player.utility(game, my_turn)
        return best_move, val, (val == float("inf") or val == float("-inf") )
    elif my_turn:
        best_move = None
        for my_move in my_moves:
            temp_game, is_over, winner = game.forecast_move(my_move)
            temp_move, temp_val, completed = alphabeta_iterative(player, temp_game, time_left, depth - 1, alpha, beta, not my_turn)
            if temp_val > alpha:
                alpha = temp_val
                best_move = my_move
            if beta <= alpha:
                break
        return best_move, alpha, is_over
    else:
        best_move = None
        for opp_move in opp_moves:
            temp_game, is_over, winner = game.forecast_move(opp_move)
            temp_move, temp_val, completed = alphabeta_iterative(player, temp_game, time_left, depth - 1, alpha, beta, not my_turn)
            if temp_val < beta:
                beta = temp_val
                best_move = opp_move
            if beta <= alpha:
                break
        return best_move, beta, is_over


def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the alphabeta algorithm.

    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you need
            from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer())
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        alpha (float): Alpha value for pruning
        beta (float): Beta value for pruning
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """

    best_move, best_val, completed = alphabeta_iterative(player, game, time_left, depth, alpha, beta, my_turn)
    #last_3 = list()
    #last_3.append(best_move)
    while time_left() > alphabeta_limit:
        depth += 1
        temp_move, temp_val, temp_completed = alphabeta_iterative(player, game, time_left, depth, alpha, beta, my_turn)
        #if len(last_3) < 3:
        #    last_3.append(temp_move)
        #else:
        #    last_3[0] = last_3[1]
        #    last_3[1] = last_3[2]
        #    last_3[2] = temp_move
        #    if last_3[0] == last_3[1] and last_3[1] == last_3[2]:  # and last_3[0] == best_move:
        #        return best_move, best_val
        if temp_val > best_val and temp_completed:
            best_val = temp_val
            best_move = temp_move
    # print("Searched to depth " + str(depth))
    return best_move, best_val


def get_mirror_move(player, game):
    opp_pos = game.get_opponent_position(player)
    # Maybe opponent hasn't gone yet
    if not game.move_is_in_board(opp_pos[0], opp_pos[1]):
        return None

    mirrorables = get_mirrorables()

    for mirrorable in mirrorables.keys():
        if opp_pos == mirrorable:
            if game.is_spot_open(mirrorables.get(mirrorable)[0], mirrorables.get(mirrorable)[1]):
                return mirrorables.get(mirrorable)[0], mirrorables.get(mirrorable)[1]
            else:
                return None
    return None


"""def get_trap_move(player, game):
    # traps_lines = get_trap_lines()
    trap_moves = get_trap_moves()
    opp_pos = game.get_opponent_position(player)
    self.trappable_positions = get_trappable_positions()

    # Maybe opponent hasn't gone yet
    if not game.move_is_in_board(opp_pos[0], opp_pos[1]):
        return None

    if self.trap_line_pursuing != -1:
        return self.get_trap_move_from_trap_move_list(game, opp_pos)
    else:
        line_index = 0
        found_in_trappable_positions = False
        for line in self.trappable_positions:
            if opp_pos in line:
                found_in_trappable_positions = True
                break
            else:
                line_index += 1
        if found_in_trappable_positions:
            self.trap_line_pursuing = line_index
            self.trap_move_list = trap_moves[self.trap_line_pursuing]
            self.trap_move_index = 0
        else:
            self.trap_line_pursuing = -1
            self.trap_move_list = ()
            self.trap_move_index = -1
    if len(self.trap_move_list) > 0:
        return self.get_trap_move_from_trap_move_list(game, opp_pos)
    return None

def get_trap_move_from_trap_move_list(player, game, opp_pos):
    my_moves = game.get_player_moves(player)
    if player.trap_move_index == -1:
        move = player.trap_move_list[player.trap_line_pursuing][0]
        # if possible to go to first move and opp is trapped, then set index to 0 and return first move,
        if move in my_moves and opp_pos in player.trappable_positions[player.trap_line_pursuing]:
            player.trap_move_index = 0
            return move
        else:
            # else, set self.trap_line_pursuing = -1, self.trap_move_index = -1, and return none
            player.trap_line_pursuing = -1
            player.trap_move_index = -1
            return None
    else:
        player.trap_move_index += 1
        if player.trap_move_index >= len(player.trap_move_list):
            player.trap_line_pursuing = -1
            player.trap_move_index = -1
            return None
        move = player.trap_move_list[player.trap_move_index]
        # if possible to go to next move and opp is trapped, then increment index and return next move,
        if move in my_moves and opp_pos in player.trappable_positions[player.trap_line_pursuing]:
            return move
        else:
            # else, set self.trap_line_pursuing = -1, self.trap_move_index = -1, and return none
            player.trap_line_pursuing = -1
            player.trap_move_index = -1
            return None"""


