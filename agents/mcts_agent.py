"""
MCTS Agent — Monte Carlo Tree Search guided by a trained DQN

Instead of random rollouts, we use the DQN's Q-values to evaluate leaf
positions. This is the same idea as AlphaZero at inference time:
the neural network guides the search, making it far stronger than a pure
DQN which has no lookahead at all.

How to use:
    from agents.dqn_rohan import DQNAgentRohan
    from agents.mcts_agent import MCTSAgent

    dqn = DQNAgentRohan(player_id=1, board_size=9)
    dqn.load_model("models_rohan/final.pt")
    dqn.epsilon = 0.0   # Pure exploitation — no random moves during search

    agent = MCTSAgent(player_id=1, dqn_agent=dqn, num_simulations=300)
    move = agent.predict(board_state)   # Works with all existing eval scripts
"""

import math
import random
import numpy as np
import torch
from agents.base_agent import BaseAgent
from agents.dqn_rohan import preprocess_board   # Reuse the board encoding DQN was trained with


# =============================================================================
# HELPER: Win Detection
# =============================================================================

def _check_winner(board, last_move, player, board_size):
    """
    Check whether placing `player`'s stone at `last_move` created 5 in a row.

    Called AFTER the stone has already been written onto the board.
    Returns True if `player` has won, False otherwise.

    We only need to check lines passing through `last_move` — any 5-in-a-row
    that exists must include the piece that was just placed.

    Args:
        board     : numpy array (board_size × board_size)
        last_move : (row, col) tuple — the move just played
        player    : 1 or -1 — the player who just moved
        board_size: int (9 for our game)
    """
    row, col = last_move

    # Check all four directions: horizontal, vertical, two diagonals
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1  # Start at 1 for the stone we just placed

        # Count consecutive pieces in the forward direction
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r += dr
            c += dc

        # Count consecutive pieces in the backward direction
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc

        if count >= 5:
            return True  # 5 or more in a row — this player has won

    return False


# =============================================================================
# THE TREE NODE
# =============================================================================

class MCTSNode:
    """
    One node in the MCTS search tree.

    Each node represents a specific board state reached by some sequence of
    moves from the root position. The tree grows one node per simulation.

    Key stored quantities:
        N  — visit count: how many times MCTS has passed through this node
        W  — total value: sum of all values seen during backpropagation
        Q  — mean value = W / N (computed as a property)

    Sign convention for W and Q:
        Q represents "how good was it for the player who MOVED HERE (the
        parent's player_to_move) to choose this move?"
        +1.0 = great (that player went on to win from here)
        -1.0 = terrible (that player went on to lose)
        This means every player at their turn simply maximises Q — consistent
        across all tree depths with no special cases needed.
    """

    # __slots__ tells Python to allocate memory more efficiently by not
    # creating a full __dict__ for each node. With thousands of nodes in
    # the tree, this meaningfully reduces memory use.
    __slots__ = ('board', 'parent', 'move', 'player_to_move',
                 'N', 'W', 'children', 'untried_moves')

    def __init__(self, board, parent=None, move=None, player_to_move=1):
        """
        board          : numpy array (9×9) — the board state AT this node
        parent         : MCTSNode that this was expanded from (None at root)
        move           : (row, col) that created this node from parent (None at root)
        player_to_move : whose turn it is to move FROM this position (1 or -1)
        """
        # Store a copy of the board — we must never mutate a node's board
        # because multiple simulations share references to parent boards.
        self.board = board.copy()

        self.parent = parent            # Link up the tree
        self.move = move                # Move that created this node
        self.player_to_move = player_to_move  # Who moves NEXT from here

        # ── Statistics (updated during backpropagation) ──────────────────────
        self.N = 0      # How many times this node has been visited
        self.W = 0.0    # Total accumulated value (see sign convention above)

        # ── Children ─────────────────────────────────────────────────────────
        # Maps (row, col) → MCTSNode for every move we have ALREADY expanded.
        # Starts empty; new children are added one per simulation during expansion.
        self.children = {}

        # ── Untried moves ─────────────────────────────────────────────────────
        # List of valid moves that have NOT yet been expanded into child nodes.
        # As we expand, we pop moves from here and create children.
        # When this list is empty, the node is "fully expanded."
        valid_moves = list(zip(*np.where(board == 0)))  # All empty cells
        self.untried_moves = valid_moves[:]
        random.shuffle(self.untried_moves)  # Shuffle for move diversity across runs

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def Q(self):
        """
        Mean value of this node. Returns 0.0 if never visited.

        W / N gives the running average of all values that have been
        backpropagated through this node. A high Q means that the move
        leading here was consistently good for the player who made it.
        """
        return self.W / self.N if self.N > 0 else 0.0

    # ── UCB Score ─────────────────────────────────────────────────────────────

    def ucb(self, c):
        """
        UCB1 (Upper Confidence Bound) score — used by the parent to decide
        which child to visit during the SELECTION phase.

        Formula:
            UCB = Q  +  c × sqrt( ln(N_parent) / N_child )
                  ↑              ↑
             exploitation     exploration

        - Q is high for moves that have won a lot (exploit what works)
        - The sqrt term is high for moves that have been visited rarely
          (explore moves we haven't tried much yet)
        - c controls the balance — higher c = more exploration

        Nodes with N=0 (never visited) return +infinity, guaranteeing
        they get explored before any node is visited twice.
        """
        if self.N == 0:
            return float('inf')

        # Exploration term: grows as parent is visited more, shrinks as
        # this child is visited more — naturally drives the search to balance
        # returning to good nodes vs trying less-explored ones.
        exploration = c * math.sqrt(math.log(self.parent.N) / self.N)

        return self.Q + exploration

    # ── Selection Helpers ─────────────────────────────────────────────────────

    def is_leaf(self):
        """
        True if this node has moves we haven't tried yet.
        A "leaf" in MCTS means "a node we can still expand."

        Note: in classical MCTS terminology, "leaf" sometimes means
        "has no children at all." Here we use it to mean "has untried moves"
        which is more practical for expansion.
        """
        return bool(self.untried_moves)

    def best_child(self, c):
        """
        Select the child with the highest UCB score.
        Called during SELECTION to walk down the tree towards the frontier.

        Every player at their turn maximises UCB — this is consistent because
        Q always represents "value for the player who moved to this child."
        """
        return max(self.children.values(), key=lambda child: child.ucb(c))

    def __repr__(self):
        return (f"MCTSNode(move={self.move}, N={self.N}, "
                f"Q={self.Q:.3f}, player_to_move={self.player_to_move})")


# =============================================================================
# THE MCTS AGENT
# =============================================================================

class MCTSAgent(BaseAgent):
    """
    DQN-guided Monte Carlo Tree Search agent.

    Runs `num_simulations` MCTS iterations per move, using the trained DQN
    to evaluate leaf positions instead of random rollouts.

    The more simulations, the stronger the play — but the slower the move.
    For our 9×9 board, 200–400 simulations typically take 0.3–1.0 seconds
    on an M4 MacBook (MPS-accelerated DQN inference).
    """

    def __init__(self, player_id, dqn_agent, board_size=9,
                 num_simulations=300, c_puct=1.4):
        """
        player_id      : 1 or -1 — which player this agent controls
        dqn_agent      : A trained DQNAgentRohan (or DQNAgent) with weights loaded
                         and epsilon set to 0.0
        board_size     : Board dimensions (default 9 for 9×9 Gomoku)
        num_simulations: How many MCTS simulations per move.
                         200 = fast (~0.3s/move), 400 = strong (~0.8s/move)
        c_puct         : Exploration constant in UCB. 1.4 = √2, the classical
                         value. Higher values push the search to explore more
                         widely; lower values focus on the best-looking lines.
        """
        super().__init__(player_id)
        self.dqn_agent = dqn_agent
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Put DQN permanently in eval mode — we only do inference here,
        # never training. This disables dropout and batch norm training
        # behaviour, giving deterministic and slightly faster forward passes.
        self.dqn_agent.q_network.eval()

    # =========================================================================
    # PUBLIC: predict()
    # =========================================================================

    def predict(self, board_state):
        """
        Run MCTS from `board_state` and return the best (row, col) move.

        This is the standard BaseAgent interface — compatible with eval_agents(),
        main.py, and all existing evaluation scripts.

        Steps:
            1. Handle trivial cases (no moves, single move)
            2. Build a root MCTSNode for the current position
            3. Run `num_simulations` simulations, each growing the tree
            4. Return the move leading to the most-visited child
        """
        valid_moves = list(zip(*np.where(board_state == 0)))

        # Edge case: board full
        if not valid_moves:
            return None

        # Edge case: only one legal move — no point searching
        if len(valid_moves) == 1:
            return valid_moves[0]

        # Build the root of our search tree.
        # player_to_move = self.player_id because it's our turn.
        root = MCTSNode(board_state, player_to_move=self.player_id)

        # Run the simulations. Each one descends to a leaf, evaluates it,
        # and backpropagates the result up to the root.
        for _ in range(self.num_simulations):
            self._simulate(root)

        # Safety: if no children were created (shouldn't happen with valid_moves)
        if not root.children:
            return random.choice(valid_moves)

        # Pick the most-visited child.
        # Why most-visited rather than highest Q?
        # - Q can be high from a single lucky simulation
        # - High visit count means MCTS kept returning to explore that line —
        #   a more robust signal of sustained quality
        return max(root.children, key=lambda move: root.children[move].N)

    # =========================================================================
    # CORE: One simulation (SELECT → EXPAND → EVALUATE → BACKPROP)
    # =========================================================================

    def _simulate(self, root):
        """
        Run one complete MCTS simulation from the root.

        This is the heart of the algorithm. Called `num_simulations` times
        per move. Each call:
            - Grows the tree by exactly one new node (EXPAND)
            - Updates visit counts and values along the path (BACKPROP)
        """

        node = root
        path = [node]   # Track every node we visit for backpropagation

        # ── PHASE 1: SELECTION ────────────────────────────────────────────────
        # Walk DOWN the tree, always picking the child with the highest UCB
        # score, until we reach a node that still has unexplored moves (a leaf).
        #
        # The condition "not node.is_leaf()" means untried_moves is empty —
        # this node has been fully expanded, all moves have child nodes.
        # The condition "node.children" ensures we don't descend into a
        # terminal node that has no children (game over).
        while not node.is_leaf() and node.children:
            node = node.best_child(self.c_puct)  # UCB-guided descent
            path.append(node)

        # ── PHASE 2: EXPANSION ────────────────────────────────────────────────
        # We've reached a node with at least one untried move.
        # Pick one of those unexplored moves and create a new child node.
        # This is how the tree GROWS — one new node per simulation.

        if node.untried_moves:

            # Pop the next unexplored move (shuffled at node creation for diversity)
            move = node.untried_moves.pop()

            # Who is about to make this move?
            moving_player = node.player_to_move

            # Apply the move to get the new board state
            new_board = node.board.copy()
            new_board[move] = moving_player

            # After this move, it's the other player's turn
            next_player = -moving_player

            # Create the new child node
            child = MCTSNode(
                new_board,
                parent=node,
                move=move,
                player_to_move=next_player
            )
            node.children[move] = child  # Add to parent's children dictionary
            path.append(child)           # Include in backprop path

            # ── PHASE 3: EVALUATION ───────────────────────────────────────────
            # Estimate the value of the newly created child position.
            # Three cases:

            if _check_winner(new_board, move, moving_player, self.board_size):
                # ── Case 1: TERMINAL WIN ──────────────────────────────────────
                # moving_player won by making this move.
                # Value = +1.0 from moving_player's perspective —
                # it was a great move for them.
                child.untried_moves = []   # Game over — mark as terminal
                value = 1.0

            elif not np.any(new_board == 0):
                # ── Case 2: TERMINAL DRAW (board full) ───────────────────────
                child.untried_moves = []   # Game over — mark as terminal
                value = 0.0

            else:
                # ── Case 3: NON-TERMINAL ──────────────────────────────────────
                # Ask the DQN how good this position is for next_player
                # (the player who is about to move from the child node).
                dqn_val = self._dqn_value(new_board, next_player)

                # SIGN FLIP: dqn_val is from next_player's perspective.
                # We need value from moving_player's perspective
                # (since Q = "how good for the player who moved to here").
                # next_player's gain = moving_player's loss, so we negate.
                value = -dqn_val

        else:
            # No untried moves AND no children: this is a terminal node
            # that was reached by selection (we already marked it terminal).
            # Treat as draw since we don't know the exact outcome.
            value = 0.0

        # ── PHASE 4: BACKPROPAGATION ──────────────────────────────────────────
        # Walk back UP the path from the newly expanded node to the root,
        # updating N and W at every node we visited.
        #
        # The sign of `value` ALTERNATES at each level:
        # - At the leaf: value is from moving_player's perspective (+1 if they won)
        # - One level up: flip to opponent's perspective (-1 for opponent)
        # - Two levels up: flip back (+1 from original player's view)
        # - ...and so on, all the way up to the root
        #
        # This works because our sign convention is:
        # "W/Q = value for the player who moved to create this node"
        # Every player at their turn then naturally maximises Q.
        self._backpropagate(path, value)

    # =========================================================================
    # BACKPROPAGATION
    # =========================================================================

    def _backpropagate(self, path, value):
        """
        Walk up the path from leaf to root, updating statistics.

        For each node in reverse order (leaf → root):
            1. Increment visit count N by 1
            2. Add current `value` to total value W
            3. Flip the sign of value for the next level up

        Why reversed? We go from the deepest new node (end of path) back
        to the root, flipping the perspective at each level.

        After backprop, root.N = total simulations run so far,
        and root.Q = running average of how good the root position
        has been found to be (should be near 0 for balanced positions).
        """
        for node in reversed(path):
            node.N += 1       # This node was visited once more
            node.W += value   # Accumulate the value
            value = -value    # Flip perspective for the next level up

    # =========================================================================
    # DQN LEAF EVALUATION
    # =========================================================================

    def _dqn_value(self, board, player):
        """
        Use the trained DQN to estimate how good `board` is for `player`.

        Returns a float in (-1.0, +1.0):
            +1.0 → DQN thinks `player` is winning strongly
             0.0 → position looks balanced / uncertain
            -1.0 → DQN thinks `player` is losing

        Steps:
            1. Encode the board into the 3-channel format the DQN was trained on
            2. Run a forward pass (no gradients — inference only)
            3. Extract Q-values as a 9×9 grid
            4. Mask occupied cells with -infinity (they're not valid moves)
            5. Take the maximum Q-value over valid moves
            6. Apply tanh to squash from unbounded float to (-1, +1)

        Why tanh?
            DQN Q-values are unbounded floats. MCTS expects values in [-1, 1]
            representing win/loss probabilities. tanh maps any real number to
            (-1, +1): very large positive Q → near +1, very negative → near -1.
            We divide by 2.0 for a softer curve — without it, even moderate
            Q-values would be squashed to near ±1, losing nuance.
        """
        # Encode board into (1, 3, 9, 9) tensor using the same preprocessing
        # function used during training — critical for correct inference
        state_tensor = torch.FloatTensor(
            preprocess_board(board, player)   # Returns (3, 9, 9) numpy array
        ).unsqueeze(0).to(self.dqn_agent.device)   # Add batch dim → (1, 3, 9, 9)

        # Forward pass — torch.no_grad() prevents storing gradients,
        # saving memory and time since we never call .backward() here
        with torch.no_grad():
            q_values = self.dqn_agent.q_network(state_tensor).squeeze()
            # squeeze() removes the batch dimension: (1, 81) → (81,)

        # Reshape flat Q-values to match board layout: (81,) → (9, 9)
        q_np = q_values.cpu().numpy().reshape(self.board_size, self.board_size)

        # Mask occupied cells — they are illegal moves and should never
        # be selected. Setting to -inf means they won't affect np.max().
        q_np[board != 0] = -np.inf

        # Get Q-values for valid (empty) cells only
        valid_q = q_np[board == 0]

        if len(valid_q) == 0:
            return 0.0   # Board is full — no valid moves (shouldn't happen here)

        # Maximum Q over valid moves = DQN's best estimate for this player
        max_q = float(np.max(valid_q))

        # tanh squashes to (-1, +1). /2.0 softens the curve so mid-range
        # Q-values don't immediately saturate to ±1.
        return float(np.tanh(max_q / 2.0))
