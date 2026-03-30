import numpy as np
import torch

from agents.alphazero_net import AlphaZeroNet, preprocess_board
from agents.base_agent import BaseAgent


# Tree Node

class AlphaZeroNode:
    """
    One node in the MCTS tree, representing a board position reached by one move.

    N: visit count — incremented every time a simulation passes through this node
    W: total accumulated value from backpropagation
    Q: mean value = W/N (0.0 if never visited) — "how good was moving here?"
    P: prior probability from the policy network; set at node creation, never changes
    children: dict mapping move (row, col) → child AlphaZeroNode
    is_terminal: True if entering this position ended the game
    is_expanded:  True after the network has been evaluated at this node
    """

    __slots__ = ('N', 'W', 'P', 'children', 'is_terminal', 'is_expanded')

    def __init__(self, P: float = 0.0):
        self.N           = 0
        self.W           = 0.0
        self.P           = P
        self.children    = {}
        self.is_terminal = False
        self.is_expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, parent_N: int, c_puct: float) -> float:
        """
        PUCT = Q + c_puct * P * sqrt(parent_N) / (1 + N)
        Balances exploitation (Q), exploration (low N), and prior trust (P).
        Silver et al. 2017, Methods — 'Selection'.
        """
        return self.Q + c_puct * self.P * (parent_N ** 0.5) / (1 + self.N)


# Candidate Moves
def get_candidate_moves(board: np.ndarray, board_size: int = 9, dist: int = 2) -> list:
    if not np.any(board != 0):
        c = board_size // 2
        return [(r, col) for r in range(c - 2, c + 3)
                         for col in range(c - 2, c + 3)
                         if 0 <= r < board_size and 0 <= col < board_size]

    candidates = set()
    for r, c in zip(*np.where(board != 0)):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == 0:
                    candidates.add((nr, nc))
    return list(candidates)


# Win Detection
def _check_win(board: np.ndarray, move: tuple, player: int, board_size: int) -> bool:
    """
    Check if placing `player` at `move` creates 5-in-a-row.
    Uses the same 4-direction logic as GomokuLogic.check_win().
    """
    r, c = move
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        for sign in [1, -1]:
            nr, nc = r + sign * dr, c + sign * dc
            while 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player:
                count += 1
                nr += sign * dr
                nc += sign * dc
        if count >= 5:
            return True
    return False


# MCTS Core: Expand / Simulate
def _expand(node: AlphaZeroNode, board: np.ndarray, player: int,
            net: AlphaZeroNet, board_size: int) -> float:
    """
    Call the network on this position, then create child nodes for all candidate moves.

    Returns the value estimate for `player` at this position (scalar in [-1, 1]).
    Each child node stores its prior P from the policy output — fixed forever.
    Children that are terminal (win or draw) have is_terminal=True set immediately.
    """
    device = next(net.parameters()).device
    state  = torch.FloatTensor(preprocess_board(board, player)).unsqueeze(0).to(device)

    net.eval()
    with torch.no_grad():
        policy_out, value_out = net(state)

    policy = policy_out.squeeze().cpu().numpy()  # (81,)
    value  = value_out.item()                    # scalar in [-1, 1]

    candidates = get_candidate_moves(board, board_size)
    for move in candidates:
        child_board = board.copy()
        child_board[move] = player

        is_win  = _check_win(child_board, move, player, board_size)
        is_draw = (not is_win) and not np.any(child_board == 0)

        child = AlphaZeroNode(P=float(policy[move[0] * board_size + move[1]]))
        child.is_terminal = is_win or is_draw
        node.children[move] = child

    node.is_expanded = True
    return value


def _simulate(root: AlphaZeroNode, board: np.ndarray, player: int,
              net: AlphaZeroNet, c_puct: float, board_size: int):
    """
    One complete simulation: SELECT → EXPAND → EVALUATE → BACKPROPAGATE.

    SELECT:  descend the tree greedily by PUCT score until reaching an
             unexpanded or terminal node (the 'leaf').
    EXPAND:  call the network at the leaf, create all child nodes with priors.
    EVALUATE: the network's value output (or ±1.0 for terminal nodes).
    BACKPROPAGATE: walk back up path, accumulating value with sign flip at each
                   level (perspective alternates between players).
    """
    node          = root
    path          = [node]
    current_board = board.copy()
    current_player = player

    # Select
    while node.is_expanded and not node.is_terminal and node.children:
        parent_N = node.N
        best_move = max(node.children,
                        key=lambda m: node.children[m].puct_score(parent_N, c_puct))
        current_board = current_board.copy()
        current_board[best_move] = current_player
        current_player = -current_player
        node = node.children[best_move]
        path.append(node)

    # Expand and Evaluate
    if node.is_terminal:
        value = -1.0  # the previous player won, so current_player lost
        # Check for draw: if no empty cells remain
        if not np.any(current_board == 0) and not _check_win(
                current_board,
                (0, 0), -current_player, board_size):
            value = 0.0
    else:
        value = _expand(node, current_board, current_player, net, board_size)

    # BACKPROPAGATE
    # Walk backwards through path, flipping value sign at each level.
    # At the leaf: value is from current_player's perspective.
    # One level up: perspective is -current_player, so flip sign.
    for node in reversed(path):
        node.N += 1
        node.W += value
        value   = -value   # flip perspective for the parent


# Public API: run_mcts / get_pi_and_move
def run_mcts(board: np.ndarray, player: int, net: AlphaZeroNet,
             num_simulations: int, c_puct: float,
             dirichlet_alpha: float, dirichlet_eps: float,
             board_size: int = 9) -> tuple:
    """
    Run MCTS:

    Dirichlet noise is added to root priors when dirichlet_eps > 0 
    
    Returns:
        root: AlphaZeroNode — fully built search tree
        pi:   np.ndarray (board_size²,) — normalised visit counts (training target)
    """
    root = AlphaZeroNode(P=1.0)  # root P is never used in PUCT (no parent)

    # Expand root first so all children have priors before any selection
    _expand(root, board, player, net, board_size)

    # Add Dirichlet noise to root child priors (training only)
    if dirichlet_eps > 0 and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        for move, eta in zip(moves, noise):
            child = root.children[move]
            child.P = (1 - dirichlet_eps) * child.P + dirichlet_eps * eta

    # Run simulations
    for _ in range(num_simulations):
        _simulate(root, board.copy(), player, net, c_puct, board_size)

    # Build the π vector: normalised visit counts over all 81 cells
    pi = np.zeros(board_size * board_size, dtype=np.float32)
    for move, child in root.children.items():
        pi[move[0] * board_size + move[1]] = child.N
    if pi.sum() > 0:
        pi /= pi.sum()

    return root, pi


def get_pi_and_move(root: AlphaZeroNode, board_size: int, tau: float) -> tuple:
    """
    Select a move from the root's children after MCTS.

    tau=1.0 (early game): sample proportionally to visit counts — diverse exploration.
    tau=0.0 (late game):  always pick the most-visited child — best known move.

    Returns:
        pi:   np.ndarray (board_size²,) — MCTS policy (training target)
        move: tuple (row, col)
    """
    moves = list(root.children.keys())
    visit_counts = np.array([root.children[m].N for m in moves], dtype=np.float32)

    # Build full 81-element pi vector
    pi = np.zeros(board_size * board_size, dtype=np.float32)
    for move, n in zip(moves, visit_counts):
        pi[move[0] * board_size + move[1]] = n
    if pi.sum() > 0:
        pi /= pi.sum()

    # Select move
    if tau > 0:
        probs = visit_counts ** (1.0 / tau)
        probs /= probs.sum()
        idx = np.random.choice(len(moves), p=probs)
    else:
        idx = int(np.argmax(visit_counts))

    return pi, moves[idx]


# Agent Wrapper (BaseAgent compatible)
class AlphaZeroMCTS(BaseAgent):
    """
    AlphaZero MCTS agent wrapping run_mcts() for use with eval_agents() and main.py.
    Implements the BaseAgent.predict(board_state) interface.
    No Dirichlet noise during evaluation (add_dirichlet_noise=False by default).
    """

    def __init__(self, net: AlphaZeroNet, player_id: int,
                 num_simulations: int = 400, c_puct: float = 1.5,
                 add_dirichlet_noise: bool = False, board_size: int = 9):
        super().__init__(player_id)
        self.net                 = net
        self.num_simulations     = num_simulations
        self.c_puct              = c_puct
        self.add_dirichlet_noise = add_dirichlet_noise
        self.board_size          = board_size

    def predict(self, board_state: np.ndarray) -> tuple:
        """Return the best move (row, col) for the current board position."""
        self.net.eval()
        d_alpha = 0.3  if self.add_dirichlet_noise else 0.0
        d_eps   = 0.25 if self.add_dirichlet_noise else 0.0

        root, _ = run_mcts(
            board=board_state,
            player=self.player_id,
            net=self.net,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=d_alpha,
            dirichlet_eps=d_eps,
            board_size=self.board_size
        )
        # Greedy: pick most-visited child (no temperature, no noise)
        return max(root.children, key=lambda m: root.children[m].N)
