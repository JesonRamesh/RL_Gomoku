"""
Threat Detection System for Gomoku
Scans board for opponent threats that require blocking.
"""

import numpy as np
from typing import List, Tuple, Dict

class ThreatDetector:
    """
    Detects threats on a Gomoku board.
    
    Threat priority:
    - Open-4: _ X X X X _ (urgency 5, must block)
    - Half-4: X X X X | or | X X X X (urgency 4)
    - Open-3: _ X X X _ (urgency 3)
    - Half-3: X X X | or | X X X (urgency 2)
    """
    
    def __init__(self, board_size: int = 9, win_length: int = 5):
        """
        Initialize threat detector.
        
        Args:
            board_size: Size of the Gomoku board (default 9x9)
            win_length: Length needed to win (default 5)
        """
        self.board_size = board_size
        self.win_length = win_length
        
        # Define directions: (row_delta, col_delta)
        self.directions = [
            (0, 1),   # Horizontal →
            (1, 0),   # Vertical ↓
            (1, 1),   # Diagonal ↘
            (1, -1),  # Diagonal ↙
        ]
        
    def find_all_threats(self, board: np.ndarray, player_id: int) -> List[Dict]:
        """
        Find all threats for a given player.
        
        Args:
            board: 9x9 numpy array (0=empty, 1=player1, -1=player2)
            player_id: Which player's threats to find (1 or -1)
            
        Returns:
            List of threat dictionaries, sorted by urgency (highest first)
            Each dict contains:
                - 'position': (row, col) where threat can be blocked
                - 'threat_type': 'open-4', 'half-4', 'open-3', 'half-3'
                - 'urgency': int 1-5 (5=most urgent)
                - 'direction': (dr, dc) direction of threat
                - 'sequence': list of positions forming the threat
        """
        threats = []
        
        # Scan every position on the board
        for row in range(self.board_size):
            for col in range(self.board_size):
                # Check all 4 directions from this position
                for direction in self.directions:
                    threat = self._check_threat_from_position(
                        board, row, col, direction, player_id
                    )
                    if threat:
                        threats.append(threat)
        
        # Remove duplicates (same threat detected from different positions)
        threats = self._deduplicate_threats(threats)
        
        # Sort by urgency (highest first)
        threats.sort(key=lambda t: t['urgency'], reverse=True)
        
        return threats
    
    def must_block_positions(self, board: np.ndarray, player_id: int) -> List[Tuple[int, int]]:
        """
        Returns positions that MUST be blocked this turn (urgency 4+).
        
        This is the main function used for training data generation.
        
        Args:
            board: 9x9 numpy array
            player_id: Opponent's ID (threats made by this player)
            
        Returns:
            List of (row, col) positions that must be blocked
        """
        threats = self.find_all_threats(board, player_id)
        
        # Filter for critical threats only (urgency 4 or 5)
        critical_threats = [t for t in threats if t['urgency'] >= 4]
        
        # Extract blocking positions (deduplicated)
        blocking_positions = []
        seen = set()
        
        for threat in critical_threats:
            pos = threat['position']
            if pos not in seen:
                blocking_positions.append(pos)
                seen.add(pos)
            
            # For open-4, add alternative blocking position too
            if 'alternative_block' in threat:
                alt_pos = threat['alternative_block']
                if alt_pos not in seen:
                    blocking_positions.append(alt_pos)
                    seen.add(alt_pos)
        
        return blocking_positions
    
    def _check_threat_from_position(
        self, 
        board: np.ndarray, 
        row: int, 
        col: int, 
        direction: Tuple[int, int],
        player_id: int
    ) -> Dict:
        """
        Check if there's a threat starting from (row, col) in given direction.
        
        This is the core logic that detects patterns like:
        - _ X X X X _ (open-4)
        - X X X X | (half-4)
        - _ X X X _ (open-3)
        etc.
        """
        dr, dc = direction
        
        # Extract sequence of 6 positions in this direction
        # (need 6 to check for 4-in-a-row with gaps)
        sequence = []
        for i in range(6):
            r = row + i * dr
            c = col + i * dc
            
            # Check bounds
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                break
            
            sequence.append((r, c, board[r, c]))
        
        # Need at least 5 positions to form a threat
        if len(sequence) < 5:
            return None
        
        # Convert to values list for easier pattern matching
        values = [val for _, _, val in sequence]
        
        # Count consecutive stones and gaps
        threat = self._analyze_sequence(values, sequence, player_id)
        
        return threat
    
    def _analyze_sequence(
        self, 
        values: List[int], 
        positions: List[Tuple],
        player_id: int
    ) -> Dict:
        """
        Analyze a sequence of 5-6 positions for threat patterns.
        
        Patterns we're looking for:
        - [0, P, P, P, P, 0]: Open-4 (urgency 5)
        - [P, P, P, P, 0]: Half-4 (urgency 4)
        - [0, P, P, P, P, X]: Half-4 blocked one side (urgency 4)
        - [0, P, P, P, 0]: Open-3 (urgency 3)
        - [P, P, P, 0]: Half-3 (urgency 2)
        
        P = player_id
        0 = empty
        X = opponent or wall
        """
        # Pattern: Open-4 [_, X, X, X, X, _]
        if len(values) >= 6:
            if (values[0] == 0 and 
                values[1] == player_id and 
                values[2] == player_id and 
                values[3] == player_id and 
                values[4] == player_id and 
                values[5] == 0):
                
                # Two possible blocking positions (both ends)
                return {
                    'position': positions[0][:2],  # Can block left side
                    'threat_type': 'open-4',
                    'urgency': 5,
                    'direction': (positions[1][0] - positions[0][0], 
                                 positions[1][1] - positions[0][1]),
                    'sequence': [p[:2] for p in positions[:6]],
                    'alternative_block': positions[5][:2]  # Or right side
                }
        
        # Pattern: Half-4 [X, X, X, X, _]
        if len(values) >= 5:
            # Check [P, P, P, P, 0]
            if (values[0] == player_id and 
                values[1] == player_id and 
                values[2] == player_id and 
                values[3] == player_id and 
                values[4] == 0):
                
                return {
                    'position': positions[4][:2],
                    'threat_type': 'half-4',
                    'urgency': 4,
                    'direction': (positions[1][0] - positions[0][0], 
                                 positions[1][1] - positions[0][1]),
                    'sequence': [p[:2] for p in positions[:5]]
                }
            
            # Check [0, P, P, P, P] with next blocked
            if (values[0] == 0 and 
                values[1] == player_id and 
                values[2] == player_id and 
                values[3] == player_id and 
                values[4] == player_id):
                
                return {
                    'position': positions[0][:2],
                    'threat_type': 'half-4',
                    'urgency': 4,
                    'direction': (positions[1][0] - positions[0][0], 
                                 positions[1][1] - positions[0][1]),
                    'sequence': [p[:2] for p in positions[:5]]
                }
        
        # Pattern: Open-3 [_, X, X, X, _]
        if len(values) >= 5:
            if (values[0] == 0 and 
                values[1] == player_id and 
                values[2] == player_id and 
                values[3] == player_id and 
                values[4] == 0):
                
                return {
                    'position': positions[0][:2],  # Can block either end
                    'threat_type': 'open-3',
                    'urgency': 3,
                    'direction': (positions[1][0] - positions[0][0], 
                                 positions[1][1] - positions[0][1]),
                    'sequence': [p[:2] for p in positions[:5]],
                    'alternative_block': positions[4][:2]
                }
        
        # Pattern: Half-3 [X, X, X, _]
        if len(values) >= 4:
            if (values[0] == player_id and 
                values[1] == player_id and 
                values[2] == player_id and 
                values[3] == 0):
                
                return {
                    'position': positions[3][:2],
                    'threat_type': 'half-3',
                    'urgency': 2,
                    'direction': (positions[1][0] - positions[0][0], 
                                 positions[1][1] - positions[0][1]),
                    'sequence': [p[:2] for p in positions[:4]]
                }
        
        # No threat found
        return None
    
    def _deduplicate_threats(self, threats: List[Dict]) -> List[Dict]:
        """
        Remove duplicate threats (same position + same threat type).
        
        Why duplicates happen:
        - Scanning from different starting positions detects same threat
        - Example: _ X X X X _ detected from both ends
        """
        unique = []
        seen = set()
        
        for threat in threats:
            # Create unique key: position + threat_type
            key = (threat['position'], threat['threat_type'])
            
            if key not in seen:
                unique.append(threat)
                seen.add(key)
        
        return unique
    
    def visualize_threats(self, board: np.ndarray, player_id: int) -> str:
        """
        Create ASCII visualization of board with threats marked.
        Useful for debugging and understanding.
        
        Returns:
            String representation of board with threat markers
        """
        threats = self.find_all_threats(board, player_id)
        blocking_positions = set(self.must_block_positions(board, player_id))
        
        # Create string representation
        lines = []
        lines.append("\n" + "="*50)
        lines.append(f"Threat Analysis for Player {player_id}")
        lines.append("="*50)
        
        # Board visualization
        for row in range(self.board_size):
            line = ""
            for col in range(self.board_size):
                if (row, col) in blocking_positions:
                    line += "🚨"  # Must block here!
                elif board[row, col] == 1:
                    line += "⚫"  # Player 1 stone
                elif board[row, col] == -1:
                    line += "⚪"  # Player 2 stone
                else:
                    line += "·"   # Empty
            lines.append(line)
        
        # Threat summary
        lines.append("\nThreats Found:")
        for i, threat in enumerate(threats[:5], 1):  # Show top 5
            lines.append(f"{i}. {threat['threat_type'].upper()} at {threat['position']} "
                        f"(urgency: {threat['urgency']})")
        
        if len(threats) > 5:
            lines.append(f"... and {len(threats) - 5} more threats")
        
        lines.append("="*50 + "\n")
        
        return "\n".join(lines)


# ============================================================
# UNIT TESTS
# ============================================================

def run_tests():
    """
    Comprehensive test suite for ThreatDetector.
    Run this to verify everything works correctly.
    """
    print("🧪 Running ThreatDetector Tests...\n")
    
    detector = ThreatDetector(board_size=9, win_length=5)
    
    # Test 1: Horizontal Open-4
    print("Test 1: Horizontal Open-4 Detection")
    board = np.zeros((9, 9), dtype=int)
    board[7, 5:9] = 1  # Place _ 1 1 1 1 _ at row 7
    threats = detector.find_all_threats(board, player_id=1)
    blocking = detector.must_block_positions(board, player_id=1)
    
    # FIXED: Open-4 has TWO valid blocking positions (both ends)
    assert len(blocking) == 2, f"Expected 2 blocking positions for open-4, got {len(blocking)}"
    assert (7, 4) in blocking, f"Should include left blocking position (7,4)"
    assert (7, 9) in blocking, f"Should include right blocking position (7,9)"
    assert threats[0]['threat_type'] == 'open-4', f"Wrong threat type: {threats[0]['threat_type']}"
    assert threats[0]['urgency'] == 5, f"Wrong urgency: {threats[0]['urgency']}"
    print(f"✅ PASS: Detected open-4 with 2 blocking positions: {blocking}\n")
    
    # Test 2: Vertical Half-4
    print("Test 2: Vertical Half-4 Detection")
    board = np.zeros((9, 9), dtype=int)
    board[3:7, 7] = -1  # Place -1 -1 -1 -1 in column 7
    threats = detector.find_all_threats(board, player_id=-1)
    blocking = detector.must_block_positions(board, player_id=-1)
    
    assert len(blocking) >= 1, f"Expected blocking positions, got {len(blocking)}"
    assert (7, 7) in blocking or (2, 7) in blocking, f"Wrong blocking position"
    print(f"✅ PASS: Detected vertical half-4 correctly (blocks: {blocking})\n")
    
    # Test 3: Diagonal Open-3
    print("Test 3: Diagonal Open-3 Detection")
    board = np.zeros((9, 9), dtype=int)
    board[5, 5] = board[6, 6] = board[7, 7] = 1  # Diagonal ↘
    threats = detector.find_all_threats(board, player_id=1)
    
    open_3_threats = [t for t in threats if t['threat_type'] == 'open-3']
    assert len(open_3_threats) > 0, "Should detect open-3 threat"
    assert open_3_threats[0]['urgency'] == 3, f"Wrong urgency for open-3"
    print("✅ PASS: Detected diagonal open-3 correctly\n")
    
    # Test 4: No Threats (Empty Board)
    print("Test 4: Empty Board (No Threats)")
    board = np.zeros((9, 9), dtype=int)
    threats = detector.find_all_threats(board, player_id=1)
    blocking = detector.must_block_positions(board, player_id=1)
    
    assert len(threats) == 0, f"Empty board should have no threats, found {len(threats)}"
    assert len(blocking) == 0, f"Empty board should have no blocking positions"
    print("✅ PASS: Correctly returns empty for no threats\n")
    
    # Test 5: Multiple Simultaneous Threats
    print("Test 5: Multiple Simultaneous Threats")
    board = np.zeros((9, 9), dtype=int)
    board[7, 5:9] = 1   # Horizontal open-4
    board[3:7, 7] = 1   # Vertical half-4
    threats = detector.find_all_threats(board, player_id=1)
    blocking = detector.must_block_positions(board, player_id=1)
    
    assert len(threats) >= 2, f"Should detect at least 2 threats, found {len(threats)}"
    # Open-4 gives 2 blocks + half-4 gives 1+ block = at least 3 total
    assert len(blocking) >= 3, f"Should have at least 3 blocking positions, got {len(blocking)}"
    print(f"✅ PASS: Detected multiple threats correctly ({len(threats)} threats, {len(blocking)} blocks)\n")
    
    # Test 6: Edge Case - Threat at Board Boundary
    print("Test 6: Threat Near Board Edge")
    board = np.zeros((9, 9), dtype=int)
    board[0, 0:4] = -1  # Top-left corner threat
    threats = detector.find_all_threats(board, player_id=-1)
    
    assert len(threats) > 0, "Should detect threat near boundary"
    print("✅ PASS: Handles board boundaries correctly\n")
    
    # Test 7: Visualization Test
    print("Test 7: Visualization")
    board = np.zeros((9, 9), dtype=int)
    board[7, 5:9] = 1
    vis = detector.visualize_threats(board, player_id=1)
    
    assert "🚨" in vis, "Visualization should mark blocking positions"
    assert "open-4" in vis.lower(), "Visualization should list threat type"
    print("✅ PASS: Visualization works\n")
    
    # Test 8: Performance Test
    print("Test 8: Performance Test (1000 board scans)")
    import time
    
    board = np.zeros((9, 9), dtype=int)
    # Add some random stones
    for _ in range(30):
        r, c = np.random.randint(0, 9, 2)
        board[r, c] = np.random.choice([1, -1])
    
    start = time.time()
    for _ in range(1000):
        threats = detector.find_all_threats(board, player_id=1)
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / 1000) * 1000
    print(f"Average scan time: {avg_time_ms:.2f} ms")
    assert avg_time_ms < 10, f"Too slow! Average: {avg_time_ms:.2f} ms (target: <10ms)"
    print("✅ PASS: Performance acceptable\n")
    
    print("="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\n✅ ThreatDetector is working correctly and ready for Phase 2A.2")
    print("\n📝 Key Insight: Open-4 threats have 2 valid blocking positions!")
    print("   This is CORRECT behavior - agent can block either end.")


if __name__ == "__main__":
    # Run tests when file is executed
    run_tests()