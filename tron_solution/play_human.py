import pygame
import sys
import time
import numpy as np
from enum import IntEnum

# Import our custom environment and opponents
# Ensure these paths match your project structure
try:
    from env.tron_env import TronEnv
    from env.opponents import MinimaxOpponent
except ImportError:
    # Fallback if running from root directory differently
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
    from tron_solution.env.tron_env import TronEnv
    from tron_solution.env.opponents import MinimaxOpponent

# --- Constants ---
CELL_SIZE = 20
GRID_WIDTH = 32
GRID_HEIGHT = 32
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 120, 255)      # Player
RED = (255, 50, 50)       # Opponent (AI)
GRAY = (40, 40, 40)       # Walls/Trails
TEXT_COLOR = (255, 255, 255)

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class HumanDuel:
    def __init__(self, difficulty='hard'):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tron Duel: Human vs Minimax AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 48, bold=True)

        # Initialize Environment
        # We use a raw env for control, handling the step logic manually
        self.env = TronEnv()
        
        # Initialize Opponent
        if difficulty == 'hard':
            self.opponent = MinimaxOpponent(depth=4) # Deep search for hard mode
        elif difficulty == 'medium':
            self.opponent = MinimaxOpponent(depth=2)
        else:
            self.opponent = MinimaxOpponent(depth=1)

        self.reset_game()

    def reset_game(self):
        self.observation, self.info = self.env.reset()
        self.game_over = False
        self.winner = None
        self.message = "Get Ready!"
        self.message_timer = 60 # frames
        
        # Extract state for AI
        self.grid = self.env.grid.copy()
        self.p1_pos = self.env.p1_pos
        self.p2_pos = self.env.p2_pos
        self.p1_dir = self.env.p1_dir
        self.p2_dir = self.env.p2_dir

    def get_opponent_move(self):
        """Calculate AI move using Minimax"""
        # The opponent needs the current grid and positions
        # We pass a copy of the state to the opponent logic
        state = {
            'grid': self.grid,
            'p1_pos': self.p1_pos,
            'p2_pos': self.p2_pos,
            'p1_dir': self.p1_dir,
            'p2_dir': self.p2_dir,
            'width': GRID_WIDTH,
            'height': GRID_HEIGHT
        }
        return self.opponent.get_action(state)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        action = None
        
        # Map keys to actions, preventing 180 turns
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            if self.p1_dir != Action.DOWN:
                action = Action.UP
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            if self.p1_dir != Action.UP:
                action = Action.DOWN
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            if self.p1_dir != Action.RIGHT:
                action = Action.LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            if self.p1_dir != Action.LEFT:
                action = Action.RIGHT
        
        return action

    def update_logic(self, p1_action, p2_action):
        """Manual step execution to sync with GUI"""
        # 1. Move Player 1 (Human)
        if p1_action == Action.UP:
            self.p1_pos = (self.p1_pos[0] - 1, self.p1_pos[1])
            self.p1_dir = Action.UP
        elif p1_action == Action.DOWN:
            self.p1_pos = (self.p1_pos[0] + 1, self.p1_pos[1])
            self.p1_dir = Action.DOWN
        elif p1_action == Action.LEFT:
            self.p1_pos = (self.p1_pos[0], self.p1_pos[1] - 1)
            self.p1_dir = Action.LEFT
        elif p1_action == Action.RIGHT:
            self.p1_pos = (self.p1_pos[0], self.p1_pos[1] + 1)
            self.p1_dir = Action.RIGHT

        # 2. Move Player 2 (AI)
        if p2_action == Action.UP:
            self.p2_pos = (self.p2_pos[0] - 1, self.p2_pos[1])
            self.p2_dir = Action.UP
        elif p2_action == Action.DOWN:
            self.p2_pos = (self.p2_pos[0] + 1, self.p2_pos[1])
            self.p2_dir = Action.DOWN
        elif p2_action == Action.LEFT:
            self.p2_pos = (self.p2_pos[0], self.p2_pos[1] - 1)
            self.p2_dir = Action.LEFT
        elif p2_action == Action.RIGHT:
            self.p2_pos = (self.p2_pos[0], self.p2_pos[1] + 1)
            self.p2_dir = Action.RIGHT

        # 3. Check Collisions
        p1_hit = False
        p2_hit = False
        reason = ""

        # Check bounds and trails for P1
        if not (0 <= self.p1_pos[0] < GRID_HEIGHT and 0 <= self.p1_pos[1] < GRID_WIDTH):
            p1_hit = True
            reason = "Wall"
        elif self.grid[self.p1_pos] != 0:
            p1_hit = True
            reason = "Trail"

        # Check bounds and trails for P2
        if not (0 <= self.p2_pos[0] < GRID_HEIGHT and 0 <= self.p2_pos[1] < GRID_WIDTH):
            p2_hit = True
        elif self.grid[self.p2_pos] != 0:
            p2_hit = True

        # Head-on collision
        if self.p1_pos == self.p2_pos:
            p1_hit = True
            p2_hit = True
            reason = "Head-on"

        # Update Grid (Draw trails) if no immediate crash at the new spot
        # Note: In standard Tron, you die if you hit a trail OR wall. 
        # If both die, it's a draw.
        
        if p1_hit and p2_hit:
            self.game_over = True
            self.winner = "Draw"
            self.message = f"Draw! ({reason})"
        elif p1_hit:
            self.game_over = True
            self.winner = "AI"
            self.message = "You Crashed! AI Wins."
        elif p2_hit:
            self.game_over = True
            self.winner = "Human"
            self.message = "AI Crashed! You Win!"
        else:
            # Safe move: update trails
            self.grid[self.p1_pos] = 1 # Player trail
            self.grid[self.p2_pos] = 2 # Opponent trail
            
            # Update internal env state for consistency if needed later
            self.env.grid = self.grid
            self.env.p1_pos = self.p1_pos
            self.env.p2_pos = self.p2_pos

    def draw(self):
        self.screen.fill(BLACK)

        # Draw Grid/Walls (Optional: draw grid lines)
        # Draw Trails
        # 1 = Player Trail (Blue), 2 = Opponent Trail (Red)
        # We iterate the grid numpy array
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell_val = self.grid[r, c]
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if cell_val == 1:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif cell_val == 2:
                    pygame.draw.rect(self.screen, RED, rect)
                elif cell_val == -1: # Wall if used
                    pygame.draw.rect(self.screen, GRAY, rect)

        # Draw Heads (Brighter)
        p1_rect = pygame.Rect(self.p1_pos[1] * CELL_SIZE, self.p1_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (100, 200, 255), p1_rect) # Light Blue
        
        p2_rect = pygame.Rect(self.p2_pos[1] * CELL_SIZE, self.p2_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 100, 100), p2_rect) # Light Red

        # Draw UI Text
        if self.game_over:
            text_surf = self.big_font.render(self.message, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(text_surf, text_rect)
            
            sub_surf = self.font.render("Press SPACE to Restart or ESC to Quit", True, (200, 200, 200))
            sub_rect = sub_surf.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
            self.screen.blit(sub_surf, sub_rect)
        else:
            # Status
            score_text = self.font.render(f"Status: Playing", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            
            # Instructions
            inst_text = self.font.render("WASD to Move", True, (150, 150, 150))
            self.screen.blit(inst_text, (10, WINDOW_HEIGHT - 40))

        pygame.display.flip()

    def run(self):
        running = True
        ai_moved = False
        player_action = None

        while running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE and self.game_over:
                        self.reset_game()
                        ai_moved = False
                        player_action = None

            if self.game_over:
                self.draw()
                self.clock.tick(FPS)
                continue

            # Game Logic Flow:
            # 1. AI Moves first (or simultaneously, but we calculate AI first)
            if not ai_moved:
                # Calculate AI move
                # We pass current state. Note: AI sees the board BEFORE player moves this turn?
                # Standard simultaneous: Both decide based on t-1 state.
                # Let's do simultaneous decision making based on current state.
                ai_action = self.get_opponent_move()
                
                # Now wait for player input to execute the frame
                # We don't update logic yet, just store AI choice
                current_ai_action = ai_action
                ai_moved = True
            
            # 2. Get Player Input
            player_action = self.handle_input()
            
            # If player hasn't pressed a valid key, don't step yet (wait for input)
            # But to make it real-time, we might want to force a move or default to forward?
            # Let's require a key press to step, making it turn-based per frame
            if player_action is not None and ai_moved:
                self.update_logic(player_action, current_ai_action)
                ai_moved = False # Reset for next turn
                player_action = None

            self.draw()
            self.clock.tick(FPS) # Cap FPS to control game speed

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Usage: python play_human.py
    print("Starting Human vs Minimax Duel...")
    print("Controls: W (Up), A (Left), S (Down), D (Right)")
    print("Press SPACE to restart after game over.")
    
    # Set difficulty: 'easy', 'medium', 'hard'
    game = HumanDuel(difficulty='hard')
    game.run()
