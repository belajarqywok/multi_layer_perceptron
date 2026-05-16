"""
generate_game_data.py
Generates synthetic training data for the Plane Shooter enemy AI model.

Features (inputs):
  rel_x        - (enemy_x - player_x) / screen_width, range [-1, 1]
  rel_y        - how close enemy is to player vertically, range [0, 1]
  bullet_near  - 1 if a player bullet is threatening the enemy, else 0
  bullet_rel_x - (bullet_x - enemy_x) / screen_width if threatening, else 0
  difficulty   - 0.0 (easy), 0.5 (medium), 1.0 (hard)

Label (output):
  aggression   - how aggressively the enemy should behave, range [0, 1]
"""

import csv, random, math, os

random.seed(42)

SCREEN_W = 800
SCREEN_H = 600
SAMPLES = 1000  # per difficulty level

def generate_sample(difficulty_val, diff_name):
    # Random enemy and player positions
    enemy_x  = random.uniform(0, SCREEN_W)
    enemy_y  = random.uniform(40, 280)
    player_x = random.uniform(80, SCREEN_W - 80)
    player_y = SCREEN_H - 80

    rel_x    = (enemy_x - player_x) / SCREEN_W           # [-1, 1]
    rel_y    = 1.0 - abs(enemy_y - player_y) / SCREEN_H  # [0, 1], higher = closer

    # Simulate whether a player bullet is nearby
    has_bullet  = random.random() < 0.35
    bullet_near = 0
    bullet_rel_x = 0.0
    if has_bullet:
        bx = random.uniform(0, SCREEN_W)
        by = random.uniform(enemy_y - 90, enemy_y + 90)
        if math.hypot(bx - enemy_x, by - enemy_y) < 80:
            bullet_near  = 1
            bullet_rel_x = (bx - enemy_x) / SCREEN_W

    # Base aggression tied to difficulty, modulated by game state
    base      = difficulty_val
    proximity = rel_y * 0.15        # closer enemy = slightly more aggressive
    threat    = bullet_near * 0.05  # under fire = slightly more urgent
    noise     = random.gauss(0, 0.04)

    aggression = max(0.0, min(1.0, base + proximity + threat + noise))

    # Clamp so modes don't fully overlap
    if diff_name == 'easy':
        aggression = min(aggression, 0.44)
    elif diff_name == 'hard':
        aggression = max(aggression, 0.56)

    return [
        round(rel_x,        4),
        round(rel_y,        4),
        bullet_near,
        round(bullet_rel_x, 4),
        round(difficulty_val, 1),
        round(aggression,   4),
    ]

os.makedirs('data', exist_ok=True)

rows = []
for name, val in [('easy', 0.0), ('medium', 0.5), ('hard', 1.0)]:
    for _ in range(SAMPLES):
        rows.append(generate_sample(val, name))

random.shuffle(rows)

with open('data/game_ai.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['rel_x', 'rel_y', 'bullet_near', 'bullet_rel_x', 'difficulty', 'aggression'])
    writer.writerows(rows)

print(f"Generated {len(rows)} samples -> data/game_ai.csv")
