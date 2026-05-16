import csv
import random
import os

os.makedirs('data', exist_ok=True)

# Generate Classification Data
# Features: f1, f2, f3
# Label: class (0 or 1)
with open('data/classification.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['f1', 'f2', 'f3', 'class'])
    for _ in range(100):
        f1 = random.uniform(0, 10)
        f2 = random.uniform(0, 10)
        f3 = random.uniform(0, 10)
        # simple rule: if f1+f2 > 10, class=1 else 0
        c = 1 if (f1 + f2) > 10 else 0
        writer.writerow([f1, f2, f3, c])

# Generate Regression Data
# Features: a, b, c, d, e
# Labels: f, g
with open('data/regression.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    for _ in range(100):
        a = random.uniform(0, 100)
        b = random.uniform(0, 100)
        c = random.uniform(0, 100)
        d = random.uniform(0, 100)
        e = random.uniform(0, 100)
        # f = a+b, g = c*d
        f_val = a + b + random.uniform(-5, 5)
        g_val = c * d + random.uniform(-10, 10)
        writer.writerow([a, b, c, d, e, f_val, g_val])

print("CSV datasets generated in 'data/' directory.")
