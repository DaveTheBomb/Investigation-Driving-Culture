import pygame
import sys

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if (y1 == y2) and (y == y1) and (x >= min(x1, x2)) and (x <= max(x1, x2)):
            return True

        if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                inside = not inside

    return inside

# Initialize Pygame
pygame.init()

# Constants for screen size and colors
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Point in Polygon")

# Define the polygon and point
irregular_polygon = [
    (200, 100),
    (250, 50),
    (350, 50),
    (400, 100),
    (350, 150),
    (250, 150),
    (200, 100)
]

polygon = irregular_polygon
point = (200, 200)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # When the mouse is clicked, update the point to the mouse pointer position
            point = pygame.mouse.get_pos()

    # Clear the screen
    screen.fill(WHITE)

    # Draw the polygon
    pygame.draw.polygon(screen, BLACK, polygon)

    # Draw the point
    pygame.draw.circle(screen, RED, point, 5)  # Draw a small circle for the point

    # Check if the point is inside the polygon and display a message
    if point_in_polygon(point, polygon):
        font = pygame.font.Font(None, 36)
        text = font.render("Point is inside the polygon", True, BLACK)
        screen.blit(text, (50, 50))

    # Update the screen
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()