import pygame

class Process:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Constants
        WIDTH, HEIGHT = 1280, 720
        
        # Create the Pygame window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Draw Points and Lines")
        
        # Load the background image and resize it
        background_image = pygame.image.load("background.jpeg")  # Replace "background.jpg" with your image file
        background_image = pygame.transform.scale(background_image, (1280, 720))  # Resize to 1280x720
        
        # Constants for drawing points and lines
        POINT_COLOR = (0, 0, 0)
        POINT_RADIUS = 5
        LINE_COLOR = (255, 0, 0)
        LINE_WIDTH = 2
        
        # List to store points
        points = []
        self.lines = []

        # Main game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        # Create a point at the mouse click position
                        points.append(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:  # Press 'D' key to exit
                        running = False
                    if event.key == pygame.K_u:  # Press 'D' key to exit
                            self.lines = self.lines[:-1] 
                            points = points[:-2]
        
            # Draw the resized background image
            screen.blit(background_image, (0, 0))
        
            # Draw lines between every second pair of consecutive points
            for i in range(0, len(points) - 1, 2):
                pygame.draw.line(screen, LINE_COLOR, points[i], points[i + 1], LINE_WIDTH)
                
                line =  (points[i], points[i + 1])
                if line not in self.lines:
                    self.lines.append(line)
                    # print(line)
        
            # Draw points
            for point in points:
                pygame.draw.circle(screen, POINT_COLOR, point, POINT_RADIUS)
        
            # Update the screen
            pygame.display.flip()
        
        # Quit Pygame
        pygame.quit()
        #sys.exit()
        return
    
    def getLines(self):
        return self.lines


"""
def main():
    myclass = Process()
    print(myclass.getLines())
    return
    
if __name__ == "__main__":
    main()
"""