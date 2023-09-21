import pyautogui
from PIL import ImageGrab
import time
import os


screen_setup_waiting = 20
time.sleep(screen_setup_waiting)


# Number of screenshots to capture
num_screenshots = 100

# Delay between each screenshot (in seconds)
delay_between_screenshots = 0.005  # 5 milliseconds

# Specify the directory where you want to save the screenshots
save_directory = "C:/Users/1892513/Desktop/INGODWETRUST/data"

# Ensure the directory exists; create it if it doesn't
os.makedirs(save_directory, exist_ok=True)

for i in range(num_screenshots):
    # Wait for the specified delay before taking each screenshot
    time.sleep(delay_between_screenshots)

    # Capture a screenshot of the entire screen
    screenshot = ImageGrab.grab()

    # Save the screenshot with a numbered filename in the specified directory
    filename = os.path.join(save_directory, f"screenshot_{i + 1}.png")
    screenshot.save(filename)

    print(f"Screenshot {i + 1} saved as {filename}")

# Optionally, you can display the last screenshot captured
# screenshot.show()