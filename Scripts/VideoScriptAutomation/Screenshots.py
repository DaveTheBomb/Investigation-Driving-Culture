import pyautogui
from PIL import ImageGrab
import time
import os

class ScreenshotCapturer:
    def __init__(self, num_screenshots, delay_between_screenshots, save_directory):
        self.num_screenshots = num_screenshots
        self.delay_between_screenshots = delay_between_screenshots
        self.save_directory = save_directory

    def create_save_directory(self):
        os.makedirs(self.save_directory, exist_ok=True)

    def capture_screenshots(self):
        self.create_save_directory()

        for i in range(self.num_screenshots):
            # Wait for the specified delay before taking each screenshot
            time.sleep(self.delay_between_screenshots)

            # Capture a screenshot of the entire screen
            screenshot = ImageGrab.grab()

            # Save the screenshot with a numbered filename in the specified directory
            filename = os.path.join(self.save_directory, f"screenshot_{i + 1}.png")
            screenshot.save(filename)

            print(f"Screenshot {i + 1} saved as {filename}")

if __name__ == "__main__":
    screen_setup_waiting = 10
    num_screenshots = 10
    delay_between_screenshots = 0.005  # 5 milliseconds
    save_directory = "C:/Users/1892513/Desktop/INGODWETRUST/data"

    screenshot_capturer = ScreenshotCapturer(num_screenshots, delay_between_screenshots, save_directory)

    # Wait for screen setup
    time.sleep(screen_setup_waiting)

    # Capture and save screenshots
    screenshot_capturer.capture_screenshots()