from setuptools import setup, Command
import os
import gdown
from tqdm import tqdm

class ExtractCommand(Command):
    description = 'Extract the data zip file'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Google Drive file IDs
        user_preference_file_id = "1skEb0wALi3h_3rdW3NaFaNzd-_-S71Ht"
        game_preference_file_id = "1a_2_LNMeYLpmjTgDhWC8jHYLn4iZVuRe"

        # File URLs
        user_preference_file_url = f'https://drive.google.com/uc?id={user_preference_file_id}'
        game_preference_file_url = f'https://drive.google.com/uc?id={game_preference_file_id}'

        # Local file paths
        my_path = os.path.abspath(os.path.dirname(__file__))
        user_preference_file_path = os.path.join(my_path, "./data/user_preferences.csv")
        game_preference_file_path = os.path.join(my_path, "./data/game_preference.csv")

        os.makedirs('./data', exist_ok=True)
        print("Created data directory if it did not exist.")

        # Download each file from Google Drive
        self.download_file(user_preference_file_url, user_preference_file_path, "User Preferences")
        self.download_file(game_preference_file_url, game_preference_file_path, "Game Preferences")

    def download_file(self, url, output, description):
        print(f"Starting download of {description} from {url} to {output}...")
        try:
            gdown.download(url, output, quiet=False)
            if os.path.exists(output):
                print(f"Successfully downloaded {description} to {output}.")
            else:
                print(f"Failed to download {description}.")
        except Exception as e:
            print(f"An error occurred while downloading {description}: {e}")

setup(
    name='steam-game-recommendation',
    version='1.0',
    packages=[],
    url='https://github.com/poojan243/AFKAdventures.git',
    license='free',
    author='Poojan Gagrani',
    author_email='gagranipoojan@yahoo.com',
    description='AFKAdventures: Steam Game Recommendation System',
    install_requires=[
        'requests==2.20.0',
        'gdown',
        'tqdm'
    ],
    cmdclass={
        'extract': ExtractCommand,
    }
)
