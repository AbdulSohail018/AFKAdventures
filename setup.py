from setuptools import setup, Command
import os
import gdown

class ExtractCommand(Command):
    description = 'Extract the data zip file'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Google Drive file IDs
        file_ids = {
            "user_preference_file_id" : "1skEb0wALi3h_3rdW3NaFaNzd-_-S71Ht",
            "game_preference_file_id" : "1a_2_LNMeYLpmjTgDhWC8jHYLn4iZVuRe"
        }

        # Corresponding local file paths
        local_paths = {
            "user_preference_file" : "./data/user_preferences.csv",
            "user_preference_file" : "./data/game_preference.csv"
        }

        os.makedirs('./data', exist_ok=True)

        # Download each file from Google Drive
        for key, file_id in file_ids.items():
            file_url = f'https://drive.google.com/uc?id={file_id}'
            local_path = local_paths[key]
            gdown.download(file_url, local_path, quiet=False)

# setup(
#     name='steam-game-recommendation',
#     version='1.0',
#     packages=['source', 'source.utils'],
#     url='https://github.com/poojan243/AFKAdventures.git',
#     license='free',
#     author='Pooojan Gagrani',
#     author_email='gagranipoojan@yahoo.com',
#     description='AFKAdventures: Steam Game Recommendation System',
#     install_requires=[
#         'requests==2.20.0',
#         'gdown'
#     ],
#     cmdclass={
#         'extract': ExtractCommand,
#     }
# )
