import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def standardize_data(df):
    """Standardize the data as specified."""
    # Renaming columns
    df.rename(columns={
        'final': 'price (USD)',
        'SteamID': 'user_id',
        'appid': 'game_id',
        'name': 'game_name',
        'PlaytimeForever (minutes)': 'playtime (minutes)',
        'CountryName': 'location'
    }, inplace=True)

    # Dropping specified columns
    df.drop([
        'CountryCode',
        'currency'
        # 'developer',
        # 'publisher'
    ], axis=1, inplace=True)

    # Filling missing 'location' values
    df['location'].fillna('Unknown', inplace=True)

    # Dropping rows where 'game_name' or 'genre' is missing
    df.dropna(subset=['game_name', 'genre'], inplace=True)

    # Standardizing country names
    country_mapping = {
        "Iran, Islamic Republic of": "Iran",
        "Korea, Republic of": "South Korea",
        "Taiwan, Province of China": "Taiwan",
        "Türkiye": "Turkey",
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Syrian Arab Republic": "Syria",
        "Czechia": "Czech Republic",
        "Micronesia, Federated States of": "Micronesia",
        "United Kingdom": "UK",
        "United States": "USA",
        "Virgin Islands, U.S.": "U.S. Virgin Islands",
    }

    df['location'] = df['location'].apply(lambda x: country_mapping.get(x, x))

    return df

def summarize_data(df):
    """Print summary statistics of the DataFrame."""
    print(df.info())
    print(df.describe())

def perform_eda(df):
    """Perform EDA on the DataFrame."""
    # Histogram of ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ratings'], kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Ratings')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate total playtime and average rating per game
    game_stats = df.groupby('game_name').agg({
        'playtime (minutes)': 'sum',  # Total playtime per game
        'ratings': 'mean'  # Average rating per game
    }).reset_index()

    # Sort and select top 10 for each category
    top_games_playtime = game_stats.sort_values(by='playtime (minutes)', ascending=False).head(10)
    top_games_ratings = game_stats.sort_values(by='ratings', ascending=False).head(10)

    # Plot Total Playtime for Top 10 Games
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_games_playtime, x='game_name', y='playtime (minutes)', palette='viridis')
    plt.title('Top 10 Games by Total Playtime')
    plt.xlabel('Game Name')
    plt.ylabel('Total Playtime (minutes)')
    plt.xticks(rotation=90)
    plt.show()

    # Ratings for top 10 games
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(data=top_games_ratings, x='game_name', y='ratings', palette='viridis')
    plt.title('Top 10 Games by Average Ratings')
    plt.xlabel('Game Name')
    plt.ylabel('Average Ratings')
    plt.xticks(rotation=90)

    # Adding the text on top of each bar
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 9),
                         textcoords='offset points')

    plt.show()

    # # Correlation matrix
    # correlation_matrix = df[['playtime (minutes)', 'price (USD)', 'ratings']].corr()

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn')
    # plt.title('Correlation Matrix')
    # plt.show()

    # # Histogram to show the distribution of playtime
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df['playtime (minutes)'], bins=30, kde=True)
    # plt.title('Distribution of Playtime')
    # plt.xlabel('Playtime in minutes')
    # plt.ylabel('Frequency')
    # plt.show()

def main():
    # File paths
    user_preference_file_path = './data/user_preferences.csv'
    game_preference_file_path = './data/game_preference.csv'

    # Check if files exist
    if not os.path.exists(user_preference_file_path) or not os.path.exists(game_preference_file_path):
        print("\033[1mData files not found. Please run the command 'python setup.py extract' to extract data first.\033[0m")
        return

    # Load data
    user_preferences_df = load_data(user_preference_file_path)
    game_preferences_df = load_data(game_preference_file_path)

    # Standardize data for user preferences only
    user_preferences_df = standardize_data(user_preferences_df)

    # Summarize data
    print("User Preferences Data Summary:")
    summarize_data(user_preferences_df)

    print("\nGame Preferences Data Summary:")
    summarize_data(game_preferences_df)

    # Perform EDA on user preferences only
    perform_eda(user_preferences_df)

if __name__ == "__main__":
    main()
