import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    print("Loading data...")
    return pd.read_csv(file_path)

def standardize_data(df):
    """Standardize the data as specified."""
    print("Standardizing data...")
    df.rename(columns={
        'final': 'price (USD)',
        'SteamID': 'user_id',
        'appid': 'game_id',
        'name': 'game_name',
        'PlaytimeForever (minutes)': 'playtime (minutes)',
        'CountryName': 'location'
    }, inplace=True)

    df.drop(['CountryCode', 'currency'], axis=1, inplace=True)
    df['location'].fillna('Unknown', inplace=True)
    df.dropna(subset=['game_name', 'genre'], inplace=True)

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
    df['log_playtime'] = np.log1p(df['playtime (minutes)'])

    return df

def preprocess_and_split_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print("\033[1mData file not found. Please run the command 'python setup.py extract' to extract data first.\033[0m")
        
        # Ask the user if they want to run the extraction command
        user_input = input("Would you like to run 'python setup.py extract' now? (y/n): ")
        if user_input.lower() == 'y':
            print("Running 'python setup.py extract'...")
            subprocess.run(["python", "setup.py", "extract"])
            # Check again if the file exists after extraction
            if not os.path.exists(file_path):
                print("\033[1mData file still not found after extraction attempt. Please check your setup.\033[0m")
                return None, None, None, None
            else:
                print("Data extraction completed successfully. Starting modeling...\n")
        else:
            return None, None, None, None

    df_user = load_data(file_path)
    df_user = standardize_data(df_user)

    print("Splitting data into train, validation, and test sets...")
    train_val_data, test_data = train_test_split(df_user, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

    print("Data preprocessing and splitting completed.\n")
    return train_data, val_data, test_data, df_user

def transform_data(train_data, val_data, test_data):
    # Define the ColumnTransformer
    column_transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['genre', 'developer', 'publisher', 'location']),
        ('num', MinMaxScaler(), ['price (USD)', 'ratings', 'log_playtime'])
    ], remainder='drop')

    # Apply transformations to train, validation, and test sets
    print("Transforming data...")
    train_features = column_transformer.fit_transform(train_data)
    val_features = column_transformer.transform(val_data)
    test_features = column_transformer.transform(test_data)

    train_feature_df = pd.DataFrame(train_features.toarray(), columns=column_transformer.get_feature_names_out())
    train_feature_df['user_id'] = train_data['user_id']
    train_feature_df['game_id'] = train_data['game_id']
    
    val_feature_df = pd.DataFrame(val_features.toarray(), columns=column_transformer.get_feature_names_out())
    val_feature_df['user_id'] = val_data['user_id']
    val_feature_df['game_id'] = val_data['game_id']

    test_feature_df = pd.DataFrame(test_features.toarray(), columns=column_transformer.get_feature_names_out())
    test_feature_df['user_id'] = test_data['user_id']
    test_feature_df['game_id'] = test_data['game_id']

    print("Data transformation completed.\n")
    return train_feature_df, val_feature_df, test_feature_df

def create_profiles(train_feature_df, test_feature_df, train_data, test_data):
    # Weight features by normalized log playtime for training data
    print("Creating user and item profiles...")
    train_weighted_features = train_feature_df.iloc[:, :-2].multiply(train_data['log_playtime'], axis=0)
    train_user_profiles = train_weighted_features.groupby(train_data['user_id']).mean()

    # Weight features by normalized log playtime for testing data
    test_weighted_features = test_feature_df.iloc[:, :-2].multiply(test_data['log_playtime'], axis=0)
    test_user_profiles = test_weighted_features.groupby(test_data['user_id']).mean()

    # Create item profiles from training data
    train_item_profiles = train_feature_df.groupby(train_data['game_id']).first().iloc[:, :-2]

    print("Profiles creation completed.\n")
    return train_user_profiles, test_user_profiles, train_item_profiles

def calculate_similarity(train_user_profiles, train_item_profiles):
    print("Calculating cosine similarity...")
    
    # Fill NaN values with 0s to handle missing data
    train_user_profiles = train_user_profiles.fillna(0)
    train_item_profiles = train_item_profiles.fillna(0)
    
    similarity_scores = cosine_similarity(train_user_profiles, train_item_profiles)
    print("Cosine similarity calculation completed.\n")
    return similarity_scores

def get_recommendations(user_id, user_profiles, item_profiles, similarity_scores, n_recommendations=5):
    if user_id not in user_profiles.index:
        return []

    user_idx = user_profiles.index.get_loc(user_id)
    similarity_scores_user = similarity_scores[user_idx]
    top_indices = np.argsort(similarity_scores_user)[-n_recommendations * 10:][::-1]
    unique_game_ids = item_profiles.iloc[top_indices].index.drop_duplicates().tolist()[:n_recommendations]
    return unique_game_ids

def get_multiple_recommendations(user_id, user_profiles, item_profiles, similarity_scores, n_recommendations=5):
    if user_id not in user_profiles.index:
        return []

    user_idx = user_profiles.index.get_loc(user_id)
    similarity_scores_user = similarity_scores[user_idx]
    top_indices = np.argsort(similarity_scores_user)[-n_recommendations * 10:][::-1]
    unique_game_ids = item_profiles.iloc[top_indices].index.drop_duplicates().tolist()[:n_recommendations]
    return unique_game_ids

def create_actual_preferences(df, user_id_col, item_id_col, preference_col, top_n=3):
    grouped = df.sort_values(by=[user_id_col, preference_col], ascending=[True, False]).groupby(user_id_col)
    return {user: list(group[item_id_col].head(top_n)) for user, group in grouped}

def precision_at_k(recommended_items, actual_items, k):
    recommended_at_k = recommended_items[:k]
    hits_k = len(set(recommended_at_k) & set(actual_items))
    return hits_k / float(k)

def recall_at_k(recommended_items, actual_items, k):
    if not actual_items:
        return 0
    recommended_at_k = recommended_items[:k]
    hits_k = len(set(recommended_at_k) & set(actual_items))
    return hits_k / float(len(actual_items))

def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_auc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr), fpr, tpr

if __name__ == "__main__":
    # Preprocess data and split into train, validation, and test sets
    print("Starting data preprocessing...")
    train_data, val_data, test_data, df_user = preprocess_and_split_data('./data/user_preferences.csv')

    # Check if data preprocessing was successful
    if train_data is None:
        print("Data preprocessing failed. Exiting.")
        exit()

    # Transform data
    train_feature_df, val_feature_df, test_feature_df = transform_data(train_data, val_data, test_data)

    # Create user and item profiles
    train_user_profiles, test_user_profiles, train_item_profiles = create_profiles(train_feature_df, test_feature_df, train_data, test_data)

    # Create game_id to game_name mapping
    game_id_to_name = pd.Series(df_user['game_name'].values, index=df_user['game_id']).to_dict()

    # Calculate cosine similarity
    similarity_scores = calculate_similarity(train_user_profiles, train_item_profiles)

    # Evaluation
    print("Evaluating model...")
    actual_preferences = create_actual_preferences(test_data, 'user_id', 'game_id', 'playtime (minutes)', top_n=3)
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    k_values = [1, 3, 5, 10, 15]

    for k in k_values:
        k_precisions = []
        k_recalls = []
        y_true = []
        y_scores = []
        for user_id in actual_preferences.keys():
            recommended_games = get_recommendations(user_id, train_user_profiles, train_item_profiles, similarity_scores, k)
            actual_games = actual_preferences[user_id]
            p_at_k = precision_at_k(recommended_games, actual_games, k)
            r_at_k = recall_at_k(recommended_games, actual_games, k)
            y_true.extend([1 if game in actual_games else 0 for game in recommended_games])
            y_scores.extend([similarity_scores[train_user_profiles.index.get_loc(user_id), train_item_profiles.index.get_loc(game)] if game in recommended_games else 0 for game in recommended_games])
            k_precisions.append(p_at_k)
            k_recalls.append(r_at_k)
        precisions.append(np.mean(k_precisions))
        recalls.append(np.mean(k_recalls))
        f1_scores.append(f1_at_k(np.mean(k_precisions), np.mean(k_recalls)))
        auc_score, fpr, tpr = calculate_auc(y_true, y_scores)
        aucs.append(auc_score)

    # Plotting Precision and Recall
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, precisions, label='Precision@K', marker='o')
    plt.plot(k_values, recalls, label='Recall@K', marker='o')
    plt.plot(k_values, f1_scores, label='F1 Score@K', marker='o')
    plt.xlabel('Number of Recommendations (K)')
    plt.ylabel('Metric Value')
    plt.title('Precision, Recall, and F1 Score @ K for Recommendation System')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting AUC
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, aucs, label='AUC@K', marker='o')
    plt.xlabel('Number of Recommendations (K)')
    plt.ylabel('AUC Value')
    plt.title('AUC @ K for Recommendation System')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Evaluation completed.\n")

    # Select random user IDs for testing recommendations
    random_user_ids = test_data['user_id'].sample(n=5, random_state=42).tolist()
    print("Randomly selected user IDs:", random_user_ids)

    # Loop through each random user ID and print their recommended games
    for user_id in random_user_ids:
        recommendations = get_multiple_recommendations(user_id, train_user_profiles, train_item_profiles, similarity_scores, n_recommendations=3)
        if recommendations:
            recommendation_names = [game_id_to_name[game_id] for game_id in recommendations]
            print(f"User ID {user_id} Recommendations: {recommendation_names}")
        else:
            print(f"User ID {user_id} not found in the profiles index.")
