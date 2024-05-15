import os
import subprocess
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.metrics import roc_curve, auc
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

def create_interaction_matrix(data, user_id_to_index, game_id_to_index):
    print("Creating interaction matrix...")
    user_indices = data['user_id'].map(user_id_to_index)
    item_indices = data['game_id'].map(game_id_to_index)
    playtime_weights = data['log_playtime'].astype(np.float32)
    return coo_matrix((playtime_weights, (user_indices, item_indices)))

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
                return None, None, None, None, None, None, None, None
            else:
                print("Data extraction completed successfully. Starting modeling...\n")
        else:
            return None, None, None, None, None, None, None, None

    df_user = load_data(file_path)
    df_user = standardize_data(df_user)

    print("Creating mappings for user IDs and game IDs...")
    user_id_to_index = pd.Series(df_user['user_id'].astype("category").cat.codes.values, index=df_user['user_id']).to_dict()
    game_id_to_index = pd.Series(df_user['game_id'].astype("category").cat.codes.values, index=df_user['game_id']).to_dict()

    # Create reverse mappings
    index_to_user_id = {v: k for k, v in user_id_to_index.items()}
    index_to_game_id = {v: k for k, v in game_id_to_index.items()}

    # Create a mapping from game ID to game name
    game_id_to_name = pd.Series(df_user['game_name'].values, index=df_user['game_id']).to_dict()

    print("Splitting data into train, validation, and test sets...")
    train_val_data, test_data = train_test_split(df_user, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

    train_interactions = create_interaction_matrix(train_data, user_id_to_index, game_id_to_index).tocsr()
    val_interactions = create_interaction_matrix(val_data, user_id_to_index, game_id_to_index).tocsr()
    test_interactions = create_interaction_matrix(test_data, user_id_to_index, game_id_to_index).tocsr()

    print("Data preprocessing and splitting completed.\n")
    return train_interactions, val_interactions, test_interactions, user_id_to_index, game_id_to_index, df_user, index_to_user_id, index_to_game_id, game_id_to_name

def train_bpr_model(train_interactions, factors=50, learning_rate=0.01, regularization=0.1, iterations=50):
    print("Training BPR model...")
    model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate, regularization=regularization, iterations=iterations)
    model.fit(train_interactions)
    print("BPR model training completed.\n")
    return model

def calculate_precision_recall_at_k(model, test_interactions, train_interactions, k):
    precision_at_k = 0
    recall_at_k = 0
    num_users = test_interactions.shape[0]

    print(f"Calculating Precision@{k} and Recall@{k}...")
    for user_id in range(num_users):
        training_items = train_interactions.getrow(user_id).indices
        test_items = test_interactions.getrow(user_id).indices

        if len(test_items) == 0:
            continue

        scores = model.user_factors[user_id] @ model.item_factors.T
        recommended_items = np.argsort(scores)[::-1]
        recommended_items = [item for item in recommended_items if item not in training_items][:k]

        num_relevant_items = len(set(test_items) & set(recommended_items))
        precision_at_k += num_relevant_items / float(k)
        recall_at_k += num_relevant_items / float(len(test_items))

    precision_at_k /= num_users
    recall_at_k /= num_users

    print(f"Precision@{k}: {precision_at_k}, Recall@{k}: {recall_at_k}")
    return precision_at_k, recall_at_k

def plot_precision_recall_vs_k(precision, recall, k_range):
    print("Plotting Precision and Recall vs Number of Recommendations...")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, precision, label='Precision@k', marker='o')
    plt.plot(k_range, recall, label='Recall@k', marker='o')
    plt.title('Precision and Recall vs Number of Recommendations')
    plt.xlabel('Number of Top K Recommendations')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Plotting Precision and Recall completed.\n")

def calculate_auc(model, test_interactions, train_interactions):
    print("Calculating AUC...")
    num_users = test_interactions.shape[0]
    actuals = []
    predictions = []

    for user_id in range(num_users):
        test_items = test_interactions.getrow(user_id).indices
        training_items = train_interactions.getrow(user_id).indices

        if len(test_items) == 0:
            continue

        scores = -model.user_factors[user_id] @ model.item_factors.T
        actual = np.zeros(scores.shape[0])
        actual[test_items] = 1
        actuals.extend(actual)
        predictions.extend(scores)

    fpr, tpr, thresholds = roc_curve(actuals, predictions)
    roc_auc = auc(fpr, tpr)

    print(f"AUC: {roc_auc}")
    return fpr, tpr, roc_auc

def plot_auc(fpr, tpr, roc_auc):
    print("Plotting AUC...")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print("Plotting AUC completed.\n")

# Function to recommend items for a given user
def recommend_items(model, original_user_id, user_id_mapping, item_index_to_id, game_id_to_name, train_interactions, N=5):
    if original_user_id not in user_id_mapping:
        return f"No data available for user ID {original_user_id}"

    user_index = user_id_mapping[original_user_id]
    recommended = model.recommend(user_index, train_interactions[user_index], N=N)

    best_games = [game_id_to_name.get(item_index_to_id[idx], "Game name not found") for idx in recommended[0]]

    return best_games

if __name__ == "__main__":
    # Preprocess data and split into train, validation, and test sets
    print("Starting data preprocessing...")
    train_interactions, val_interactions, test_interactions, user_id_to_index, game_id_to_index, df_user, index_to_user_id, index_to_game_id, game_id_to_name = preprocess_and_split_data('./data/user_preferences.csv')

    # Check if data preprocessing was successful
    if train_interactions is None:
        print("Data preprocessing failed. Exiting.")
        exit()

    # Train the BPR model
    model = train_bpr_model(train_interactions)

    # Calculate Precision and Recall for different K values
    k_values = list(range(1, 11))
    precisions = []
    recalls = []

    for k in k_values:
        prec, rec = calculate_precision_recall_at_k(model, test_interactions, train_interactions, k=k)
        precisions.append(prec)
        recalls.append(rec)

    plot_precision_recall_vs_k(precisions, recalls, k_values)

    # Calculate and plot AUC
    fpr, tpr, roc_auc = calculate_auc(model, test_interactions, train_interactions)
    plot_auc(fpr, tpr, roc_auc)

    # Select random user IDs for testing
    random_user_ids = df_user['user_id'].sample(n=5, random_state=42).tolist()
    print("Randomly selected user IDs:", random_user_ids)

    # Loop through each random user ID and print their recommended games
    for user_id in random_user_ids:
        recommended_games = recommend_items(model, user_id, user_id_to_index, index_to_game_id, game_id_to_name, train_interactions, N=5)
        print(f"Recommended Games for User ID {user_id}: {recommended_games}\n")
