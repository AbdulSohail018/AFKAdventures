import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    """Load data from a CSV file."""
    print("Loading data...")
    return pd.read_csv(file_path)

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
                return None, None, None, None, None, None
            else:
                print("Data extraction completed successfully. Starting modeling...\n")
        else:
            return None, None, None, None, None, None

    df = load_data(file_path)
    columns_to_drop = ['developer', 'publisher', 'score_rank', 'positive', 'negative', 'owners', 'currency', 'positive_ratio']
    df= df.drop(columns=columns_to_drop)

    df.rename(columns = {'final' : 'price'}, inplace = True)

    # Ensure 'genre' is either list or string
    if any(df['genre'].apply(lambda x: isinstance(x, list))):
        df['genre'] = df['genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    df['categories 2'] = df['categories 2'].str.strip()

    # Filter 'categories 2' to include only 'Single-player' and 'Multi-player'
    df['categories 2'] = df['categories 2'].apply(lambda x: x if x in ['Single-player', 'Multi-player'] else None)

    # OneHotEncoder initialization with handle_unknown set to 'ignore'
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Encoding 'categories 1'
    categories_1_valid = df[['categories 1']].dropna()
    categories_encoded_1 = one_hot_encoder.fit_transform(categories_1_valid)
    cat1_feature_names = ['cat_1_' + feature.split('_')[-1] for feature in one_hot_encoder.get_feature_names_out()]
    categories_encoded_df_1 = pd.DataFrame(categories_encoded_1, columns=cat1_feature_names, index=categories_1_valid.index)

    # Encoding 'categories 2'
    categories_2_valid = df[['categories 2']].fillna('Ignore')
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[['Single-player', 'Multi-player']])
    categories_encoded_2 = one_hot_encoder.fit_transform(categories_2_valid)
    cat2_feature_names = ['cat_2_' + feature.split('_')[-1] for feature in one_hot_encoder.categories_[0]]
    categories_encoded_df_2 = pd.DataFrame(categories_encoded_2, columns=cat2_feature_names, index=categories_2_valid.index)

    # Joining the new DataFrames with the original data
    data_encoded = df.join(categories_encoded_df_1).join(categories_encoded_df_2)

    # Function to split the genres into lists of individual genres
    def split_genres(genre):
        if pd.isna(genre):
            return []
        return [g.strip() for g in genre.split(',')]

    # Apply the function to the genre column
    data_encoded['genre'] = data_encoded['genre'].apply(split_genres)

    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit and transform the genre data
    genres_encoded = mlb.fit_transform(data_encoded['genre'])

    # Create a DataFrame from the encoded genres
    genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

    # Merge the encoded DataFrame with the original data
    data_encoded = data_encoded.join(genres_encoded_df)

    # now removing unnecessary columns
    columns_to_drop = ['categories 2','categories 1', 'genre']
    data_encoded= data_encoded.drop(columns=columns_to_drop)

    # Drop rows with any null values directly in the original DataFrame
    df.dropna(inplace=True)

    # Summing the columns for 'Single-player'
    data_encoded['single player'] = data_encoded['cat_1_Single-player'] + data_encoded['cat_2_Single-player']
    data_encoded['single player'] = data_encoded['single player'].clip(upper=1)  # Ensure the maximum value is 1

    # Summing the columns for 'Multi-player'
    data_encoded['multi player'] = data_encoded['cat_1_Multi-player'] + data_encoded['cat_2_Multi-player']
    data_encoded['multi player'] = data_encoded['multi player'].clip(upper=1)  # Ensure the maximum value is 1

    # Create the 'both' column based on the condition
    data_encoded['both'] = ((data_encoded['single player'] == 1) & (data_encoded['multi player'] == 1)).astype(int)

    # Set 'single player' and 'multi player' to 0 where 'both' is 1
    data_encoded.loc[data_encoded['both'] == 1, ['single player', 'multi player']] = 0

    # now removing unnecessary columns
    columns_to_drop = ['cat_1_Multi-player','cat_1_Single-player', 'cat_2_Single-player', 'cat_2_Multi-player']
    data_encoded= data_encoded.drop(columns=columns_to_drop)

    # Specify the columns you want to plot
    columns_to_transform = ['average_forever', 'average_2weeks', 'median_forever',
                            'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average',
                            'MutualPlayerCount']

    # Determine the layout of the subplots
    num_columns = 3  # Define the number of columns in your subplot grid
    num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns  # Calculate the number of rows needed

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through the list of columns and create a plot for each one
    for i, col in enumerate(columns_to_transform):
        sns.histplot(data_encoded[col], bins=30, kde=True, color='blue', alpha=0.6, ax=axes[i])
        axes[i].set_title(f'Raw data Distribution with KDE - {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # If there are any leftover axes, turn them off
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Show the plot
    plt.show()

    def apply_log_transformation(data, columns):
        transformed_data = data.copy()  # Create a copy of the data to keep the original dataframe untouched
        constant = 1  # You can adjust the constant as necessary
        for col in columns:
            transformed_data[col] = np.log(transformed_data[col] + constant)
        return transformed_data

    # List of columns to transform
    columns_to_transform = ['average_forever', 'average_2weeks', 'median_forever',
           'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average',
           'MutualPlayerCount']  # Add your specific column names here

    # Apply the log transformation function to a copy for plotting
    df_transformed = apply_log_transformation(data_encoded, columns_to_transform)

    # Determine the layout of the subplots
    num_columns = 3  # Define the number of columns in your subplot grid
    num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns  # Calculate the number of rows needed

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through the list of columns and create a plot for each one
    for i, col in enumerate(columns_to_transform):
        sns.histplot(df_transformed[col], bins=30, kde=True, color='blue', alpha=0.6, ax=axes[i])
        axes[i].set_title(f'Log Transformed Distribution with KDE - {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # If there are any leftover axes, turn them off
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Show the plot
    plt.show()

    # Your specified columns to transform
    columns_to_transform = ['average_forever', 'average_2weeks', 'median_forever',
                            'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average',
                            'MutualPlayerCount']

    # Prepare a subplot layout
    num_columns = 3
    num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    # Explore a range of lambda values for each column
    for index, col in enumerate(columns_to_transform):
        lambdas = np.linspace(-2, 2, num=100)  # Range of lambda values to explore
        best_pvalue = 0
        best_lambda = None
        best_transformed_data = None

        # Ensure all data are positive
        data_shifted = data_encoded[col] + 1 - np.min(data_encoded[col])

        # Try different lambda values
        for l in lambdas:
            transformed_data = stats.boxcox(data_shifted, lmbda=l)  # Note the change here, only one output
            shapiro_test = stats.shapiro(transformed_data)
            if shapiro_test.pvalue > best_pvalue:
                best_pvalue = shapiro_test.pvalue
                best_lambda = l
                best_transformed_data = transformed_data

        zeros_count = (transformed_data == 0).sum()
        print(f"Number of zero values after transforming {col}: {zeros_count}")

        # Plot the best transformed data
        sns.histplot(best_transformed_data, bins=30, kde=True, color='blue', alpha=0.6, ax=axes[index])
        axes[index].set_title(f'box-cox - {col} - λ={best_lambda:.2f}, p-value={best_pvalue:.3f}')
        axes[index].set_xlabel('Transformed Value')
        axes[index].set_ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # If there are any leftover axes, turn them off
    for i in range(index + 1, len(axes)):
        axes[i].axis('off')

    # Show the plot
    plt.show()

    df_model = data_encoded.copy()

    # Normalize column names in the DataFrame
    df_model.columns = [col.lower().replace(' ', '_') for col in df_model.columns]

    # Split the data into train, validation, and test sets
    train_data, temp_data = train_test_split(df_model, test_size=0.2, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Apply quantile transformation
    def apply_quantile_transformation(train, validation, test):
        quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        feature_cols = ['average_forever', 'average_2weeks', 'median_forever',
                        'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average',
                        'mutualplayercount']  # Replace with actual feature column names

        train_transformed = train.copy()
        validation_transformed = validation.copy()
        test_transformed = test.copy()

        # Fit the transformer on the training data
        quantile_transformer.fit(train[feature_cols])

        # Apply the transformation to the training data
        train_transformed_cols = quantile_transformer.transform(train[feature_cols])
        validation_transformed_cols = quantile_transformer.transform(validation[feature_cols])
        test_transformed_cols = quantile_transformer.transform(test[feature_cols])

        # Add suffix to new columns and assign transformed data
        for i, col in enumerate(feature_cols):
            train_transformed[col + '_qt_normal'] = train_transformed_cols[:, i]
            validation_transformed[col + '_qt_normal'] = validation_transformed_cols[:, i]
            test_transformed[col + '_qt_normal'] = test_transformed_cols[:, i]

        return train_transformed, validation_transformed, test_transformed

    train_transformed, validation_transformed, test_transformed = apply_quantile_transformation(train_data, validation_data, test_data)

    df_model_transformed = pd.concat([train_transformed, validation_transformed, test_transformed], ignore_index=True)

    feature_cols = [col for col in train_transformed.columns if col.endswith('_qt_normal')]

    train_transformed_combined = df_model_transformed.iloc[:train_transformed.shape[0]]
    validation_transformed_combined = df_model_transformed.iloc[train_transformed.shape[0]:train_transformed.shape[0] + validation_transformed.shape[0]]
    test_transformed_combined = df_model_transformed.iloc[train_transformed.shape[0] + validation_transformed.shape[0]:]

    return df_model_transformed, train_transformed_combined, validation_transformed_combined, test_transformed_combined, feature_cols

def prepare_data_for_dbscan_clustering(df, feature_cols, validation_data, test_data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # Define ranges to test
    eps_values = np.arange(0.65, 0.95, 0.05)
    min_samples_values = range(3, 10)

    # Find the best parameters
    best_params = find_best_dbscan_params(X_scaled, eps_values, min_samples_values)

    # Apply DBSCAN clustering using the best parameters found
    eps_best, min_samples_best = best_params
    dbscan_best = DBSCAN(eps=eps_best, min_samples=min_samples_best)
    labels_best = dbscan_best.fit_predict(X_scaled)

    df['cluster'] = labels_best

    # Calculate silhouette scores for validation and test data
    X_val_scaled = scaler.transform(validation_data[feature_cols])
    X_test_scaled = scaler.transform(test_data[feature_cols])

    val_labels = dbscan_best.fit_predict(X_val_scaled)
    test_labels = dbscan_best.fit_predict(X_test_scaled)

    if len(set(val_labels)) > 1:
        val_score = silhouette_score(X_val_scaled, val_labels)
        print(f"Silhouette Score on Validation Data: {val_score}")
    else:
        print(f"No valid clusters formed on Validation Data.")

    if len(set(test_labels)) > 1:
        test_score = silhouette_score(X_test_scaled, test_labels)
        print(f"Silhouette Score on Test Data: {test_score}")
    else:
        print(f"No valid clusters formed on Test Data.")

    return df

def find_best_dbscan_params(X_scaled, eps_values, min_samples_values):
    best_score = -1
    best_params = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                print(f"Testing eps={eps}, min_samples={min_samples}, Silhouette Score={score}")
            else:
                print(f"Testing eps={eps}, min_samples={min_samples}, No valid clusters formed.")

    print(f"Best parameters found: eps={best_params[0]}, min_samples={best_params[1]} with Silhouette Score={best_score}")
    return best_params

def calculate_similarity(df, features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[features])
    return cosine_similarity(features_scaled)

def recommend_games_interactive(df, similarity_scores, num_recommendations=5):
    available_categories = ['single player', 'multi player', 'both']
    available_genres = ['Action', 'Adventure', 'Animation & Modeling', 'Casual', 'Design & Illustration',
                        'Early Access', 'Education', 'Free to Play', 'Game Development', 'Gore', 'Indie',
                        'Massively Multiplayer', 'Movie', 'Nudity', 'RPG', 'Racing', 'Sexual Content',
                        'Simulation', 'Sports', 'Strategy', 'Utilities', 'Violent', 'Web Publishing']

    print("Available Categories:")
    for category in available_categories:
        print(f"- {category}")
    user_category = input("\nEnter the category you would like to play (e.g., 'single player'): ").lower().replace(' ', '_')

    print("\nAvailable Genres:")
    for genre in available_genres:
        print(f"- {genre}")
    user_genre = input("\nEnter the genre you like (e.g., 'action'): ").lower().replace(' ', '_')

    if user_category not in df.columns or user_genre not in df.columns:
        print("Invalid category or genre provided. Please check your input.")
        return None

    filtered_df = df[(df[user_genre] == 1) & (df[user_category] == 1)]
    if filtered_df.empty:
        print("No games found matching your criteria.")
        return None

    cluster_mode = filtered_df['cluster'].mode()[0]  # Most common cluster in filtered games
    cluster_filtered_df = filtered_df[filtered_df['cluster'] == cluster_mode]

    game_indices = cluster_filtered_df.index
    user_game_similarities = similarity_scores[game_indices][:, game_indices]
    similarity_sum = np.sum(user_game_similarities, axis=0)
    if np.max(similarity_sum) == 0:
        print("Similarity scores are all zero.")
        return None
    similarity_sum_normalized = similarity_sum / np.max(similarity_sum)  # Normalize the scores

    most_similar_indices = np.argsort(-similarity_sum_normalized)[:num_recommendations]
    recommendations = cluster_filtered_df.iloc[most_similar_indices]

    recommendations = recommendations[['name', 'ratings', 'price', 'cluster']]
    recommendations['scaled_similarity_score'] = similarity_sum_normalized[most_similar_indices]
    recommendations[user_genre] = 1
    recommendations[user_category] = 1

    formatted_recommendations = recommendations.copy()
    formatted_recommendations['ratings'] = formatted_recommendations['ratings'].apply(lambda x: f"{x:.2f}")
    formatted_recommendations['price'] = formatted_recommendations['price'].apply(lambda x: f"${x:.2f}")
    formatted_recommendations['scaled_similarity_score'] = formatted_recommendations['scaled_similarity_score'].apply(lambda x: f"{x:.4f}")

    return formatted_recommendations

if __name__ == "__main__":
    file_path = './data/game_preference.csv'
    df_model_transformed, train_transformed_combined, validation_transformed_combined, test_transformed_combined, feature_cols = preprocess_and_split_data(file_path)

    if df_model_transformed is None:
        print("Data preprocessing failed. Exiting.")
        exit()

    df_model_transformed = prepare_data_for_dbscan_clustering(df_model_transformed, feature_cols, validation_transformed_combined, test_transformed_combined)
    similarity_scores = calculate_similarity(df_model_transformed, ['average_forever_qt_normal', 'ccu_qt_normal', 'ratings_qt_normal'])
    print("Similarity scores calculated successfully.")

    num_clusters = len(set(df_model_transformed['cluster'])) - (1 if -1 in df_model_transformed['cluster'] else 0)
    print(f"Number of clusters formed (excluding noise): {num_clusters}")

    recommended_games = recommend_games_interactive(df_model_transformed, similarity_scores, num_recommendations=5)
    if recommended_games is not None:
        print("Based on your interest, games recommended by AFKAdventures are:")
        print(recommended_games.to_string(index=False))
