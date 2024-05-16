import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score


def load_data(filepath):
    if not os.path.exists(filepath):
        print("\033[1mData file not found. Please run the command 'python setup.py extract' to extract data first.\033[0m")
        user_input = input("Would you like to run 'python setup.py extract' now? (y/n): ")
        if user_input.lower() == 'y':
            print("Running 'python setup.py extract'...")
            subprocess.run(["python", "setup.py", "extract"])
            if not os.path.exists(filepath):
                print("\033[1mData file still not found after extraction attempt. Please check your setup.\033[0m")
                return None
            else:
                print("Data extraction completed successfully. Starting modeling...\n")
        else:
            return None
    df = pd.read_csv(filepath)
    columns_to_drop = ['developer', 'publisher', 'score_rank', 'positive', 'negative', 'owners', 'currency', 'positive_ratio']
    df = df.drop(columns=columns_to_drop)
    df.rename(columns={'final': 'price'}, inplace=True)
    return df


def preprocess_genre_column(df):
    if any(df['genre'].apply(lambda x: isinstance(x, list))):
        df['genre'] = df['genre'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    return df


def preprocess_categories(df):
    df['categories 2'] = df['categories 2'].str.strip()
    df['categories 2'] = df['categories 2'].apply(lambda x: x if x in ['Single-player', 'Multi-player'] else None)
    return df


def encode_categories(df):
    try:
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categories_1_valid = df[['categories 1']].dropna()
        categories_encoded_1 = one_hot_encoder.fit_transform(categories_1_valid)
        cat1_feature_names = ['cat_1_' + feature.split('_')[-1] for feature in one_hot_encoder.get_feature_names_out()]
        categories_encoded_df1_1 = pd.DataFrame(categories_encoded_1, columns=cat1_feature_names, index=categories_1_valid.index)

        categories_2_valid = df[['categories 2']].fillna('Ignore')
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[['Single-player', 'Multi-player']])
        categories_encoded_2 = one_hot_encoder.fit_transform(categories_2_valid)
        cat2_feature_names = ['cat_2_' + feature.split('_')[-1] for feature in one_hot_encoder.categories_[0]]
        categories_encoded_df1_2 = pd.DataFrame(categories_encoded_2, columns=cat2_feature_names, index=categories_2_valid.index)

        df_encoded = df.join(categories_encoded_df1_1).join(categories_encoded_df1_2)
        return df_encoded
    except KeyError as e:
        print(f"KeyError encountered during encoding categories: {e}")
        return None


def split_genres(genre):
    if pd.isna(genre):
        return []
    return [g.strip() for g in genre.split(',')]


def encode_genres(df):
    try:
        df['genre'] = df['genre'].apply(split_genres)
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(df['genre'])
        genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
        df_encoded = df.join(genres_encoded_df)
        return df_encoded
    except KeyError as e:
        print(f"KeyError encountered during encoding genres: {e}")
        return None


def drop_unnecessary_columns(df):
    try:
        columns_to_drop = ['categories 2', 'categories 1', 'genre']
        df = df.drop(columns=columns_to_drop)
        return df
    except KeyError as e:
        print(f"KeyError encountered during dropping columns: {e}")
        return None


def create_player_columns(df):
    try:
        print("Columns available before creating player columns:", df.columns)
        df['single player'] = df['cat_1_Single-player'] + df['cat_2_Single-player']
        df['single player'] = df['single player'].clip(upper=1)
        df['multi player'] = df['cat_1_Multi-player'] + df['cat_2_Multi-player']
        df['multi player'] = df['multi player'].clip(upper=1)
        df['both'] = ((df['single player'] == 1) & (df['multi player'] == 1)).astype(int)
        df.loc[df['both'] == 1, ['single player', 'multi player']] = 0
        return df
    except KeyError as e:
        print(f"KeyError encountered during creating player columns: {e}")
        return None


def plot_data_distribution(data, columns_to_transform):
    try:
        num_columns = 3
        num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(columns_to_transform):
            sns.histplot(data[col], bins=30, kde=True, color='blue', alpha=0.6, ax=axes[i])
            axes[i].set_title(f'Raw data Distribution with KDE - {col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        plt.tight_layout()
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.show()
    except KeyError as e:
        print(f"KeyError encountered during plotting data distribution: {e}")


def apply_log_transformation(data, columns):
    transformed_data = data.copy()
    constant = 1
    try:
        for col in columns:
            transformed_data[col] = np.log(transformed_data[col] + constant)
    except KeyError as e:
        print(f"KeyError encountered during log transformation: {e}")
    return transformed_data


def plot_transformed_distribution(data, columns_to_transform):
    try:
        num_columns = 3
        num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(columns_to_transform):
            sns.histplot(data[col], bins=30, kde=True, color='blue', alpha=0.6, ax=axes[i])
            axes[i].set_title(f'Log Transformed Distribution with KDE - {col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
        plt.tight_layout()
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.show()
    except KeyError as e:
        print(f"KeyError encountered during plotting transformed distribution: {e}")


def apply_boxcox_transformation(data, columns_to_transform):
    try:
        num_columns = 3
        num_rows = (len(columns_to_transform) + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()
        for index, col in enumerate(columns_to_transform):
            lambdas = np.linspace(-2, 2, num=100)
            best_pvalue = 0
            best_lambda = None
            best_transformed_data = None
            data_shifted = data[col] + 1 - np.min(data[col])
            for l in lambdas:
                transformed_data = stats.boxcox(data_shifted, lmbda=l)
                shapiro_test = stats.shapiro(transformed_data)
                if shapiro_test.pvalue > best_pvalue:
                    best_pvalue = shapiro_test.pvalue
                    best_lambda = l
                    best_transformed_data = transformed_data
            sns.histplot(best_transformed_data, bins=30, kde=True, color='blue', alpha=0.6, ax=axes[index])
            axes[index].set_title(f'box-cox - {col} - λ={best_lambda:.2f}, p-value={best_pvalue:.3f}')
            axes[index].set_xlabel('Transformed Value')
            axes[index].set_ylabel('Frequency')
        plt.tight_layout()
        for i in range(index + 1, len(axes)):
            axes[i].axis('off')
        plt.show()
    except KeyError as e:
        print(f"KeyError encountered during Box-Cox transformation: {e}")


def split_data(df):
    try:
        train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        return train_data, validation_data, test_data
    except KeyError as e:
        print(f"KeyError encountered during data splitting: {e}")
        return None, None, None


def apply_quantile_transformation(train, validation, test):
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    feature_cols = ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount']
    try:
        train_transformed = train.copy()
        validation_transformed = validation.copy()
        test_transformed = test.copy()
        quantile_transformer.fit(train[feature_cols])
        train_transformed_cols = quantile_transformer.transform(train[feature_cols])
        validation_transformed_cols = quantile_transformer.transform(validation[feature_cols])
        test_transformed_cols = quantile_transformer.transform(test[feature_cols])
        for i, col in enumerate(feature_cols):
            train_transformed[col + '_qt_normal'] = train_transformed_cols[:, i]
            validation_transformed[col + '_qt_normal'] = validation_transformed_cols[:, i]
            test_transformed[col + '_qt_normal'] = test_transformed_cols[:, i]
        return train_transformed, validation_transformed, test_transformed
    except KeyError as e:
        print(f"KeyError encountered during quantile transformation: {e}")
        return None, None, None


def plot_transformed_density_histograms(data, feature_cols):
    try:
        num_columns = 3
        num_rows = (len(feature_cols) + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            transformed_col = col + '_qt_normal'
            sns.histplot(data[transformed_col], kde=True, color='blue', alpha=0.6, ax=axes[i])
            axes[i].set_title(f'Quantile Transformed - {col}')
            axes[i].set_xlabel('Transformed Value')
            axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.show()
    except KeyError as e:
        print(f"KeyError encountered during plotting transformed density histograms: {e}")


def prepare_data_for_clustering(train, validation, test):
    try:
        feature_cols = train.filter(regex='qt_normal$').columns.tolist()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[feature_cols])
        X_validation_scaled = scaler.transform(validation[feature_cols])
        X_test_scaled = scaler.transform(test[feature_cols])
        Z = linkage(X_train_scaled, method='ward', metric='euclidean')
        max_clusters = 15
        silhouette_scores = []
        for n_clusters in range(2, max_clusters):
            cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            labels = cluster.fit_predict(X_train_scaled)
            score = silhouette_score(X_train_scaled, labels)
            silhouette_scores.append(score)
        best_cluster_size = np.argmax(silhouette_scores) + 2
        best_silhouette_score = max(silhouette_scores)
        best_cluster = AgglomerativeClustering(n_clusters=best_cluster_size, affinity='euclidean', linkage='ward')
        train['cluster'] = best_cluster.fit_predict(X_train_scaled)
        validation['cluster'] = best_cluster.fit_predict(X_validation_scaled)
        test['cluster'] = best_cluster.fit_predict(X_test_scaled)
        validation_silhouette_score = silhouette_score(X_validation_scaled, validation['cluster'])
        test_silhouette_score = silhouette_score(X_test_scaled, test['cluster'])
        print(f"Best number of clusters based on silhouette score: {best_cluster_size}")
        print(f"Silhouette score for best cluster size ({best_cluster_size}): {best_silhouette_score:.3f}")
        print(f"Silhouette score for validation data: {validation_silhouette_score:.3f}")
        print(f"Silhouette score for test data: {test_silhouette_score:.3f}")
        plt.figure(figsize=(12, 8))
        dendrogram(Z, color_threshold=Z[-(best_cluster_size-1), 2])
        plt.title(f'Dendrogram for Hierarchical Clustering with {best_cluster_size} Clusters')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        return train, validation, test
    except KeyError as e:
        print(f"KeyError encountered during data preparation for clustering: {e}")
        return None, None, None


def calculate_similarity(df, features):
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[features])
        return cosine_similarity(features_scaled)
    except KeyError as e:
        print(f"KeyError encountered during calculating similarity: {e}")
        return None


def recommend_games_interactive(df, similarity_scores, num_recommendations=5):
    available_categories = ['single player', 'multi player', 'both']
    available_genres = ['Action', 'Adventure', 'Animation & Modeling', 'Casual', 'Design & Illustration', 'Early Access', 'Education', 'Free to Play', 'Game Development', 'Gore', 'Indie', 'Massively Multiplayer', 'Movie', 'Nudity', 'RPG', 'Racing', 'Sexual Content', 'Simulation', 'Sports', 'Strategy', 'Utilities', 'Violent', 'Web Publishing']
    print("Available Categories:")
    for category in available_categories:
        print(f"- {category}")
    user_category = input("\nEnter the category you would like to play (e.g., 'single player'): ").lower().replace(' ', '_')
    print("\nAvailable Genres:")
    for genre in available_genres:
        print(f"- {genre}")
    user_genre = input("\nEnter the genre you like (e.g., 'Action'): ").lower().replace(' ', '_')
    if user_category not in df.columns or user_genre not in df.columns:
        print("Invalid category or genre provided. Please check your input.")
        return None
    filtered_df = df[(df[user_genre] == 1) & (df[user_category] == 1)]
    if filtered_df.empty:
        print("No games found matching your criteria.")
        return None
    cluster_mode = filtered_df['cluster'].mode()[0]
    cluster_filtered_df = filtered_df[filtered_df['cluster'] == cluster_mode]
    game_indices = cluster_filtered_df.index
    user_game_similarities = similarity_scores[game_indices][:, game_indices]
    similarity_sum = np.sum(user_game_similarities, axis=0)
    if np.max(similarity_sum) == 0:
        print("Similarity scores are all zero.")
        return None
    similarity_sum_normalized = similarity_sum / np.max(similarity_sum)
    most_similar_indices = np.argsort(-similarity_sum_normalized)[:num_recommendations]
    recommendations = cluster_filtered_df.iloc[most_similar_indices]
    recommendations = recommendations[['name', 'ratings', 'price', 'cluster']]
    recommendations['scaled_similarity_score'] = similarity_sum_normalized[most_similar_indices]
    recommendations[user_genre] = 1
    recommendations[user_category] = 1
    print("Top Recommended Games:")
    print(recommendations[['name', 'ratings', 'price', 'scaled_similarity_score']])


def main():
    filepath = './data/game_preference.csv'
    df = load_data(filepath)
    if df is None:
        return
    df = preprocess_genre_column(df)
    df = preprocess_categories(df)
    df_encoded = encode_categories(df)
    if df_encoded is None:
        return
    df_encoded = encode_genres(df_encoded)
    if df_encoded is None:
        return
    df_encoded = drop_unnecessary_columns(df_encoded)
    if df_encoded is None:
        return
    print("Columns after encoding and dropping unnecessary columns:", df_encoded.columns)
    df_encoded = create_player_columns(df_encoded)
    if df_encoded is None:
        return
    plot_data_distribution(df_encoded, ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount'])
    df_encoded_log_transformed = apply_log_transformation(df_encoded, ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount'])
    plot_transformed_distribution(df_encoded_log_transformed, ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount'])
    apply_boxcox_transformation(df_encoded, ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount'])
    train_data, validation_data, test_data = split_data(df_encoded)
    if train_data is None or validation_data is None or test_data is None:
        return
    train_transformed, validation_transformed, test_transformed = apply_quantile_transformation(train_data, validation_data, test_data)
    if train_transformed is None or validation_transformed is None or test_transformed is None:
        return
    plot_transformed_density_histograms(train_transformed, ['average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'ccu', 'price', 'ratings', 'owners_average', 'MutualPlayerCount'])
    train_clustered, validation_clustered, test_clustered = prepare_data_for_clustering(train_transformed, validation_transformed, test_transformed)
    if train_clustered is None or validation_clustered is None or test_clustered is None:
        return
    df_model = df_encoded.copy()
    df_model.loc[train_clustered.index, train_clustered.columns] = train_clustered
    df_model.loc[validation_clustered.index, validation_clustered.columns] = validation_clustered
    df_model.loc[test_clustered.index, test_clustered.columns] = test_clustered
    similarity_scores = calculate_similarity(df_model, ['average_forever_qt_normal', 'ccu_qt_normal', 'ratings_qt_normal'])
    if similarity_scores is None:
        return
    df_model['similarity'] = similarity_scores.tolist()
    df_model.columns = [col.lower().replace(' ', '_') for col in df_model.columns]
    recommend_games_interactive(df_model, similarity_scores, num_recommendations=5)


if __name__ == "__main__":
    main()
