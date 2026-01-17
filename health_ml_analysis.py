"""
US State Health Rankings - ML Analysis Pipeline
Clustering, Classification, and Predictive Analytics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_and_prepare_data(filepath):
    """Load health data and prepare it for ML analysis"""
    df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    
    # Main health categories
    main_cats = ['Overall', 'Health Outcomes', 'Social and Economic Factors', 
                 'Physical Environment', 'Clinical Care', 'Behaviors']
    
    # Filter to main categories and states only
    state_data = df[
        (df['Measure'].isin(main_cats)) & 
        (df['State'] != 'ALL') & 
        (df['State'] != 'DC')
    ].copy()
    
    # Pivot to get states x categories
    pivot_data = state_data.pivot_table(
        index='State',
        columns='Measure',
        values='Score',
        aggfunc='first'
    )
    
    # Convert scores to numeric
    for col in pivot_data.columns:
        pivot_data[col] = pd.to_numeric(pivot_data[col], errors='coerce')
    
    # Also get rank data
    rank_data = state_data.pivot_table(
        index='State',
        columns='Measure',
        values='Rank',
        aggfunc='first'
    )
    
    for col in rank_data.columns:
        rank_data[col] = pd.to_numeric(rank_data[col], errors='coerce')
    
    return pivot_data, rank_data


def find_optimal_clusters(data, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette scores"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def perform_clustering(data, n_clusters=4):
    """Perform K-Means clustering on health data"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_data)
    
    # Add results to dataframe
    results = data.copy()
    results['Cluster'] = clusters
    results['PCA1'] = pca_coords[:, 0]
    results['PCA2'] = pca_coords[:, 1]
    
    # Calculate cluster statistics
    cluster_stats = results.groupby('Cluster').agg({
        col: ['mean', 'std'] for col in data.columns
    }).round(3)
    
    return {
        'results': results,
        'kmeans': kmeans,
        'scaler': scaler,
        'pca': pca,
        'cluster_stats': cluster_stats,
        'explained_variance': pca.explained_variance_ratio_
    }


def label_clusters(cluster_results, rank_data):
    """Create interpretable cluster labels based on characteristics"""
    results = cluster_results['results']
    
    # Merge with rank data to understand cluster characteristics
    merged = results.merge(rank_data['Overall'], left_index=True, right_index=True, how='left')
    merged.columns = list(results.columns) + ['Overall_Rank']
    
    # Analyze each cluster
    cluster_labels = {}
    for cluster_id in sorted(results['Cluster'].unique()):
        cluster_states = results[results['Cluster'] == cluster_id]
        avg_overall = merged[merged['Cluster'] == cluster_id]['Overall_Rank'].mean()
        
        # Find strongest category for this cluster
        category_means = cluster_states.drop(['Cluster', 'PCA1', 'PCA2'], axis=1).mean()
        strongest_category = category_means.idxmax()
        weakest_category = category_means.idxmin()
        
        # Create label based on overall performance
        if avg_overall <= 15:
            label = f"High Performers"
        elif avg_overall <= 30:
            label = f"Above Average"
        elif avg_overall <= 40:
            label = f"Mixed Outcomes"
        else:
            label = f"Facing Challenges"
        
        cluster_labels[cluster_id] = {
            'label': label,
            'avg_rank': round(avg_overall, 1),
            'n_states': len(cluster_states),
            'strongest': strongest_category,
            'weakest': weakest_category,
            'states': list(cluster_states.index)
        }
    
    return cluster_labels


def train_cluster_classifier(data, clusters):
    """Train a classifier to predict cluster membership"""
    X = data.drop(['Cluster', 'PCA1', 'PCA2'], axis=1)
    y = clusters
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rf_model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Cross-validation score
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    
    return {
        'model': rf_model,
        'feature_importance': feature_importance,
        'test_accuracy': rf_model.score(X_test, y_test),
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'actuals': y_test
    }


def detect_anomalies(data):
    """Detect anomalous states using Isolation Forest"""
    features = data.drop(['Cluster', 'PCA1', 'PCA2'], axis=1, errors='ignore')
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features)
    
    data_with_anomalies = data.copy()
    data_with_anomalies['Anomaly'] = anomaly_labels
    data_with_anomalies['Anomaly_Score'] = iso_forest.score_samples(features)
    
    anomalies = data_with_anomalies[data_with_anomalies['Anomaly'] == -1]
    
    return {
        'anomaly_data': data_with_anomalies,
        'anomalies': anomalies,
        'n_anomalies': len(anomalies)
    }


def calculate_state_similarity(state, data, top_n=5):
    """Find states most similar to a given state using cosine similarity"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    features = data.drop(['Cluster', 'PCA1', 'PCA2'], axis=1, errors='ignore')
    
    if state not in features.index:
        return None
    
    state_vector = features.loc[state].values.reshape(1, -1)
    similarities = cosine_similarity(state_vector, features.values)[0]
    
    sim_df = pd.DataFrame({
        'State': features.index,
        'Similarity': similarities
    }).sort_values('Similarity', ascending=False)
    
    # Exclude the state itself
    sim_df = sim_df[sim_df['State'] != state].head(top_n)
    
    return sim_df


if __name__ == "__main__":
    # Run full analysis pipeline
    print("=" * 60)
    print("US STATE HEALTH RANKINGS - ML ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    score_data, rank_data = load_and_prepare_data('/mnt/user-data/uploads/us_health_2025.csv')
    print(f"   - Loaded {len(score_data)} states with {len(score_data.columns)} health categories")
    
    # Find optimal clusters
    print("\n2. Finding optimal number of clusters...")
    cluster_metrics = find_optimal_clusters(score_data, max_k=8)
    best_k = cluster_metrics['k_range'][np.argmax(cluster_metrics['silhouette_scores'])]
    print(f"   - Optimal clusters: {best_k} (max silhouette score: {max(cluster_metrics['silhouette_scores']):.3f})")
    
    # Perform clustering
    print(f"\n3. Performing K-Means clustering with k={best_k}...")
    cluster_results = perform_clustering(score_data, n_clusters=best_k)
    print(f"   - Explained variance by first 2 PCs: {sum(cluster_results['explained_variance']):.1%}")
    
    # Label clusters
    print("\n4. Creating interpretable cluster labels...")
    cluster_labels = label_clusters(cluster_results, rank_data)
    for cid, info in cluster_labels.items():
        print(f"   Cluster {cid}: {info['label']} (n={info['n_states']}, avg rank={info['avg_rank']})")
        print(f"      Strongest: {info['strongest']}, Weakest: {info['weakest']}")
    
    # Train classifier
    print("\n5. Training Random Forest classifier...")
    classifier_results = train_cluster_classifier(
        cluster_results['results'],
        cluster_results['results']['Cluster']
    )
    print(f"   - Test Accuracy: {classifier_results['test_accuracy']:.3f}")
    print(f"   - CV Accuracy: {classifier_results['cv_accuracy']:.3f} (±{classifier_results['cv_std']:.3f})")
    print("\n   Top 3 Most Important Features:")
    for idx, row in classifier_results['feature_importance'].head(3).iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.3f}")
    
    # Detect anomalies
    print("\n6. Detecting anomalous states...")
    anomaly_results = detect_anomalies(cluster_results['results'])
    print(f"   - Found {anomaly_results['n_anomalies']} anomalous states:")
    for state in anomaly_results['anomalies'].index:
        score = anomaly_results['anomalies'].loc[state, 'Anomaly_Score']
        print(f"      {state} (anomaly score: {score:.3f})")
    
    # Example similarity
    print("\n7. Example: States similar to California...")
    ca_similar = calculate_state_similarity('CA', cluster_results['results'], top_n=5)
    if ca_similar is not None:
        print("   Top 5 most similar states:")
        for idx, row in ca_similar.iterrows():
            print(f"      {row['State']}: {row['Similarity']:.3f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Ready for Streamlit dashboard.")
    print("=" * 60)
