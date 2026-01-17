"""
US State Health Rankings - Interactive Dashboard
Health analytics application using machine learning for state-level analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import ML functions
from health_ml_analysis import (
    load_and_prepare_data,
    perform_clustering,
    label_clusters,
    train_cluster_classifier,
    detect_anomalies,
    calculate_state_similarity
)

# Page config
st.set_page_config(
    page_title="US Health Rankings",
    page_icon="📊",  # Will be replaced by custom favicon
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Font Awesome Icons */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 4rem 2rem 4rem;
    }
    
    /* Content container with max-width */
    .main > div {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 3rem;
    }
    
    /* Hero section */
    .hero {
        background: white;
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero h1 i {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero p {
        font-size: 1.25rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Cluster badges */
    .cluster-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-above {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .badge-mixed {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .badge-challenges {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box strong {
        color: #667eea;
    }
    
    /* Card container */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    
    /* About page styling */
    .card h3 {
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .card h3:first-child {
        margin-top: 0;
    }
    
    .card ul {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    .card li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Plotly text labels - add outline for better visibility */
    .js-plotly-plot .textpoint {
        text-shadow: 
            -1px -1px 0 rgba(0,0,0,0.8),
            1px -1px 0 rgba(0,0,0,0.8),
            -1px 1px 0 rgba(0,0,0,0.8),
            1px 1px 0 rgba(0,0,0,0.8),
            0 0 4px rgba(0,0,0,0.9);
        font-weight: 900 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# State names
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming"
}

@st.cache_data
def load_data():
    score_data, rank_data = load_and_prepare_data('us_health_2025.csv')
    return score_data, rank_data

@st.cache_data
def run_analysis(score_data, rank_data, n_clusters=4):
    cluster_results = perform_clustering(score_data, n_clusters=n_clusters)
    cluster_labels = label_clusters(cluster_results, rank_data)
    classifier_results = train_cluster_classifier(
        cluster_results['results'],
        cluster_results['results']['Cluster']
    )
    anomaly_results = detect_anomalies(cluster_results['results'])
    return cluster_results, cluster_labels, classifier_results, anomaly_results

def get_cluster_badge_class(label):
    """Get badge class based on cluster label"""
    if "High Performer" in label:
        return "badge-high"
    elif "Above Average" in label:
        return "badge-above"
    elif "Mixed" in label:
        return "badge-mixed"
    else:
        return "badge-challenges"

def main():
    # Create narrow column layout - more padding on sides
    _, col_main, _ = st.columns([2, 5, 2])
    
    with col_main:
        # Hero section
        st.markdown("""
        <div class="hero">
            <h1><i class="fas fa-chart-line"></i> US State Health Rankings</h1>
            <p>Discover health patterns across America using machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        with st.spinner("🔄 Loading health data..."):
            score_data, rank_data = load_data()
            cluster_results, cluster_labels, classifier_results, anomaly_results = run_analysis(
                score_data, rank_data
            )
        
        results_df = cluster_results['results']
        
        # Key metrics
        st.markdown('<div class="section-header"><i class="fas fa-chart-bar"></i> Quick Stats</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">States Analyzed</div>
                <div class="metric-value">50</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Health Groups</div>
                <div class="metric-value">4</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ML Accuracy</div>
                <div class="metric-value">{classifier_results['test_accuracy']:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            variance = sum(cluster_results['explained_variance'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Explained</div>
                <div class="metric-value">{variance:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Spacer between metrics and tabs
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["Explore States", "Find Your State", "About This Data"])
        
        with tab1:
            st.markdown('<div class="section-header"><i class="fas fa-map-marked-alt"></i> State Health Groups</div>', unsafe_allow_html=True)
        
            st.markdown("""
            <div class="info-box">
                <strong>What am I looking at?</strong> Each dot is a state. States close together have similar health outcomes. 
                Colors show which health group they belong to.
            </div>
            """, unsafe_allow_html=True)
        
        # Create interactive visualization
            plot_df = results_df.copy()
            plot_df['State_Code'] = plot_df.index
            plot_df['State_Name'] = plot_df.index.map(STATE_NAMES)
            plot_df['Cluster_Label'] = plot_df['Cluster'].map(
                lambda x: cluster_labels[x]['label']
            )
            plot_df['Overall_Rank'] = rank_data.loc[plot_df.index, 'Overall']
        
        # Modern color scheme - vibrant
            color_map = {
                'High Performers': '#10b981',
                'Above Average': '#3b82f6',
                'Mixed Outcomes': '#f59e0b',
                'Facing Challenges': '#ef4444'
            }
        
            fig = px.scatter(
                plot_df,
                x='PCA1',
                y='PCA2',
                color='Cluster_Label',
                size='Overall_Rank',
                size_max=25,
                text='State_Code',
                hover_name='State_Name',
                hover_data={
                    'PCA1': False,
                    'PCA2': False,
                    'Cluster_Label': True,
                    'Overall_Rank': ':.0f',
                    'Behaviors': ':.2f',
                    'Clinical Care': ':.2f'
                },
                labels={
                    'PCA1': f'Health Dimension 1 ({cluster_results["explained_variance"][0]:.0%})',
                    'PCA2': f'Health Dimension 2 ({cluster_results["explained_variance"][1]:.0%})',
                    'Cluster_Label': 'Health Group'
                },
                color_discrete_map=color_map,
                height=600
            )
        
            fig.update_traces(
                textposition='middle center',
                textfont=dict(
                    size=12, 
                    color='white', 
                    family='Inter', 
                    weight=700
                ),
                marker=dict(line=dict(width=2, color='white')),
                mode='markers+text',
                cliponaxis=False
            )
        
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                legend=dict(
                    title_text='Health Groups',
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='closest'
            )
        
            st.plotly_chart(fig, use_container_width=True)
        
        # Show cluster details
            st.markdown('<div class="section-header"><i class="fas fa-layer-group"></i> Health Group Details</div>', unsafe_allow_html=True)
        
            cols = st.columns(4)
            for idx, (cid, info) in enumerate(cluster_labels.items()):
                with cols[idx]:
                    badge_class = get_cluster_badge_class(info['label'])
                    st.markdown(f"""
                    <div class="card">
                        <span class="cluster-badge {badge_class}">{info['label']}</span>
                        <h3 style="margin-top: 1rem;">{info['n_states']} States</h3>
                        <p style="color: #64748b;">Average Rank: #{info['avg_rank']:.0f}</p>
                        <p style="font-size: 0.875rem; color: #94a3b8;">
                            <strong>Best at:</strong> {info['strongest']}<br>
                            <strong>Needs work:</strong> {info['weakest']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                    with st.expander("See states in this group"):
                        for state in sorted(info['states']):
                            st.write(f"• {STATE_NAMES[state]}")
    
    with tab2:
            st.markdown('<div class="section-header"><i class="fas fa-search-location"></i> Find Your State</div>', unsafe_allow_html=True)
        
            st.markdown("""
            <div class="info-box">
                <strong>Compare your state</strong> to similar states and see what makes it unique.
            </div>
            """, unsafe_allow_html=True)
        
        # State selector
            selected_state = st.selectbox(
                "Choose a state:",
                options=sorted(results_df.index),
                format_func=lambda x: STATE_NAMES[x],
                index=4  # California default
            )
        
            state_data = results_df.loc[selected_state]
            cluster_id = int(state_data['Cluster'])
            cluster_info = cluster_labels[cluster_id]
            overall_rank = int(rank_data.loc[selected_state, 'Overall'])
        
        # State overview
            col1, col2 = st.columns([1, 2])
        
            with col1:
                badge_class = get_cluster_badge_class(cluster_info['label'])
                st.markdown(f"""
                <div class="card">
                    <h2>{STATE_NAMES[selected_state]}</h2>
                    <span class="cluster-badge {badge_class}">{cluster_info['label']}</span>
                    <h1 style="margin-top: 1rem; font-size: 3rem;">#{overall_rank}</h1>
                    <p style="color: #64748b;">Overall Health Rank</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Check if anomaly
                is_anomaly = selected_state in anomaly_results['anomalies'].index
                if is_anomaly:
                    st.markdown("""
                    <div class="card" style="background: linear-gradient(135deg, #fdcb6e15, #e1705515); border-left: 4px solid #fdcb6e;">
                        <strong><i class="fas fa-exclamation-triangle"></i> Unique Pattern Detected</strong>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #64748b;">
                        This state has an unusual combination of health factors that doesn't fit typical patterns.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
            with col2:
            # Category scores radar
                categories_for_radar = ['Behaviors', 'Clinical Care', 'Health Outcomes', 
                                        'Physical Environment', 'Social and Economic Factors']
                values = [state_data[cat] for cat in categories_for_radar]
                cluster_avg = results_df[results_df['Cluster'] == cluster_id][categories_for_radar].mean()
            
                fig = go.Figure()
            
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_for_radar,
                    fill='toself',
                    name=STATE_NAMES[selected_state],
                    line=dict(color='#667eea', width=3)
                ))
            
                fig.add_trace(go.Scatterpolar(
                    r=cluster_avg.values,
                    theta=categories_for_radar,
                    fill='toself',
                    name='Group Average',
                    line=dict(color='#94a3b8', width=2, dash='dash')
                ))
            
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[-2, 2]),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=True,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter')
                )
            
                st.plotly_chart(fig, use_container_width=True)
        
        # Similar states
            st.markdown('<div class="section-header">States Similar to ' + STATE_NAMES[selected_state] + '</div>', unsafe_allow_html=True)
        
            similar_states = calculate_state_similarity(selected_state, results_df, top_n=5)
        
            if similar_states is not None:
                cols = st.columns(5)
                for idx, (_, row) in enumerate(similar_states.iterrows()):
                    with cols[idx]:
                        similarity_pct = row['Similarity'] * 100
                        st.markdown(f"""
                        <div class="card" style="text-align: center;">
                            <h3>{STATE_NAMES[row['State']]}</h3>
                            <div style="font-size: 2rem; font-weight: 700; color: #667eea;">
                                {similarity_pct:.0f}%
                            </div>
                            <p style="font-size: 0.875rem; color: #64748b;">similar</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab3:
            st.markdown('<div class="section-header"><i class="fas fa-info-circle"></i> About This Analysis</div>', unsafe_allow_html=True)
        
            st.markdown('<div class="card">', unsafe_allow_html=True)
        
            st.markdown("### What is this?")
            st.markdown("This dashboard uses **machine learning** to find patterns in health data across all 50 US states. Instead of just ranking states, we group them by *similar health profiles*.")
        
            st.markdown("### How does it work?")
            st.markdown("We analyzed 6 major health categories:")
            st.markdown("""
            - **Behaviors** - Smoking, exercise, diet
            - **Clinical Care** - Access to doctors, hospitals
            - **Health Outcomes** - Life expectancy, disease rates
            - **Physical Environment** - Air quality, housing
            - **Social & Economic Factors** - Education, poverty, employment
            - **Overall** - Combined health score
            """)
        
            st.markdown("### Why is this useful?")
            st.markdown("States can learn from others in their group. If Vermont and New Hampshire are both 'High Performers,' they might share successful policies. If Mississippi is in 'Facing Challenges,' it can look at what interventions worked for similar states.")
        
            st.markdown("### The Machine Learning")
            st.markdown(f"We used **K-Means Clustering** to group states and **Random Forest** to predict which factors matter most. The model is **{classifier_results['test_accuracy']:.0%} accurate** at predicting health groups.")
        
            st.markdown("### Data Source")
            st.markdown("America's Health Rankings 2025 Annual Report by United Health Foundation")
        
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.875rem; padding: 2rem 0;">
        Made with <i class="fas fa-heart" style="color: #ef4444;"></i> using Python, Streamlit, and Machine Learning<br>
        Data: America's Health Rankings 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
