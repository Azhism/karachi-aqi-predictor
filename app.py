"""
Streamlit Dashboard for Karachi AQI Predictor
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from src.database import MongoDBHandler
from src.model_registry import ModelRegistry

# Page configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern & Professional
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .pred-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .pred-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .pred-category {
        font-size: 1.1rem;
        font-weight: 600;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* AQI Badge Colors */
    .aqi-good { background: linear-gradient(135deg, #00e400 0%, #00b300 100%); }
    .aqi-satisfactory { background: linear-gradient(135deg, #ffff00 0%, #ffcc00 100%); color: #333; }
    .aqi-moderate { background: linear-gradient(135deg, #ff7e00 0%, #ff5500 100%); }
    .aqi-poor { background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%); }
    .aqi-verypoor { background: linear-gradient(135deg, #8f3f97 0%, #6b2f73 100%); }
    .aqi-severe { background: linear-gradient(135deg, #7e0023 0%, #5a0019 100%); }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def init_registry():
    """Initialize model registry"""
    registry = ModelRegistry()
    registry.load_all_models()
    return registry

@st.cache_resource
def init_database():
    """Initialize database connection"""
    return MongoDBHandler()

def get_aqi_category_and_style(aqi_value):
    """Get AQI category, color, and CSS class"""
    if aqi_value <= 1:
        return "Good", "#00e400", "aqi-good", "😊"
    elif aqi_value <= 2:
        return "Satisfactory", "#ffff00", "aqi-satisfactory", "🙂"
    elif aqi_value <= 3:
        return "Moderate", "#ff7e00", "aqi-moderate", "😐"
    elif aqi_value <= 4:
        return "Poor", "#ff0000", "aqi-poor", "😷"
    elif aqi_value <= 5:
        return "Very Poor", "#8f3f97", "aqi-verypoor", "😨"
    else:
        return "Severe", "#7e0023", "aqi-severe", "☠️"

def main():
    # Header with AQI Scale on the right
    col_header, col_scale = st.columns([3, 1])
    
    with col_header:
        st.markdown('<h1 class="main-header" style="text-align: left; padding-left: 2rem;">🌍 Karachi Air Quality Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="text-align: left; padding-left: 2rem;">Real-time AQI predictions powered by Machine Learning | Updated hourly with live data</p>', unsafe_allow_html=True)
    
    with col_scale:
        st.markdown("""<div style='font-size: 0.75rem; margin-top: 1rem;'>
        <strong>📊 AQI Scale</strong><br>
        <div style='padding: 0.2rem; background: #00e400; border-radius: 3px; margin: 0.1rem 0; color: white;'>🟢 Good (0-1)</div>
        <div style='padding: 0.2rem; background: #ffcc00; color: #333; border-radius: 3px; margin: 0.1rem 0;'>🟡 Satisfactory (1-2)</div>
        <div style='padding: 0.2rem; background: #ff7e00; border-radius: 3px; margin: 0.1rem 0; color: white;'>🟠 Moderate (2-3)</div>
        <div style='padding: 0.2rem; background: #ff0000; border-radius: 3px; margin: 0.1rem 0; color: white;'>🔴 Poor (3-4)</div>
        <div style='padding: 0.2rem; background: #8f3f97; border-radius: 3px; margin: 0.1rem 0; color: white;'>🟣 Very Poor (4-5)</div>
        <div style='padding: 0.2rem; background: #7e0023; border-radius: 3px; margin: 0.1rem 0; color: white;'>⚫ Severe (5+)</div>
        </div>""", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        st.markdown("**Model Selection**")
        model_option = st.radio(
            "Choose prediction model:",
            ["Best Model (Auto)", "RandomForest", "XGBoost", "LightGBM"],
            label_visibility="collapsed"
        )
        
        selected_model = None if model_option == "Best Model (Auto)" else model_option
        
        st.divider()
        
        # About section
        st.markdown("### 📖 About This App")
        st.markdown("""
        This dashboard predicts **Air Quality Index (AQI)** for Karachi using:
        
        - 🤖 **Machine Learning** models
        - 📊 **78+ days** of historical data
        - 🔄 **Hourly updates** via GitHub Actions
        - 🌐 **Live APIs** (Open-Meteo & OpenWeather)
        
        **Prediction Horizons:**
        - ☀️ **24h**: Tomorrow's AQI
        - 🌤️ **48h**: Day after tomorrow
        - 🌥️ **72h**: 3 days ahead
        """)
        
        st.divider()
        
        st.divider()
        
        # Tech Stack
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **ML**: Scikit-learn, XGBoost, LightGBM
        - **Database**: MongoDB Atlas
        - **Automation**: GitHub Actions
        - **APIs**: Open-Meteo, OpenWeather
        """)
        
        # Refresh button
        st.markdown("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content
    try:
        # Load registry and database
        registry = init_registry()
        db = init_database()
        
        # Get predictions
        with st.spinner("🔮 Generating predictions..."):
            predictions = registry.predict_multi_horizon(model_name=selected_model)
        
        if not predictions:
            st.error("❌ Unable to generate predictions. Please check the model and data.")
            return
        
        # Get current AQI from latest data
        df_current = db.get_latest_features(n_hours=1)
        if not df_current.empty:
            current_aqi = df_current.iloc[0]['aqi']
            current_time = df_current.iloc[0]['datetime']
            current_category, current_color, current_class, current_emoji = get_aqi_category_and_style(current_aqi)
            
            # Current AQI with Gauge
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # AQI Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_aqi,
                    title={'text': "Current AQI", 'font': {'size': 24, 'color': '#2c3e50'}},
                    number={'font': {'size': 60, 'color': current_color}},
                    gauge={
                        'axis': {'range': [0, 6], 'tickwidth': 2, 'tickcolor': "#2c3e50"},
                        'bar': {'color': current_color, 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e9ecef",
                        'steps': [
                            {'range': [0, 1], 'color': '#00e40044'},
                            {'range': [1, 2], 'color': '#ffff0044'},
                            {'range': [2, 3], 'color': '#ff7e0044'},
                            {'range': [3, 4], 'color': '#ff000044'},
                            {'range': [4, 5], 'color': '#8f3f9744'},
                            {'range': [5, 6], 'color': '#7e002344'}
                        ],
                        'threshold': {
                            'line': {'color': current_color, 'width': 4},
                            'thickness': 0.75,
                            'value': current_aqi
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=80, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter'}
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Info Card with better contrast
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 15px; height: 100%; display: flex; flex-direction: column; justify-content: center;
                            box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                    <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 600;">Current Air Quality in Karachi</h2>
                    <div style="margin: 1.5rem 0;">
                        <span style="font-size: 5rem; line-height: 1;">{current_emoji}</span>
                    </div>
                    <h3 style="color: white; margin: 0.5rem 0; font-size: 2.5rem; font-weight: 700;">{current_category}</h3>
                    <p style="color: rgba(255,255,255,0.9); margin: 1rem 0 0 0; font-size: 1rem;">Last updated: {current_time.strftime('%B %d, %Y at %H:%M')}</p>
                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.15); border-radius: 10px;">
                        <p style="color: white; margin: 0; font-size: 0.9rem; line-height: 1.6;">
                            <strong>AQI Level {int(current_aqi)}</strong> - {'Air quality is acceptable.' if current_aqi <= 2 else 'Air quality is degraded.' if current_aqi <= 3 else 'Health effects may be experienced.' if current_aqi <= 4 else 'Health alert: everyone may experience serious effects.'}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display model info
        st.info(f"🤖 **Model:** {predictions['model_used']} | **Accuracy:** {predictions['model_metrics']['accuracy']:.1%} | **Precision:** {predictions['model_metrics']['precision']:.1%}")
        
        # Predictions Section
        st.markdown('<h2 class="section-header">🔮 Future AQI Predictions</h2>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        # 24h prediction
        if '24h_ahead' in predictions:
            pred_24h = predictions['24h_ahead']
            category_24h, color_24h, class_24h, emoji_24h = get_aqi_category_and_style(pred_24h['prediction'])
            
            with cols[0]:
                st.markdown(f"""
                <div class="prediction-card {class_24h}">
                    <div class="pred-label">Tomorrow</div>
                    <div class="pred-value">{emoji_24h} {int(pred_24h['prediction'])}</div>
                    <div class="pred-category">{category_24h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_24h['prediction_time'].strftime('%b %d, %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 48h prediction
        if '48h_ahead' in predictions:
            pred_48h = predictions['48h_ahead']
            category_48h, color_48h, class_48h, emoji_48h = get_aqi_category_and_style(pred_48h['prediction'])
            
            with cols[1]:
                st.markdown(f"""
                <div class="prediction-card {class_48h}">
                    <div class="pred-label">Day After Tomorrow</div>
                    <div class="pred-value">{emoji_48h} {int(pred_48h['prediction'])}</div>
                    <div class="pred-category">{category_48h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_48h['prediction_time'].strftime('%b %d, %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 72h prediction
        if '72h_ahead' in predictions:
            pred_72h = predictions['72h_ahead']
            category_72h, color_72h, class_72h, emoji_72h = get_aqi_category_and_style(pred_72h['prediction'])
            
            with cols[2]:
                st.markdown(f"""
                <div class="prediction-card {class_72h}">
                    <div class="pred-label">3 Days Ahead</div>
                    <div class="pred-value">{emoji_72h} {int(pred_72h['prediction'])}</div>
                    <div class="pred-category">{category_72h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_72h['prediction_time'].strftime('%b %d, %H:%M')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("")
        st.markdown("")
        
        # Historical Data Section
        st.markdown('<h2 class="section-header">📈 Historical AQI Trends</h2>', unsafe_allow_html=True)
        
        # Get historical data
        df_history = db.get_latest_features(n_hours=168)  # Last 7 days
        
        if not df_history.empty:
            df_history = df_history.sort_values('datetime')
            
            # Plot historical AQI
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['datetime'],
                y=df_history['aqi'],
                mode='lines+markers',
                name='Historical AQI',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            # Add prediction points
            pred_times = []
            pred_values = []
            
            if '24h_ahead' in predictions:
                pred_times.append(predictions['24h_ahead']['prediction_time'])
                pred_values.append(predictions['24h_ahead']['prediction'])
            if '48h_ahead' in predictions:
                pred_times.append(predictions['48h_ahead']['prediction_time'])
                pred_values.append(predictions['48h_ahead']['prediction'])
            if '72h_ahead' in predictions:
                pred_times.append(predictions['72h_ahead']['prediction_time'])
                pred_values.append(predictions['72h_ahead']['prediction'])
            
            if pred_times:
                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_values,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=12, color='red', symbol='star')
                ))
            
            fig.update_layout(
                title={
                    'text': "AQI Trend - Last 7 Days + Future Predictions",
                    'font': {'size': 20, 'color': '#2c3e50', 'family': 'Inter'}
                },
                xaxis_title="Date & Time",
                yaxis_title="AQI Level",
                hovermode='x unified',
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No historical data available")
        
        st.divider()
        
        # Model Comparison Section
        st.markdown('<h2 class="section-header">🏆 Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get all models from registry
            if registry.model_metadata:
                model_data = []
                for model_name, metadata in registry.model_metadata.items():
                    model_data.append({
                        'Model': model_name,
                        'Accuracy': f"{metadata['metrics']['accuracy']:.1%}",
                        'Precision': f"{metadata['metrics']['precision']:.1%}",
                        'F1 Score': f"{metadata['metrics']['f1_score']:.1%}",
                        'Best': '🥇' if metadata.get('is_best', False) else ''
                    })
                
                df_models = pd.DataFrame(model_data)
                df_models = df_models.sort_values('Accuracy', ascending=False)
                
                st.dataframe(
                    df_models,
                    hide_index=True,
                    width='stretch'
                )
        
        with col2:
            # Accuracy comparison chart
            if registry.model_metadata:
                # Extract numeric accuracy values for chart
                acc_data = []
                for model_name, metadata in registry.model_metadata.items():
                    acc_data.append({
                        'Model': model_name,
                        'Accuracy': metadata['metrics']['accuracy'] * 100,
                        'Best': metadata.get('is_best', False)
                    })
                
                fig_acc = go.Figure(data=[
                    go.Bar(
                        x=[m['Model'] for m in acc_data],
                        y=[m['Accuracy'] for m in acc_data],
                        marker_color=['#7fcd00' if m.get('Best') else '#1f77b4' for m in acc_data]
                    )
                ])
                
                fig_acc.update_layout(
                    title={
                        'text': "Model Accuracy Comparison",
                        'font': {'size': 16, 'color': '#2c3e50', 'family': 'Inter'}
                    },
                    xaxis_title="Model",
                    yaxis_title="Accuracy (%)",
                    height=300,
                    template='plotly_white',
                    showlegend=False,
                    font={'family': 'Inter'}
                )
                
                st.plotly_chart(fig_acc, width='stretch')
        
        st.divider()
        
        # Statistics
        st.markdown('<h2 class="section-header">📊 Dataset Statistics</h2>', unsafe_allow_html=True)
        
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Total Records", f"{len(df_history):,}")
        
        with stat_cols[1]:
            if not df_history.empty:
                st.metric("Avg AQI", f"{df_history['aqi'].mean():.1f}")
            else:
                st.metric("Avg AQI", "N/A")
        
        with stat_cols[2]:
            if not df_history.empty:
                st.metric("Max AQI", f"{df_history['aqi'].max():.1f}")
            else:
                st.metric("Max AQI", "N/A")
        
        with stat_cols[3]:
            if not df_history.empty:
                st.metric("Min AQI", f"{df_history['aqi'].min():.1f}")
            else:
                st.metric("Min AQI", "N/A")
        
        # Custom Footer
        st.markdown("""
        <div class="custom-footer">
            <p><strong>Karachi AQI Predictor</strong> | Powered by Machine Learning</p>
            <p>Data updated hourly via GitHub Actions | Predictions based on 78+ days of historical data</p>
            <p>Models: RandomForest, XGBoost, LightGBM | Database: MongoDB Atlas</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">© 2026 | Built with ❤️ using Streamlit & Python</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
