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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .metric-container {
        text-align: center;
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

def get_aqi_category(aqi_value):
    """Get AQI category and color"""
    if aqi_value <= 1:
        return "Good", "#00e400"
    elif aqi_value <= 2:
        return "Satisfactory", "#ffff00"
    elif aqi_value <= 3:
        return "Moderate", "#ff7e00"
    elif aqi_value <= 4:
        return "Poor", "#ff0000"
    elif aqi_value <= 5:
        return "Very Poor", "#8f3f97"
    else:
        return "Severe", "#7e0023"

def main():
    # Header
    st.markdown('<p class="main-header">🌍 Karachi AQI Predictor</p>', unsafe_allow_html=True)
    st.markdown("**Real-time Air Quality Index predictions powered by Machine Learning**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Selection")
        model_option = st.radio(
            "Choose prediction model:",
            ["Best Model (Auto)", "RandomForest", "XGBoost", "LightGBM"]
        )
        
        selected_model = None if model_option == "Best Model (Auto)" else model_option
        
        st.divider()
        
        # About section
        st.header("📖 About")
        st.info(
            "This dashboard predicts Air Quality Index (AQI) "
            "for Karachi using historical weather and pollution data. "
            "\n\n**Prediction Horizons:**\n"
            "- 24h: Tomorrow's AQI\n"
            "- 48h: Day after tomorrow\n"
            "- 72h: 3 days ahead"
        )
        
        st.divider()
        
        # AQI Scale
        st.header("📊 AQI Scale")
        st.markdown("""
        - 🟢 **Good** (0-1)
        - 🟡 **Satisfactory** (1-2)
        - 🟠 **Moderate** (2-3)
        - 🔴 **Poor** (3-4)
        - 🟣 **Very Poor** (4-5)
        - ⚫ **Severe** (5+)
        """)
    
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
        
        # Display model info
        st.success(f"✅ Using Model: **{predictions['model_used']}** | Accuracy: {predictions['model_metrics']['accuracy']:.1%}")
        
        # Predictions Section
        st.header("🔮 AQI Predictions")
        
        cols = st.columns(3)
        
        # 24h prediction
        if '24h_ahead' in predictions:
            pred_24h = predictions['24h_ahead']
            category_24h, color_24h = get_aqi_category(pred_24h['prediction'])
            
            with cols[0]:
                st.markdown("### 📍 Tomorrow (24h)")
                st.metric(
                    label="Predicted AQI",
                    value=f"{int(pred_24h['prediction'])}",
                    delta=category_24h
                )
                st.caption(f"Prediction for: {pred_24h['prediction_time'].strftime('%b %d, %Y %H:%M')}")
                st.markdown(f"**Category:** :{color_24h.replace('#', '')}[{category_24h}]")
        
        # 48h prediction
        if '48h_ahead' in predictions:
            pred_48h = predictions['48h_ahead']
            category_48h, color_48h = get_aqi_category(pred_48h['prediction'])
            
            with cols[1]:
                st.markdown("### 📍 Day After (48h)")
                st.metric(
                    label="Predicted AQI",
                    value=f"{int(pred_48h['prediction'])}",
                    delta=category_48h
                )
                st.caption(f"Prediction for: {pred_48h['prediction_time'].strftime('%b %d, %Y %H:%M')}")
                st.markdown(f"**Category:** :{color_48h.replace('#', '')}[{category_48h}]")
        
        # 72h prediction
        if '72h_ahead' in predictions:
            pred_72h = predictions['72h_ahead']
            category_72h, color_72h = get_aqi_category(pred_72h['prediction'])
            
            with cols[2]:
                st.markdown("### 📍 3 Days Later (72h)")
                st.metric(
                    label="Predicted AQI",
                    value=f"{int(pred_72h['prediction'])}",
                    delta=category_72h
                )
                st.caption(f"Prediction for: {pred_72h['prediction_time'].strftime('%b %d, %Y %H:%M')}")
                st.markdown(f"**Category:** :{color_72h.replace('#', '')}[{category_72h}]")
        
        st.divider()
        
        # Historical Data Section
        st.header("📈 Historical AQI Trends")
        
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
                title="AQI Trend - Last 7 Days + Predictions",
                xaxis_title="Date",
                yaxis_title="AQI Value",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No historical data available")
        
        st.divider()
        
        # Model Comparison Section
        st.header("🏆 Model Performance Comparison")
        
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
                    title="Accuracy Comparison",
                    xaxis_title="Model",
                    yaxis_title="Accuracy (%)",
                    height=300
                )
                
                st.plotly_chart(fig_acc, width='stretch')
        
        st.divider()
        
        # Statistics
        st.header("📊 Dataset Statistics")
        
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
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
