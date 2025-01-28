import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(layout="wide")

try:
    df = pd.read_csv("data/cleaned_KL_data.csv")  # If the CSV is in the same directory as your script
except FileNotFoundError:
    st.error("Please ensure cleaned_KL_data.csv is in the same directory as this script.")
    st.stop()

def descriptive_analytics(df):
    # Custom CSS for metric boxes
    st.markdown("""
        <style>
        .metric-box {
            background-color: #00796B;
            padding: 20px;
            border-radius: 5px;
            color: white;
            text-align: center;
            margin: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Color palette
    colors = {
        'primary': '#00796B',      # Teal
        'secondary': '#0288D1',    # Light Blue
        'tertiary': '#FFC107',     # Amber
        'accent': '#FF5722',       # Deep Orange
        'background': '#E0F2F1'    # Light Teal
    }

    # Location filter
    st.markdown("### 📍 Select Location")
    all_locations = ['All Locations'] + sorted(df['location'].unique().tolist())
    selected_location = st.selectbox('', all_locations, label_visibility='collapsed')

    if selected_location != 'All Locations':
        filtered_df = df[df['location'] == selected_location]
    else:
        filtered_df = df

    st.title("🏢 RENTAL MARKET ANALYSIS DASHBOARD")

    # Metrics row
    metrics = st.columns(6)
    metric_data = [
        ("Maximum Rental Price", f"RM {filtered_df['monthly_rent'].max():,.2f}"),
        ("Average Rental Price", f"RM {filtered_df['monthly_rent'].mean():,.2f}"),
        ("Minimum Rental Price", f"RM {filtered_df['monthly_rent'].min():,.2f}"),
        ("Total Properties", f"{len(filtered_df)}"),
        ("Average Size", f"{filtered_df['size'].mean():,.0f} sq.ft"),
        ("Properties with Parking", f"{(filtered_df['parking']==True).mean()*100:.0f}%")
    ]

    for col, (label, value) in zip(metrics, metric_data):
        col.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)

    # Charts
    row1_col1, row1_col2, row1_col3 = st.columns([1.5, 1.5, 1])

    with row1_col1:
        # Rental Price Distribution
        rent_bins = pd.cut(filtered_df['monthly_rent'], bins=15)
        rent_dist = rent_bins.value_counts().sort_index()

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=[interval.mid for interval in rent_dist.index],
            y=rent_dist.values,
            mode='lines+markers',
            line=dict(color=colors['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 121, 107, 0.2)',
            name='Count'
        ))

        fig_dist.update_layout(
            title='Rental Price Distribution',
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Monthly Rent (RM)",
            yaxis_title="Number of Properties",
            showlegend=False,
            plot_bgcolor='white',
            font=dict(weight='bold'),
            title_font_size=14
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with row1_col2:
        # Property Type Distribution (Pie Chart)
        prop_counts = filtered_df['property_type'].value_counts()
        fig_prop = go.Figure(data=[go.Pie(
            labels=prop_counts.index,
            values=prop_counts.values,
            hole=0.4,
            textinfo='percent+label',
            marker=dict(colors=[colors['primary'], colors['secondary'], colors['tertiary']])
        )])
        fig_prop.update_layout(
            title='Property Type Distribution',
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            font=dict(weight='bold'),
            title_font_size=14
        )
        st.plotly_chart(fig_prop, use_container_width=True)

    with row1_col3:
        # Scatter plot for Property Type vs Average Rent
        fig_scatter = px.scatter(
            filtered_df,
            x='size',
            y='monthly_rent',
            color='property_type',
            color_discrete_sequence=[colors['primary'], colors['secondary'], colors['tertiary']],
            title='Property Size vs Monthly Rent',
            labels={'size': 'Size (sq.ft)', 'monthly_rent': 'Monthly Rent (RM)'},
            hover_data=['location']
        )

        fig_scatter.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white',
            font=dict(weight='bold'),
            title_font_size=14,
            showlegend=False,
            xaxis=dict(tickfont=dict(weight='bold')),
            yaxis=dict(tickfont=dict(weight='bold'))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Second row of charts
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        avg_rent = filtered_df.groupby('property_type')['monthly_rent'].mean().reset_index()
        # Sort values to apply gradient colors
        avg_rent = avg_rent.sort_values('monthly_rent', ascending=True)
        
        # Create color gradient from light to dark green
        n_bars = len(avg_rent)
        colors = [f'rgba(0, 121, 107, {0.3 + (i/n_bars)*0.7})' for i in range(n_bars)]
        
        fig_avg = px.bar(
            avg_rent, 
            x='property_type', 
            y='monthly_rent',
            title='Average Rent by Property Type',
            color='monthly_rent',
            color_continuous_scale=['#b7e4d9', '#00796B'],
            width=600
        )
        fig_avg.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white',
            bargap=0.2,  # Reduced gap between bars
            coloraxis_showscale=False,
            font=dict(weight='bold'),
            xaxis=dict(tickfont=dict(weight='bold')),
            yaxis=dict(tickfont=dict(weight='bold')),
            title=dict(font=dict(weight='bold'))
        )
        st.plotly_chart(fig_avg, use_container_width=True)

    with row2_col2:
        cols = ['parking', 'additional_near_ktm/lrt']
        facilities = filtered_df[cols].mean() * 100
        # Sort values for gradient effect
        facilities = facilities.sort_values(ascending=True)
        
        fig_fac = px.bar(
            x=facilities.index, 
            y=facilities.values,
            title='Facilities Available (%)',
            labels={'x': 'Facility', 'y': 'Percentage'},
            color=facilities.values,
            color_continuous_scale=['#b7e4d9', '#00796B'],
            width=600
        )
        fig_fac.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white',
            bargap=0.2,
            coloraxis_showscale=False,
            font=dict(weight='bold'),
            xaxis=dict(tickfont=dict(weight='bold')),
            yaxis=dict(tickfont=dict(weight='bold')),
            title=dict(font=dict(weight='bold'))
        )
        st.plotly_chart(fig_fac, use_container_width=True)


# L

def predict_price(features, xgb_model, encoders):
    """Make rental price prediction using saved XGBoost model
    
    Args:
        features (dict): Dictionary containing the input features
        xgb_model: Loaded XGBoost model from pickle
        encoders: Loaded encoders from pickle
        
    Returns:
        dict: Contains prediction, feature importance, and input features
    """
    try:
        # Create DataFrame with input features
        input_df = pd.DataFrame([features])
        
        # Encode categorical variables
        input_df['location_encoded'] = encoders['location'].transform([features['location']])
        input_df['property_type_encoded'] = encoders['property_type'].transform([features['property_type']])
        input_df['furnished_encoded'] = encoders['furnished'].transform([features['furnished']])
        
        # Calculate derived features
        #input_df['property_age'] = 2024 - input_df['completion_year']
        #input_df['price_per_sqft'] = 0
        
        # Select and order features
        feature_order = [
            'size',
            'rooms',
            'bathroom',
            'parking',
            'additional_near_ktm/lrt',
            'location_encoded',
            'property_type_encoded',
            'furnished_encoded'
            
        ]
        
        # Prepare final features
        X_pred = input_df[feature_order]
        
        # Convert boolean to int
        X_pred['parking'] = X_pred['parking'].astype(int)
        X_pred['additional_near_ktm/lrt'] = X_pred['additional_near_ktm/lrt'].astype(int)
        
        # Make prediction
        prediction = xgb_model.predict(X_pred)[0]
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_order,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'prediction': prediction,
            'feature_importance': feature_importance,
            'input_features': X_pred
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Debug information:")
        st.write("Input features:", features)
        st.write("Available encoders:", list(encoders.keys()))
        return None


def show_market_analysis(df):

    """Display market analysis visualizations"""
    st.subheader("Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average rent by property type
        fig_property = px.box(
            df,
            x='property_type',
            y='monthly_rent',
            title='Rent Distribution by Property Type'
        )
        st.plotly_chart(fig_property)
    
    with col2:
        # Average rent by size category
        avg_size = df.groupby('size_category')['monthly_rent'].mean().reset_index()
        fig_size = px.bar(
            avg_size,
            x='size_category',
            y='monthly_rent',
            title='Average Rent by Size Category'
        )
        st.plotly_chart(fig_size)

def calculate_location_stats(df):
    """
    Calculate comprehensive location statistics
    
    Parameters:
    -----------
    df : pandas DataFrame
        The rental dataset
        
    Returns:
    --------
    DataFrame
        Location-wise statistics
    """
    try:
        # Calculate basic statistics for each location
        location_stats = df.groupby('location').agg({
            'monthly_rent': ['mean', 'median', 'min', 'max', 'count'],
            'size': ['mean', 'median'],
            'rooms': ['mean', 'median'],
            'bathroom': ['mean'],
            'parking': ['mean'],
            'facility_gymnasium': ['mean']
        }).round(2)
        
        # Flatten the column names
        location_stats.columns = [
            f"{col[0]}_{col[1]}" for col in location_stats.columns
        ]
        
        # Reset index to make location a column
        location_stats = location_stats.reset_index()
        
        # Add price per sqft
        location_stats['price_per_sqft'] = (
            location_stats['monthly_rent_mean'] / location_stats['size_mean']
        ).round(2)
        
        # Add market position
        overall_mean = df['monthly_rent'].mean()
        location_stats['market_position'] = (
            (location_stats['monthly_rent_mean'] - overall_mean) / overall_mean * 100
        ).round(2)
        
        return location_stats
    
    except Exception as e:
        st.error(f"Error calculating location statistics: {str(e)}")
        st.write("Debug information:")
        st.write("DataFrame columns:", df.columns.tolist())
        return pd.DataFrame()  # Return empty DataFrame on error
    
def create_feature_inputs():
   """Create input fields for property features"""
   # Load the original KL dataset to get valid categories
   df = st.session_state['df']
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.subheader("Location Details")
       location = st.selectbox(
           "Location",
           options=sorted(df['location'].unique()),
           help="Only KL locations are available for prediction"
       )
       
       property_type = st.selectbox(
           "Property Type",
           options=sorted(df['property_type'].unique()),
           help="Only property types from KL dataset are available"
       )
       
       size = st.number_input(
           "Size (sq ft)",
           min_value=100,
           max_value=10000,
           value=1000
       )
   
   with col2:
       st.subheader("Property Features")
       rooms = st.number_input(
           "Number of Rooms",
           min_value=0,
           max_value=10,
           value=3
       )
       
       bathrooms = st.number_input(
           "Number of Bathrooms",
           min_value=0,
           max_value=10,
           value=2
       )
       
       parking = st.checkbox("Parking Available", value=True)
       near_ktm_lrt = st.checkbox("Near KTM/LRT Station", value=False)
   
   with col3:
       st.subheader("Additional Details")
       furnished = st.selectbox(
           "Furnished Status",
           options=sorted(df['furnished'].unique()),
           help="Only furnished status from KL dataset are available"
       )

   return {
       'location': location,
       'property_type': property_type,
       'size': size,
       'furnished': furnished,
       'rooms': rooms,
       'bathroom': bathrooms,
       'parking': parking,
       'additional_near_ktm/lrt': near_ktm_lrt
       
   }

def add_rental_suggestions(df):
    """Add rental suggestions based on user preferences"""
    st.title("🏠 Kuala Lumpur Rental Property Finder")
    st.write("Let us help you find your ideal rental property based on your preferences!")
    
    # Create input form
    with st.form(key="property_preferences_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_pref = st.selectbox(
                "Preferred Location",
                options=['Any'] + sorted(df['location'].unique().tolist()),
                key='location'
            )
            
            property_type_pref = st.selectbox(
                "Property Type",
                options=['Any'] + sorted(df['property_type'].unique().tolist()),
                key='property_type'
            )
            
            rooms_pref = st.number_input(
                "Number of Rooms",
                min_value=0,
                max_value=int(df['rooms'].max()),
                value=0,
                step=1,
                key='rooms'
            )
        
        with col2:
            bathroom_pref = st.number_input(
                "Number of Bathrooms",
                min_value=0,
                max_value=int(df['bathroom'].max()),
                value=0,
                step=1,
                key='bathrooms'
            )
            
            furnished_pref = st.selectbox(
                "Furnished Status",
                options=['Any', 'Fully Furnished', 'Partially Furnished', 'Not Furnished'],
                key='furnished'
            )
            
            parking_pref = st.selectbox(
                "Parking Required",
                options=['Any', 'Yes', 'No'],
                key='parking'
            )
        
        with col3:
            ktm_lrt_pref = st.selectbox(
                "Near KTM/LRT",
                options=['Any', 'Yes', 'No'],
                key='ktm_lrt'
            )
            
            max_rent = st.number_input(
                "Maximum Monthly Rent (RM)",
                min_value=0,
                max_value=int(df['monthly_rent'].max()),
                value=int(df['monthly_rent'].median()),
                step=100,
                key='max_rent'
            )
        
        # Add submit button
        submit_button = st.form_submit_button("Find Properties")
    
    if submit_button:
        # Filter properties based on preferences
        filtered_df = df.copy()
        
        # Apply filters
        if location_pref != 'Any':
            filtered_df = filtered_df[filtered_df['location'] == location_pref]
            
        if property_type_pref != 'Any':
            filtered_df = filtered_df[filtered_df['property_type'] == property_type_pref]
            
        if rooms_pref > 0:
            filtered_df = filtered_df[filtered_df['rooms'] >= rooms_pref]
            
        if bathroom_pref > 0:
            filtered_df = filtered_df[filtered_df['bathroom'] >= bathroom_pref]
            
        if furnished_pref != 'Any':
            filtered_df = filtered_df[filtered_df['furnished'] == furnished_pref]
            
        if parking_pref != 'Any':
            has_parking = True if parking_pref == 'Yes' else False
            filtered_df = filtered_df[filtered_df['parking'] == has_parking]
            
        if ktm_lrt_pref != 'Any':
            near_ktm = True if ktm_lrt_pref == 'Yes' else False
            filtered_df = filtered_df[filtered_df['additional_near_ktm/lrt'] == near_ktm]
            
        filtered_df = filtered_df[filtered_df['monthly_rent'] <= max_rent]
        
        # Display results
        if len(filtered_df) > 0:
            st.success(f"Found {len(filtered_df)} matching properties!")
            
            # Sort by rent for better display
            filtered_df = filtered_df.sort_values('monthly_rent')
            
            # Display properties in cards
            for idx, row in filtered_df.iterrows():
                with st.container():
                    
                    
                    
                        st.markdown(f"""
                        ### {row['property_type']} in {row['location']}
                        - **Property Name:** {row['prop_name']}
                        - **Monthly Rent:** RM {row['monthly_rent']:,.2f}
                        - **Rooms:** {int(row['rooms'])} | **Bathrooms:** {int(row['bathroom'])}
                        - **Size:** {int(row['size'])} sq ft
                        - **Furnished Status:** {row['furnished']}
                        - **Parking:** {'Available' if row['parking'] else 'Not Available'}
                        - **Near KTM/LRT:** {'Yes' if row['additional_near_ktm/lrt'] else 'No'}
                        """)
                    
                        st.divider()
        else:
            st.error("Oops! No properties match your preferences. Try adjusting your criteria.")
            
            # Show nearby suggestions
            st.subheader("Suggested Alternatives")
            alt_df = df[
                (df['location'] == location_pref if location_pref != 'Any' else True) &
                (df['property_type'] == property_type_pref if property_type_pref != 'Any' else True)
            ].copy()
            
            if len(alt_df) > 0:
                st.info("Here are some properties that partially match your criteria:")
                
                # Show top 3 closest matches
                alt_df = alt_df.sort_values('monthly_rent').head(3)
                
                for idx, row in alt_df.iterrows():
                    with st.container():
                        st.markdown(f"""
                        ### {row['property_type']} in {row['location']}
                        - **Property Name:** {row['prop_name']}
                        - **Monthly Rent:** RM {row['monthly_rent']:,.2f}
                        - **Rooms:** {int(row['rooms'])} | **Bathrooms:** {int(row['bathroom'])}
                        - **Furnished Status:** {row['furnished']}
                        """)
                        st.divider()
            else:
                st.info("Try broadening your search criteria to see more options.")


def load_default_dataset():
    """Load the default KL dataset and train model"""
    try:
        df = pd.read_csv('data/cleaned_KL_data.csv')
        
        # Train model for default dataset
        encoders = {
            'location': LabelEncoder(),
            'property_type': LabelEncoder(),
            'furnished': LabelEncoder()
        }
        
        # Create a copy for modeling
        model_df = df.copy()
        
        # Encode categorical variables
        for column, encoder in encoders.items():
            model_df[f'{column}_encoded'] = encoder.fit_transform(model_df[column])
        
        # Select features
        feature_list = [
            'size',
            'rooms',
            'bathroom',
            'parking',
            'additional_near_ktm/lrt',
            'location_encoded',
            'property_type_encoded',
            'furnished_encoded'
        ]
        
        X = model_df[feature_list]
        y = model_df['monthly_rent']
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X, y)
        
        # Store model artifacts
        st.session_state['model_artifacts'] = {
            'model': xgb_model,
            'encoders': encoders,
            'feature_list': feature_list
        }
        
        return df
        
    except Exception as e:
        st.error(f"Error loading default dataset: {str(e)}")
        return None

def add_model_comparison():
    """Add model comparison visualizations"""
    st.title("Model Performance Comparison")

    metrics = {
        'XGBoost': {
            'MAE': 288.44,  # Replace with your actual MAE
            'RMSE': 402.21, # Replace with your actual RMSE
            'R2': 0.7367    # Replace with your actual R2
        },
        'Random Forest': {
            'MAE': 295.95,  # Replace with your actual MAE
            'RMSE': 421.41, # Replace with your actual RMSE
            'R2': 0.7275     # Replace with your actual R2
        },
        'Linear Regression': {
            'MAE': 425.32,  # Replace with your actual MAE
            'RMSE': 589.67, # Replace with your actual RMSE
            'R2': 0.4782    # Replace with your actual R2
        }
    }

    # Create DataFrame for metrics
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')

    # Create a single-page layout using columns
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # 1. MAE Comparison
        st.subheader("Mean Absolute Error (MAE) Comparison")
        fig_mae = px.bar(
            df_metrics,
            y='MAE',
            title='Mean Absolute Error by Model',
            color=df_metrics.index,
            labels={'index': 'Model', 'value': 'MAE (RM)'}
        )
        fig_mae.update_traces(texttemplate='RM %{y:.2f}', textposition='outside')
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        # 2. RMSE Comparison
        st.subheader("Root Mean Square Error (RMSE) Comparison")
        fig_rmse = px.bar(
            df_metrics,
            y='RMSE',
            title='Root Mean Square Error by Model',
            color=df_metrics.index,
            labels={'index': 'Model', 'value': 'RMSE (RM)'}
        )
        fig_rmse.update_traces(texttemplate='RM %{y:.2f}', textposition='outside')
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col3:
        # 3. R2 Score Comparison
        st.subheader("R² Score Comparison")
        fig_r2 = px.bar(
            df_metrics,
            y='R2',
            title='R² Score by Model',
            color=df_metrics.index,
            labels={'index': 'Model', 'value': 'R²'}
        )
        fig_r2.update_traces(texttemplate='%{y:.4f}', textposition='outside')
        st.plotly_chart(fig_r2, use_container_width=True)

    # Add metrics table below charts
    st.subheader("Performance Metrics Summary")
    styled_metrics = pd.DataFrame({
        'Model': df_metrics.index,
        'Mean Absolute Error': df_metrics['MAE'].apply(lambda x: f'RM {x:,.2f}'),
        'Root Mean Square Error': df_metrics['RMSE'].apply(lambda x: f'RM {x:,.2f}'),
        'R² Score': df_metrics['R2'].apply(lambda x: f'{x:.4f}')
    }).set_index('Model')

    st.dataframe(styled_metrics)

    # Add explanation below metrics table
    st.info("""
    **Understanding the Metrics:**

    1. **Mean Absolute Error (MAE)**
       - Represents the average absolute difference between predicted and actual rental prices
       - Lower values indicate better performance
       - More intuitive as it's in the same unit as rental prices (RM)

    2. **Root Mean Square Error (RMSE)**
       - Square root of the average squared prediction errors
       - Penalizes larger errors more heavily than MAE
       - Also in rental price units (RM)
       - Lower values indicate better performance

    3. **R² Score**
       - Indicates how well the model explains the variance in rental prices
       - Ranges from 0 to 1 (1 being perfect prediction)
       - Higher values indicate better model fit

    **Key Findings:**
    - XGBoost shows the best overall performance with lowest errors and highest R²
    - Random Forest performs similarly well, showing robust prediction capability
    - Linear Regression shows higher errors, indicating rental prices have non-linear relationships with features
    """)

def load_all_models():
    """Load all three saved models and their artifacts"""
    try:
        # Load XGBoost model
        with open("model/tuned_xgboost_model.pkl", 'rb') as file:  # Changed path
            xgb_artifacts = pickle.load(file)
        
        return {
            'xgboost': {
                'model': xgb_artifacts['model'],
                'encoders': xgb_artifacts['encoders']
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def main():
    try:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Rental Suggestions",
            "Rental Prediction",
            "Descriptive Analytics",
            "🔍Model Comparison"
        ])
        
        if 'df' not in st.session_state:
            df = load_default_dataset()
            st.session_state['df'] = df
        
        # Get dataset from session state
        df = st.session_state['df']
        # Load data once
        models = load_all_models()
        if models is None:
            st.error("Failed to load models")
            return
        
        # Load model and encoders
        # Calculate location statistics for predictions
        location_stats = calculate_location_stats(df)
        
        with tab1:
            add_rental_suggestions(df)

        with tab2:
            st.title("🏠 Kuala Lumpur Rental Price Prediction")
            st.write("Powered by XGBoost machine learning model with 73.67% accuracy")
            # Get user inputs
            features = create_feature_inputs()
            
            if st.button("Predict Rental Price", type="primary"):
                result = predict_price(features, models['xgboost']['model'], models['xgboost']['encoders'])
                
                if result is not None:
                    prediction = result['prediction']
                    
                    # Show prediction
                    st.subheader("Prediction Results")
                    st.success(f"Predicted Monthly Rent: RM {prediction:,.2f}")
                    
                    # Show location context
                    location_stats_filtered = location_stats[
                        location_stats['location'] == features['location']
                    ].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Rent",
                            f"RM {prediction:,.2f}",
                            f"{((prediction - location_stats_filtered['monthly_rent_mean']) / location_stats_filtered['monthly_rent_mean'] * 100):,.1f}% vs. average"
                        )
                    
                    with col2:
                        st.metric(
                            "Location Average",
                            f"RM {location_stats_filtered['monthly_rent_mean']:,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Location Median",
                            f"RM {location_stats_filtered['monthly_rent_median']:,.2f}"
                        )
                    
                    # Show feature importance
                    if 'feature_importance' in result:
                        st.subheader("Feature Importance")
                        fig = px.bar(
                            result['feature_importance'].head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 10 Most Important Features in Prediction'
                        )
                        st.plotly_chart(fig)

                
        with tab3:
            descriptive_analytics(df)
        
        with tab4:
            add_model_comparison()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try refreshing the page or contact support if the error persists.")

if __name__ == "__main__":
    main()
