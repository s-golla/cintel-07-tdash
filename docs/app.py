"""
Penguins Dashboard Example using Shiny for Python
This app demonstrates interactive filtering, summary statistics, data visualization,
and a new predictive analytics module using a simulated time-series.
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import seaborn as sns # For data visualization
from faicons import icon_svg # For icon support in UI
from shiny import reactive # For reactive programming
from shiny.express import input, render, ui # Shiny for Python UI and server logic
import palmerpenguins # Example dataset
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
from datetime import datetime, timedelta # For generating synthetic dates

# Import Prophet for time-series forecasting.
# Note: Prophet is a powerful library for forecasting time series data.
# It requires the data to be in a specific format with columns 'ds' and 'y'.
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the Palmer Penguins dataset
df = palmerpenguins.load_penguins()

# --- NEW: Create a synthetic time-series for forecasting ---
# The original dataset is not a time-series, so we'll simulate one
# by creating daily penguin counts over a two-year period.
def create_synthetic_timeseries():
    """Generates a synthetic time-series of penguin counts."""
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(730)] # 2 years of data
    
    # Simulate a count with a linear trend and some seasonality
    counts = np.arange(len(dates)) * 0.1 + 100 + 20 * np.sin(np.arange(len(dates)) / 30)
    
    # Add some random noise
    counts += np.random.normal(0, 10, size=len(dates))
    
    # Create the DataFrame in the format Prophet expects
    ts_df = pd.DataFrame({
        'ds': dates,
        'y': counts.round().astype(int)
    })
    return ts_df

# Generate the synthetic time-series data once at the start
synthetic_ts_df = create_synthetic_timeseries()

# Set dashboard title and layout options
ui.page_opts(title="Penguins dashboard with Forecasting", fillable=True, favicon="favicon.ico")

# Sidebar: filter controls for the dashboard
with ui.sidebar(title="Filter controls"):
    # Slider to filter by penguin mass
    ui.input_slider("mass", "Mass", 2000, 6000, 6000)
    # Checkbox group to select penguin species
    ui.input_checkbox_group(
        "species",
        "Species",
        ["Adelie", "Gentoo", "Chinstrap"],
        selected=["Adelie", "Gentoo", "Chinstrap"],
    )
    ui.hr()
    
    # --- NEW: Forecast controls ---
    ui.h5("Forecast Controls")
    ui.input_select(
        "model_selection",
        "Select Model",
        {"Prophet": "Prophet"},
        selected="Prophet"
    )
    ui.input_slider(
        "forecast_horizon",
        "Forecast Horizon (days)",
        min=1,
        max=365,
        value=30
    )
    ui.hr()
    # Section for useful links and resources
    ui.h6("Links")
    ui.a(
        "GitHub Source",
        href="https://github.com/denisecase/cintel-07-tdash",
        target="_blank",
    )
    ui.a(
        "GitHub App",
        href="https://denisecase.github.io/cintel-07-tdash/",
        target="_blank",
    )
    ui.a(
        "GitHub Issues",
        href="https://github.com/denisecase/cintel-07-tdash/issues",
        target="_blank",
    )
    ui.a("PyShiny", href="https://shiny.posit.co/py/", target="_blank")
    ui.a(
        "Template: Basic Dashboard",
        href="https://shiny.posit.co/py/templates/dashboard/",
        target="_blank",
    )
    ui.a(
        "See also",
        href="https://github.com/denisecase/pyshiny-penguins-dashboard-express",
        target="_blank",
    )

# Value boxes: summary statistics for filtered data
with ui.layout_column_wrap(fill=False):
    # Shows the number of penguins after filtering
    with ui.value_box(showcase=icon_svg("earlybirds")):
        "Total Penguins in Selection"

        @render.text
        def count():
            return filtered_df().shape[0]

    # Shows the average bill length
    with ui.value_box(showcase=icon_svg("ruler-horizontal")):
        "Average Bill Length (mm)"

        @render.text
        def bill_length():
            return f"{filtered_df()['bill_length_mm'].mean():.1f} mm"

    # Shows the average bill depth
    with ui.value_box(showcase=icon_svg("ruler-vertical")):
        "Average Bill Depth (mm)"

        @render.text
        def bill_depth():
            return f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"

# Main content: plot and interactive data table
with ui.layout_columns():
    # Scatterplot of bill length vs bill depth, colored by species (Seaborn)
    with ui.card(full_screen=True):
        ui.card_header("Bill Length vs Depth by Species")

        @render.plot
        def length_depth():
            import matplotlib.pyplot as plt
            df_plot = filtered_df()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_plot,
                x="bill_length_mm",
                y="bill_depth_mm",
                hue="species",
                ax=ax
            )
            ax.set_title("Bill Length vs Depth by Species")
            ax.set_xlabel("Bill Length (mm)")
            ax.set_ylabel("Bill Depth (mm)")
            return fig

    # Interactive data table of penguin summary statistics
    with ui.card(full_screen=True):
        ui.card_header("Penguin Data")

        @render.data_frame
        def summary_statistics():
            cols = [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "body_mass_g",
            ]
            return render.DataGrid(filtered_df()[cols], filters=True)
            
    # --- NEW: Card for the predictive analytics plot ---
    with ui.card(full_screen=True):
        ui.card_header("Penguin Count Forecast")

        @render.plot
        def forecast_plot():
            # Get the forecast data from the reactive expression
            forecast = forecast_data()

            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the original synthetic data
            ax.plot(synthetic_ts_df['ds'], synthetic_ts_df['y'], label='Historical Data', color='blue')
            
            # Plot the forecast
            ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
            
            # Plot the confidence interval
            ax.fill_between(
                forecast['ds'], 
                forecast['yhat_lower'], 
                forecast['yhat_upper'], 
                color='orange', 
                alpha=0.2, 
                label='Confidence Interval'
            )
            
            # Add labels and title
            ax.set_title(f"Penguin Count Forecast for {input.forecast_horizon()} days")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Penguins")
            ax.legend()
            ax.grid(True)
            
            return fig


# Reactive calculation: filters the DataFrame based on sidebar controls
@reactive.calc
def filtered_df():
    # Filter the DataFrame based on selected species and mass
    filt_df = df[df["species"].isin(input.species())]
    filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
    return filt_df

# --- NEW: Reactive calculation for the forecast ---
@reactive.calc
def forecast_data():
    """Fits a Prophet model and generates a forecast."""
    # The model works with the synthetic_ts_df, not the filtered_df,
    # as the filtering is for a different purpose in this dashboard.
    
    # Initialize the Prophet model
    m = Prophet()
    
    # Fit the model to our synthetic time-series data
    m.fit(synthetic_ts_df)
    
    # Create a future DataFrame to hold the dates for the forecast
    future = m.make_future_dataframe(periods=input.forecast_horizon())
    
    # Make the predictions
    forecast = m.predict(future)
    
    return forecast
