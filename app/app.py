
"""
Penguins Dashboard Example using Shiny for Python
This app demonstrates interactive filtering, summary statistics, and data visualization
using the Palmer Penguins dataset. Comments are provided to help new team members learn Shiny for Python.
"""
import seaborn as sns  # For data visualization
from faicons import icon_svg  # For icon support in UI
import plotly.express as px  # For interactive charts
from shiny import reactive  # For reactive programming
from shiny.express import input, render, ui  # Shiny for Python UI and server logic
import palmerpenguins  # Example dataset

# Load the Palmer Penguins dataset
df = palmerpenguins.load_penguins()

# Set dashboard title and layout options
ui.page_opts(title="Penguins dashboard", fillable=True)

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
    # Scatterplot of bill length vs bill depth, colored by species
    with ui.card(full_screen=True):
        ui.card_header("Bill Length vs Depth by Species")

        @render.plot
        def length_depth():
            # Create an interactive scatterplot using Plotly
            df_plot = filtered_df()
            fig = px.scatter(
                df_plot,
                x="bill_length_mm",
                y="bill_depth_mm",
                color="species",
                title="Bill Length vs Depth by Species",
                labels={
                    "bill_length_mm": "Bill Length (mm)",
                    "bill_depth_mm": "Bill Depth (mm)",
                    "species": "Species"
                },
                hover_data=["island", "body_mass_g"]
            )
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

# Example of including custom CSS for further UI customization
#ui.include_css(app_dir / "styles.css")

# Reactive calculation: filters the DataFrame based on sidebar controls
@reactive.calc
def filtered_df():
    # Filter the DataFrame based on selected species and mass
    filt_df = df[df["species"].isin(input.species())]
    filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
    return filt_df
    # Filter the DataFrame based on selected species and mass
