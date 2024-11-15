from dash import Dash
from dashboard.layout import layout  # Import the layout
import dashboard.callbacks  # Import callbacks to register them

app = Dash(__name__, suppress_callback_exceptions=True)

# Set the app layout
app.layout = layout

if __name__ == "__main__":
    app.run_server(debug=True)
