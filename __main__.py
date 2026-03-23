import logging
from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint

from src.routes.agent import agent_bp
from src.routes.health import health_bp

logging.basicConfig(filename='news_sites_agent.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"
swagger_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "News Site Analyzer Agent"
    }
)

def create_app() -> Flask:
    """
    Function to create the Flask instance

    Returns:
        Flask: Flask server/app
    """
    app = Flask(__name__)
    app.register_blueprint(agent_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(swagger_blueprint, url_prefix=SWAGGER_URL)

    return app

def main():
    """
    Main entry point for the Flask News Sites Agentic Flow API.
    Starts the web server to gather news sites via HTTP POST requests.
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()