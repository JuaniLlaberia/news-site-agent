import os
import io
import logging
from urllib.parse import urlparse
from flask import Blueprint, jsonify, request, send_file

from src.agents.orchestrator.orchestrator import Orchestrator
from src.agents.reporter.reporter import Reporter
from src.utils.validators.observations_validator import validate_observations
from src.utils.validators.site_config_validator import validate_site_config
from src.agents.orchestrator.models.content import Observation

agent_bp = Blueprint("agent_blueprint", __name__)

@agent_bp.route("/", methods=["POST"])
def agent_endpoint():
    """
    Endpoint to scan and analyze news site. It returns a configuration and all the observations made in the process

    Request JSON Structure:
        {
            "name": "<string>",
            "url": "<string>"
        }

    Success Response JSON Structure (200):
        {
            "config": {
                "site_name": "<string>",
                "url": "https://...",
                "url_dict": { "home": "/", "articles": "/articles" },
                "title_selector": "<css selector>",
                "content_selector": "<css selector>",
                "subtitle_selector": "<css selector>",
                "author_selector": "<css selector>",
                "img_url_selector": "<css selector>",
                "date_selector": "<css selector>",
                "rate_limit": 1
            },
            "observations": [
                {
                    "type": "info|warning|error|success",
                    "message": "<string>",
                    "agent_type": "orchestrator|web_inspector|tester|rate_limit_tester"
                }
            ]
        }

    Error responses (examples):
        { "error": "Request must be in JSON format" }
        { "error": "Request must have 'name' (site name) and 'url' (site base url)" }
        { "error": "Invalid URL" }
        { "error": "URL must be using `https`" }
        { "error": "Validation error: <details>" }
        { "error": "Server error: <details>" }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        news_site_JSON = request.get_json()
        if "name" not in news_site_JSON or "url" not in news_site_JSON:
            return jsonify({"error": "Request must have 'name' (site name) and 'url' (site base url)"}), 400

        parsed_url = urlparse(news_site_JSON["url"])
        if not parsed_url.scheme or not parsed_url.netloc or not parsed_url.hostname:
            return jsonify({"error": "Invalid URL"}), 400

        if parsed_url.scheme != "https":
            return jsonify({"error": "URL must be using `https`"}), 400

        valid_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        orchestrator = Orchestrator()
        news_site_config, observations = orchestrator.run(
            site_name=news_site_JSON["name"],
            url=valid_url
        )

        # Serilize the observation to be valid for JSON
        serilized_observations = [obvservation.model_dump() for obvservation in observations]

        return jsonify({
            "config": news_site_config,
            "observations": serilized_observations
        }), 200

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({"error": f"Validation error: {e}"}), 400
    except Exception as e:
        print(e)
        logging.error(f"Unexpected error when running the agent: {e}")
        return jsonify({"error": f"Server error: {e}"}), 400

@agent_bp.route("/generate_report", methods=["POST"])
def report_agent_endpoint():
    """
    Endpoint to scan news sites and generate a report containing the needed html structure and the gathered information

    Request JSON Structure:
        {
            "config": {
                "site_name": "<string>",
                "url": "https://...",
                "url_dict": { "home": "/", "articles": "/articles" },
                "title_selector": "<css selector>",
                "content_selector": "<css selector>",
                "subtitle_selector": "<css selector>",
                "author_selector": "<css selector>",
                "img_url_selector": "<css selector>",
                "date_selector": "<css selector>",
                "rate_limit": 1
            },
            "observations": [
                {
                    "type": "info|warning|error|success",
                    "message": "<string>",
                    "agent_type": "orchestrator|web_inspector|tester|rate_limit_tester"
                }
            ]
        }

    Success Response (200):
        - Returns PDF binary stream (Content-Type: application/pdf)
        - Attachment filename: "report.pdf"

    Error responses (examples):
        { "error": "Request must be in JSON format" }
        { "error": "Request must have 'config' (site configuration) and 'observations' (analysis observations)" }
        { "error": "Failed to validate site configuration", "details": <validation_details> }
        { "error": "Some observations are missing data", "details": <validation_details> }
        { "error": "Validation error: <details>" }
        { "error": "Server error: <details>" }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        news_site_JSON = request.get_json()
        if "config" not in news_site_JSON or "observations" not in news_site_JSON:
            return jsonify({"error": "Request must have 'config' (site configuration) and 'observations' (analysis observations)"}), 400

        # Validate that all required fields are inside the configuration (with a value)
        isValid, error = validate_site_config(site_config=news_site_JSON["config"])
        if not isValid:
            return jsonify({"error": "Failed to validate site configuration", "details": error}), 500

        # Validate that all observations have all required fields
        isValid, error = validate_observations(observations=news_site_JSON["observations"])
        if not isValid:
            return jsonify({"error": "Some observations are missing data", "details": error}), 500

        observations = [Observation(**a) for a in news_site_JSON["observations"]]

        reporter = Reporter(ollama_model=os.getenv("OLLAMA_MODEL"),
                            ollama_base_url=os.getenv("OLLAMA_BASE_URL"))
        report_bytes = reporter.run(site_name=news_site_JSON["config"]["site_name"],
                                    url=news_site_JSON["config"]["url"],
                                    url_dict=news_site_JSON["config"]["url_dict"],
                                    title_selector=news_site_JSON["config"]["title_selector"],
                                    content_selector=news_site_JSON["config"]["content_selector"],
                                    subtitle_selector=news_site_JSON["config"]["subtitle_selector"],
                                    author_selector=news_site_JSON["config"]["author_selector"],
                                    img_url_selector=news_site_JSON["config"]["img_url_selector"],
                                    date_selector=news_site_JSON["config"]["date_selector"],
                                    rate_limit=news_site_JSON["config"]["rate_limit"],
                                    observations=observations)

        return send_file(
            io.BytesIO(report_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="report.pdf"
        ), 200

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({"error": f"Validation error: {e}"}), 400
    except Exception as e:
        print(e)
        logging.error(f"Unexpected error when running the reporter agent: {e}")
        return jsonify({"error": f"Server error: {e}"}), 400
