from .config import Settings


def create_app(settings: Settings | None = None):
    """Create and configure the Flask application."""
    from flask import Flask

    from .rag.service import AiEngineeringRagService
    from .routes import main_bp

    resolved_settings = settings or Settings.from_env()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = resolved_settings.secret_key
    app.config["SETTINGS"] = resolved_settings
    app.rag_service = AiEngineeringRagService(resolved_settings)  # type: ignore[attr-defined]
    app.register_blueprint(main_bp)

    return app
