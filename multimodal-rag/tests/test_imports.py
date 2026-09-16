def test_import_config():
    from app.core.config import get_settings

    settings = get_settings()
    assert settings.app_name == "multimodal-rag"
