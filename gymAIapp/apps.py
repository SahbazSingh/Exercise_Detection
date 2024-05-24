from django.apps import AppConfig


class GymaiappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gymAIapp"

    def ready(self):
        # Import the function here to avoid circular imports
        from .models_ml.main import load_machine_learning_models
        load_machine_learning_models()
