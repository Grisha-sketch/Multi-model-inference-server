from app.core.model_loader import load_model


class ModelRegistry:
    """
    Stores all registered models in memory.
    Supports multiple models, multiple versions, and hot-swapping.
    """

    def __init__(self):
        # Structure: { "model_name:version": { "model": <loaded>, "file_path": "..." } }
        self._registry: dict = {}
        self._active: dict = {}  # { "model_name": "version" }

    def register(self, model_name: str, version: str, file_path: str):
        """Load model from disk and store it in the registry."""
        key = f"{model_name}:{version}"

        if key in self._registry:
            raise ValueError(f"Model '{key}' is already registered.")

        loaded_model = load_model(file_path)

        self._registry[key] = {
            "model": loaded_model,
            "file_path": file_path
        }

        # Auto-activate if it's the first version of this model
        if model_name not in self._active:
            self._active[model_name] = version

        print(f"[Registry] Registered model: {key}")

    def activate(self, model_name: str, version: str):
        """Switch the active version of a model (hot-swap, no restart needed)."""
        key = f"{model_name}:{version}"

        if key not in self._registry:
            raise ValueError(f"Model '{key}' not found. Register it first.")

        self._active[model_name] = version
        print(f"[Registry] Activated: {key}")

    def get_active_model(self, model_name: str):
        """Return the currently active loaded model for a given model name."""
        if model_name not in self._active:
            raise ValueError(f"No active version found for model '{model_name}'.")

        version = self._active[model_name]
        key = f"{model_name}:{version}"
        return self._registry[key]["model"], version

    def get_model(self, model_name: str, version: str):
        """Return a specific version of a model."""
        key = f"{model_name}:{version}"

        if key not in self._registry:
            raise ValueError(f"Model '{key}' not found.")

        return self._registry[key]["model"]

    def list_models(self):
        """Return a list of all registered models with their active status."""
        result = []
        for key, value in self._registry.items():
            model_name, version = key.split(":", 1)
            is_active = self._active.get(model_name) == version
            result.append({
                "model_name": model_name,
                "version": version,
                "file_path": value["file_path"],
                "is_active": is_active
            })
        return result


# Single shared instance used across the entire app
registry = ModelRegistry()