from model_utils import ModelManager
import os

def test_model_responds():
    mgr = ModelManager(
        storage_account="miaamlopsresources",
        container=os.getenv("AZURE_CONTAINER_NAME"),
        model_blob=os.getenv("AZURE_MODEL_BLOB"),
        log_blob="test_predictions.txt",
        conn_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )

    mgr.ensure_model()

    sample_input = [0.1, 0.2, 0.3, 0.4]
    result = mgr.predict(sample_input)

    assert result is not None
    assert isinstance(result, list)

