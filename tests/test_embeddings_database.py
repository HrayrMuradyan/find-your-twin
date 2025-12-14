import pytest
import numpy as np
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from find_your_twin.embeddings_database import DatabaseServiceClient
import logging
from pathlib import Path

def mock_response(status_code, json_data=None):
    """
    Creates a valid httpx.Response object with a dummy Request attached.
    This prevents 'RuntimeError: Cannot call raise_for_status...'
    """
    request = httpx.Request("POST", "http://test-url")
    response = httpx.Response(status_code, json=json_data, request=request)
    return response

@pytest.fixture
def mock_drive_service():
    """Mocks the Google Drive Service Resource."""
    service = MagicMock()
    # Mock the chain: service.files().delete(fileId=...).execute()
    service.files.return_value.delete.return_value.execute.return_value = None
    return service

@pytest.fixture
def mock_env_vars():
    """Sets necessary environment variables."""
    with patch.dict("os.environ", {"DATABASE_SERVICE_URL": "http://test-db.com"}):
        yield

@pytest.fixture
async def client(mock_drive_service, mock_env_vars):
    """
    Creates the client instance with mocked HTTP client and models.
    """
    # Patch httpx.AsyncClient to prevent real network calls
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_http = AsyncMock()
        MockClientClass.return_value = mock_http
        
        # Initialize client
        service = DatabaseServiceClient(
            face_detect_model="dummy_face.onnx",
            embeddings_model="dummy_embed.onnx",
            drive_service=mock_drive_service,
            drive_folder_id="folder_123"
        )
        
        # Manually attach mocks for the lazy-loaded models
        # so we don't need to patch load_model logic every time
        service.face_detector = MagicMock()
        service.embedder = MagicMock()
        
        yield service
        
        await service.close()

@pytest.fixture
def run_in_threadpool_passthrough():
    """
    Patches run_in_threadpool to immediately execute the function.
    This allows us to test the logic INSIDE the threadpool wrappers
    (like face detection and drive uploading).
    """
    async def side_effect(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("find_your_twin.embeddings_database.run_in_threadpool", side_effect=side_effect) as mock:
        yield mock


def test_init_raises_without_env(mock_drive_service):
    """Should raise ValueError if no URL config is present."""
    # Ensure environment is empty for this test
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="must set either DATABASE_SERVICE_URL"):
            DatabaseServiceClient(drive_service=mock_drive_service)

@pytest.mark.asyncio
async def test_get_index_count(client):
    """Test health check parsing."""
    # Setup mock response using helper
    client.http_client.get.return_value = mock_response(200, json_data={"count": 42})
    
    result = await client.get_index_count()
    assert result == {"count": 42}
    
@pytest.mark.asyncio
async def test_get_index_count_failure(client):
    """Should return count 0 on error."""
    client.http_client.get.side_effect = httpx.RequestError("Boom")
    
    result = await client.get_index_count()
    assert result == {"count": 0}

@pytest.mark.asyncio
async def test_search_image_success(client, run_in_threadpool_passthrough):
    """
    Test the full flow: Read -> Detect -> Embed -> Search.
    """
    # Mock External Helpers
    with patch("find_your_twin.embeddings_database.read_image") as mock_read:
        
        # Setup Helper Mocks
        mock_read.return_value = np.zeros((100, 100, 3)) 
        client.face_detector.detect.return_value = "face_crop" 
        client.embedder.compute_embeddings.return_value = np.array([0.1, 0.2]) 
        
        # Setup DB Response using helper
        client.http_client.post.return_value = mock_response(200, json_data={"results": ["uuid1", "uuid2"]})

        # Run
        results = await client.search_image("test.jpg")

        # Verify
        assert results == ["uuid1", "uuid2"]
        client.face_detector.detect.assert_called()
        client.embedder.compute_embeddings.assert_called()
        
        # Verify JSON payload sent to DB
        client.http_client.post.assert_called_once()
        call_kwargs = client.http_client.post.call_args[1]
        assert call_kwargs['json']['vector'] == [0.1, 0.2]

@pytest.mark.asyncio
async def test_search_image_success_with_numpy(client, run_in_threadpool_passthrough):
    """
    Test the full flow: Numpy Image -> Detect -> Embed -> Search.
    """
        
    # Setup Helper Mocks
    image = np.zeros((100, 100, 3)) 
    client.face_detector.detect.return_value = "face_crop" 
    client.embedder.compute_embeddings.return_value = np.array([0.1, 0.2]) 
    
    # Setup DB Response using helper
    client.http_client.post.return_value = mock_response(200, json_data={"results": ["uuid1", "uuid2"]})

    # Run
    results = await client.search_image(image)

    # Verify
    assert results == ["uuid1", "uuid2"]
    client.face_detector.detect.assert_called()
    client.embedder.compute_embeddings.assert_called()
    
    # Verify JSON payload sent to DB
    client.http_client.post.assert_called_once()
    call_kwargs = client.http_client.post.call_args[1]
    assert call_kwargs['json']['vector'] == [0.1, 0.2]

@pytest.mark.asyncio
async def test_search_image_generic_failure_logs_and_raises(
    client, 
    run_in_threadpool_passthrough, 
    caplog
):
    """
    Verifies that unexpected exceptions (not ValueErrors) are caught, 
    logged, and wrapped in a "Failed to process image" ValueError.
    """
    # Use caplog to capture log messages at ERROR level
    caplog.set_level(logging.ERROR)

    with patch("find_your_twin.embeddings_database.read_image"):
        # Simulate a generic, unexpected crash inside the inference logic
        client.face_detector.detect.side_effect = RuntimeError("Unexpected Model Crash")

        # Assert that it raises ValueError (wrapping the original error)
        with pytest.raises(ValueError, match="Failed to process image"):
            await client.search_image("test.jpg")

    # Assert that the error was actually logged
    assert "Inference failed with the following error message" in caplog.text
    assert "Unexpected Model Crash" in caplog.text

@pytest.mark.asyncio
async def test_search_image_http_error_logs_and_reraises(
    client, 
    run_in_threadpool_passthrough, 
    caplog
):
    """
    Verifies that if the external Search Service returns an HTTP error (e.g. 500),
    we log it with a specific message and re-raise the exception.
    """
    caplog.set_level(logging.ERROR)

    # Setup Inference Success (so we actually reach the HTTP call)
    with patch("find_your_twin.embeddings_database.read_image"):
        client.face_detector.detect.return_value = "face"
        client.embedder.compute_embeddings.return_value = np.array([0.1])

        # Simulate the Search Service failing (e.g., 500 Internal Server Error)
        mock_req = httpx.Request("POST", "http://test-url/search")
        mock_resp = httpx.Response(500, request=mock_req)
        
        # Configure the client to raise this error when .post() is called
        client.http_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mock_req, response=mock_resp
        )

        # Assert that the exception is re-raised
        with pytest.raises(httpx.HTTPStatusError):
            await client.search_image("test.jpg")

    # Verify the logging
    assert "Search Service Error" in caplog.text
    assert "Server Error" in caplog.text

@pytest.mark.asyncio
async def test_search_image_no_face(client, run_in_threadpool_passthrough):
    """Should raise ValueError if face detection fails."""
    with patch("find_your_twin.embeddings_database.read_image"):
        # Simulate no face found
        client.face_detector.detect.return_value = None
        
        with pytest.raises(ValueError, match="No face detected"):
            await client.search_image("ghost.jpg")

@pytest.mark.asyncio
async def test_add_image_raises_without_drive_config(mock_env_vars):
    """
    Verifies that add_image raises ValueError if Google Drive is not configured.
    """
    # Initialize a client WITHOUT a drive_service or folder_id
    client_no_drive = DatabaseServiceClient(
        face_detect_model="dummy.onnx",
        embeddings_model="dummy.onnx",
        drive_service=None,       
        drive_folder_id=None     
    )

    # Mock _ensure_models_loaded
    client_no_drive._ensure_models_loaded = MagicMock()

    # Call add_image and expect it to fail
    with pytest.raises(ValueError, match="Google Drive not configured"):
        await client_no_drive.add_image("test.jpg", metadata={})
        
    await client_no_drive.close()

@pytest.mark.asyncio
async def test_add_image_no_face_detected(client, run_in_threadpool_passthrough):
    """
    Verifies that add_image raises ValueError if the face detector returns None.
    """
    with patch("find_your_twin.embeddings_database.read_image") as mock_read, \
         patch("find_your_twin.embeddings_database.upload_bytes_to_folder") as mock_upload:

        # Setup mocks
        mock_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Simulate Face Detector returning 'None' (No face found)
        client.face_detector.detect.return_value = None 

        # Call add_image and expect failure
        with pytest.raises(ValueError, match="No face detected"):
            await client.add_image("landscape.jpg", metadata={})
        
        # Verify we STOPPED immediately
        client.embedder.compute_embeddings.assert_not_called()
        # We should NOT have uploaded anything to Drive
        mock_upload.assert_not_called()

@pytest.mark.asyncio
async def test_add_image_drive_upload_returns_none(client, run_in_threadpool_passthrough):
    """
    Verifies that add_image raises ValueError if the Drive upload returns None.
    """
    with patch("find_your_twin.embeddings_database.read_image") as mock_read, \
         patch("find_your_twin.embeddings_database.upload_bytes_to_folder") as mock_upload:

        mock_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        client.face_detector.detect.return_value = "face"
        client.embedder.compute_embeddings.return_value = np.array([0.1])
        
        # Simulate Drive Upload FAILING (returning None)
        mock_upload.return_value = None

        # Call add_image and expect failure
        with pytest.raises(ValueError, match="Drive upload failed"):
            await client.add_image("test.jpg", metadata={})
        
        # Verify we aborted BEFORE calling the database
        mock_upload.assert_called_once()
        # We never called the DB because upload failed
        client.http_client.post.assert_not_called()

@pytest.mark.asyncio
async def test_add_image_rollback_failure_logs_critical(
    client, 
    run_in_threadpool_passthrough, 
    mock_drive_service, 
    caplog
):
    """
    Verifies that if the rollback (file deletion) ITSELF fails, 
    we catch it, log a CRITICAL error, and still re-raise the original DB error.
    """
    # Capture only CRITICAL logs to keep things clean
    caplog.set_level(logging.CRITICAL)

    with patch("find_your_twin.embeddings_database.read_image") as mock_read, \
         patch("find_your_twin.embeddings_database.upload_bytes_to_folder") as mock_upload:

        # Setup Success for everything BEFORE the DB call
        mock_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        client.face_detector.detect.return_value = "face"
        client.embedder.compute_embeddings.return_value = np.array([0.1])
        mock_upload.return_value = "file_id_to_delete"

        # Make DB fail
        client.http_client.post.side_effect = httpx.HTTPError("Primary DB Failure")

        # Make Drive Delete fail
        mock_drive_service.files.return_value.delete.return_value.execute.side_effect = Exception("Secondary Drive Failure")

        # Run and expect the ORIGINAL error (DB Failure)
        with pytest.raises(httpx.HTTPError, match="Primary DB Failure"):
            await client.add_image("test.jpg", {})

    # Verify the CRITICAL log exists
    assert "Could not delete file_id_to_delete during rollback" in caplog.text
    assert "Manual cleanup required" in caplog.text
    assert "Secondary Drive Failure" in caplog.text


@pytest.mark.asyncio
async def test_delete_image_returns_false_without_drive_service(mock_env_vars):
    """
    Verifies that delete_image_by_uuid returns False immediately if 
    drive_service is None (Drive not configured).
    """
    # Initialize client with drive_service=None
    client = DatabaseServiceClient(
        face_detect_model="dummy.onnx",
        embeddings_model="dummy.onnx",
        drive_service=None,  # <--- The condition we are testing
        drive_folder_id="folder_123"
    )
    
    # Mock the HTTP client just to be safe (though it shouldn't be called)
    client.http_client = AsyncMock()

    try:
        # Attempt to delete
        result = await client.delete_image_by_uuid("some-uuid")

        # Assertions
        assert result is False
        
        # Verify that the database (Access Layer) should NOT be touched
        client.http_client.post.assert_not_called()
        
    finally:
        await client.close()

@pytest.mark.asyncio
async def test_delete_image_drive_failure_logs_critical_but_returns_true(
    client, 
    run_in_threadpool_passthrough, 
    mock_drive_service, 
    caplog
):
    """
    Verifies that if the DB delete succeeds but the Drive delete fails,
    we return True (success for user) but log a CRITICAL error (action for admin).
    """
    caplog.set_level(logging.CRITICAL)

    with patch("find_your_twin.embeddings_database.get_file_by_uuid") as mock_lookup:
        # File exists in Drive
        mock_lookup.return_value = {"id": "orphan_file_id"}
        
        # Database Delete SUCCEEDS
        client.http_client.post.return_value = mock_response(200)

        # Drive Delete FAILS
        # This mimics a Google Drive API outage or permission error.
        mock_drive_service.files.return_value.delete.return_value.execute.side_effect = Exception("Google Drive Down")

        # Run the delete function
        result = await client.delete_image_by_uuid("some-uuid")
        
        # Return TRUE
        assert result is True

    # CRITICAL Log
    # We must have a log telling us to delete 'orphan_file_id' manually.
    assert "Database deleted, but Drive delete failed for orphan_file_id" in caplog.text
    assert "Google Drive Down" in caplog.text

@pytest.mark.asyncio
async def test_add_image_success(client, run_in_threadpool_passthrough):
    """
    Test adding an image: Detect -> Drive Upload -> DB Insert.
    """
    with patch("find_your_twin.embeddings_database.read_image") as mock_read, \
         patch("find_your_twin.embeddings_database.upload_bytes_to_folder") as mock_upload:

        # Setup Mocks
        mock_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        client.face_detector.detect.return_value = "face_crop"
        client.embedder.compute_embeddings.return_value = np.array([0.5, 0.5])
        
        # Mock Drive Upload returning a File ID
        mock_upload.return_value = "drive_file_id_123"
        
        # Mock DB Success using helper
        client.http_client.post.return_value = mock_response(200, json_data={"status": "ok"})

        # Run
        uuid_str = await client.add_image("me.jpg", metadata={"name": "Hrayr"})
        
        # Verify
        assert uuid_str is not None
        mock_upload.assert_called_once()
        
        # Verify payload to DB
        call_payload = client.http_client.post.call_args[1]['json']
        assert call_payload['drive_file_id'] == "drive_file_id_123"
        assert call_payload['source'] == "Hrayr"

@pytest.mark.asyncio
async def test_add_image_rollback_logic(client, run_in_threadpool_passthrough, mock_drive_service):
    """
    CRITICAL TEST: If DB fails, we must delete the file from Drive.
    """
    with patch("find_your_twin.embeddings_database.read_image") as mock_read, \
         patch("find_your_twin.embeddings_database.upload_bytes_to_folder") as mock_upload:

        # Setup Success for everything BEFORE the DB call
        mock_read.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        client.face_detector.detect.return_value = "face"
        client.embedder.compute_embeddings.return_value = np.array([0.1])
        mock_upload.return_value = "uploaded_file_id_999"

        # Setup FAILURE for the DB call
        client.http_client.post.side_effect = httpx.HTTPError("Database Dead")

        # Run and Expect Error
        with pytest.raises(httpx.HTTPError):
            await client.add_image("fail.jpg", {})

        # ASSERT ROLLBACK
        mock_drive_service.files.return_value.delete.assert_called_with(fileId="uploaded_file_id_999")
        mock_drive_service.files.return_value.delete.return_value.execute.assert_called_once()

@pytest.mark.asyncio
async def test_delete_image_success(client, run_in_threadpool_passthrough, mock_drive_service):
    """Test delete flow: Lookup -> DB Delete -> Drive Delete."""
    
    with patch("find_your_twin.embeddings_database.get_file_by_uuid") as mock_lookup:
        # Mock finding the file
        mock_lookup.return_value = {"id": "drive_id_555"}
        
        # Mock DB Success using helper
        client.http_client.post.return_value = mock_response(200)

        # Run
        result = await client.delete_image_by_uuid("some-uuid")
        
        assert result is True
        
        # Verify DB delete called
        client.http_client.post.assert_called_with("/delete", json={"drive_file_id": "drive_id_555"})
        
        # Verify Drive delete called
        mock_drive_service.files.return_value.delete.assert_called_with(fileId="drive_id_555")

@pytest.mark.asyncio
async def test_delete_image_not_found(client, run_in_threadpool_passthrough):
    """If file not in Drive, abort early."""
    with patch("find_your_twin.embeddings_database.get_file_by_uuid") as mock_lookup:
        mock_lookup.return_value = None
        
        result = await client.delete_image_by_uuid("ghost-uuid")
        
        assert result is False
        client.http_client.post.assert_not_called()

@pytest.mark.asyncio
async def test_delete_image_db_failure_aborts_drive(client, run_in_threadpool_passthrough, mock_drive_service):
    """If DB delete fails, we should NOT delete from Drive (preserve data)."""
    with patch("find_your_twin.embeddings_database.get_file_by_uuid") as mock_lookup:
        mock_lookup.return_value = {"id": "drive_id_777"}
        
        # DB Fails
        client.http_client.post.side_effect = httpx.HTTPError("DB Error")

        # Run
        result = await client.delete_image_by_uuid("uuid")
        
        assert result is False
        
        # Assert Drive delete was NEVER called
        mock_drive_service.files.return_value.delete.assert_not_called()


def test_ensure_models_loaded_triggers_loading_process(client):
    """
    Verifies that if models are None, the full loading pipeline runs:
    Validate -> Read Config -> Load Model.
    """
    # Reset the client state
    client.face_detector = None
    client.embedder = None
    client.face_detect_model_config = None
    client.embeddings_model_config = None
    
    # Set dummy paths so validation has something to check
    client.face_detect_model_path = Path("dummy_face")
    client.embeddings_model_path = Path("dummy_embed")

    # Patch the 3 internal helper functions
    with patch("find_your_twin.embeddings_database.validate_model") as mock_validate, \
         patch("find_your_twin.embeddings_database.read_model_config") as mock_read_config, \
         patch("find_your_twin.embeddings_database.load_model") as mock_load_model:

        # Setup return values for the sequence
        mock_read_config.side_effect = ["face_config_obj", "embed_config_obj"]
        mock_load_model.side_effect = ["real_face_model", "real_embed_model"]

        client._ensure_models_loaded()

        # Assertions

        # Should be called for both paths
        assert mock_validate.call_count == 2
        mock_validate.assert_any_call(Path("dummy_face"))
        mock_validate.assert_any_call(Path("dummy_embed"))

        # Should update self.config attributes
        assert client.face_detect_model_config == "face_config_obj"
        assert client.embeddings_model_config == "embed_config_obj"

        # Should be called with the new configs
        mock_load_model.assert_any_call("face_config_obj")
        mock_load_model.assert_any_call("embed_config_obj")

        # Client attributes should hold the loaded models
        assert client.face_detector == "real_face_model"
        assert client.embedder == "real_embed_model"