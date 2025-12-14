import pytest
from unittest.mock import MagicMock, patch, mock_open
from googleapiclient.errors import HttpError
from find_your_twin.google_drive import (
    get_drive_service,
    list_all_files,
    get_or_create_app_folder,
    upload_file_to_folder,
    upload_bytes_to_folder,
    get_image_bytes_by_id,
    get_image_base64_by_id,
    get_file_by_uuid,
    delete_file_by_uuid,
    delete_file_by_id
)

def create_http_error(status):
    """Helper to create HttpError with specific status code."""
    resp = MagicMock()
    resp.status = status
    return HttpError(resp, b"Mock Error")

# --- Service & Auth Tests ---

@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.build")
@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
def test_get_drive_service_valid_token(mock_token_path, mock_build, mock_creds):
    """Verifies that a service is returned immediately if a valid token file exists."""
    mock_token_path.exists.return_value = True
    mock_creds.from_authorized_user_file.return_value.valid = True
    assert get_drive_service() is not None

@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.build")
@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
def test_get_drive_service_refresh_success(mock_token_path, mock_build, mock_creds):
    """Verifies that an expired but refreshable token is refreshed and saved."""
    mock_token_path.exists.return_value = True
    creds = mock_creds.from_authorized_user_file.return_value
    creds.valid = False
    creds.expired = True
    creds.refresh_token = True
    
    with patch("find_your_twin.google_drive.open", mock_open()):
        get_drive_service()
        creds.refresh.assert_called_once()

@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
def test_get_drive_service_refresh_failure_deletes_token(mock_token_path, mock_creds):
    """Verifies that if token refresh fails, the invalid token file is deleted."""
    mock_token_path.exists.return_value = True
    creds = mock_creds.from_authorized_user_file.return_value
    creds.valid = False
    creds.expired = True
    creds.refresh_token = True
    creds.refresh.side_effect = Exception("Refresh failed")

    with patch("find_your_twin.google_drive.CLIENT_SECRET_FILE_PATH") as m_secret:
        m_secret.exists.return_value = False 
        get_drive_service()
        mock_token_path.unlink.assert_called_once()

@patch("find_your_twin.google_drive.InstalledAppFlow")
@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.build")
@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
@patch("find_your_twin.google_drive.CLIENT_SECRET_FILE_PATH")
def test_get_drive_service_new_auth_flow(mock_secret, mock_token, mock_build, mock_creds, mock_flow):
    """Verifies that the full OAuth flow runs when no token exists but client secret does."""
    mock_token.exists.return_value = False
    mock_creds.from_authorized_user_file.return_value = None
    mock_secret.exists.return_value = True
    
    with patch("find_your_twin.google_drive.open", mock_open()):
        get_drive_service()
        mock_flow.from_client_secrets_file.assert_called_once()

@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
@patch("find_your_twin.google_drive.CLIENT_SECRET_FILE_PATH")
def test_get_drive_service_missing_secret(mock_secret, mock_token):
    """Verifies that the function returns None and logs error if client secret is missing."""
    mock_token.exists.return_value = False
    mock_secret.exists.return_value = False
    assert get_drive_service() is None

@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.build")
def test_get_drive_service_build_http_error(mock_build, mock_creds, mock_token):
    """Verifies HttpError handling if the service build step fails."""
    mock_token.exists.return_value = True
    mock_creds.from_authorized_user_file.return_value.valid = True
    mock_build.side_effect = create_http_error(500)
    assert get_drive_service() is None

@patch("find_your_twin.google_drive.TOKEN_FILE_PATH")
@patch("find_your_twin.google_drive.Credentials")
@patch("find_your_twin.google_drive.build")
def test_get_drive_service_generic_error(mock_build, mock_creds, mock_token):
    """Verifies generic exception handling if the service build step crashes."""
    mock_token.exists.return_value = True
    mock_creds.from_authorized_user_file.return_value.valid = True
    mock_build.side_effect = Exception("Unexpected Crash")
    assert get_drive_service() is None

# --- List Files Tests ---

def test_list_all_files_missing_service():
    """Verifies that an empty list is returned if service is None."""
    assert list_all_files(None) == []

def test_list_all_files_pagination():
    """Verifies that the function iterates through pages until no page token remains."""
    service = MagicMock()
    mock_list = service.files.return_value.list
    mock_list.return_value.execute.side_effect = [
        {"files": ["a"], "nextPageToken": "page2"},
        {"files": ["b"]} 
    ]
    files = list_all_files(service)
    assert len(files) == 2

def test_list_all_files_error():
    """Verifies exception handling during file listing."""
    service = MagicMock()
    service.files.return_value.list.side_effect = Exception("API Error")
    assert list_all_files(service) == []

# --- Folder Creation Tests ---

def test_get_or_create_app_folder_found():
    """Verifies that existing folder ID is returned if found."""
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": [{"id": "existing_id"}]}
    assert get_or_create_app_folder(service) == "existing_id"

def test_get_or_create_app_folder_created():
    """Verifies that a new folder is created if search returns empty."""
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": []}
    service.files.return_value.create.return_value.execute.return_value = {"id": "new_id"}
    assert get_or_create_app_folder(service) == "new_id"

def test_get_or_create_app_folder_error():
    """Verifies exception handling during folder search or creation."""
    service = MagicMock()
    service.files.return_value.list.side_effect = create_http_error(500)
    assert get_or_create_app_folder(service) is None

# --- Upload Tests ---

def test_upload_file_to_folder_invalid_type():
    """Verifies that a TypeError is raised for invalid path arguments."""
    with pytest.raises(TypeError):
        upload_file_to_folder(None, "id", 123)

def test_upload_file_to_folder_missing_args():
    """Verifies early return if service or folder_id is missing."""
    assert upload_file_to_folder(None, "id", "path") is None

def test_upload_file_to_folder_not_found(tmp_path):
    """Verifies early return if the local file does not exist."""
    assert upload_file_to_folder(MagicMock(), "id", tmp_path / "ghost.txt") is None

def test_upload_file_to_folder_success_no_uuid(tmp_path):
    """Verifies successful upload without UUID."""
    f = tmp_path / "test.txt"
    f.touch()
    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "fid", "name": "test.txt"}
    assert upload_file_to_folder(service, "id", f) == "fid"

def test_upload_file_to_folder_success_with_uuid(tmp_path):
    """Verifies successful upload WITH UUID triggers appProperties logic."""
    f = tmp_path / "test.txt"
    f.touch()
    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "fid", "name": "test.txt"}
    
    upload_file_to_folder(service, "id", f, uuid="my-uuid")
    
    # Check that appProperties was passed in the body
    call_args = service.files.return_value.create.call_args
    assert call_args[1]['body']['appProperties']['uuid'] == "my-uuid"

def test_upload_file_to_folder_error(tmp_path):
    """Verifies exception handling during file upload."""
    f = tmp_path / "test.txt"
    f.touch()
    service = MagicMock()
    service.files.return_value.create.side_effect = create_http_error(500)
    assert upload_file_to_folder(service, "id", f) is None

def test_upload_bytes_to_folder_success():
    """Verifies successful bytes upload returns the file ID."""
    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "fid", "name": "bytes_file"}
    assert upload_bytes_to_folder(service, "id", "name", b"data", "type", "uuid") == "fid"

def test_upload_bytes_to_folder_error():
    """Verifies exception handling during bytes upload."""
    service = MagicMock()
    service.files.return_value.create.side_effect = create_http_error(500)
    assert upload_bytes_to_folder(service, "id", "name", b"data", "type") is None

# --- Download Tests ---

def test_get_image_bytes_by_id_success():
    """Verifies that bytes are returned from the downloader buffer."""
    service = MagicMock()
    
    mock_downloader = MagicMock()
    mock_downloader.next_chunk.return_value = (MagicMock(), True)
    
    with patch("find_your_twin.google_drive.MediaIoBaseDownload", return_value=mock_downloader):
        with patch("find_your_twin.google_drive.io.BytesIO") as mock_io:
            mock_io.return_value.getvalue.return_value = b"img_data"
            assert get_image_bytes_by_id(service, "fid") == b"img_data"

def test_get_image_bytes_by_id_http_errors():
    """Verifies handling of 404 and 403 errors during download."""
    service = MagicMock()
    
    # Test 404
    service.files.return_value.get_media.side_effect = create_http_error(404)
    assert get_image_bytes_by_id(service, "fid") is None
    
    # Test 403
    service.files.return_value.get_media.side_effect = create_http_error(403)
    assert get_image_bytes_by_id(service, "fid") is None

def test_get_image_bytes_by_id_generic_exception():
    """Verifies handling of unexpected exceptions during download."""
    service = MagicMock()
    service.files.return_value.get_media.side_effect = Exception("Unexpected Crash")
    assert get_image_bytes_by_id(service, "fid") is None

def test_get_image_base64_by_id_success():
    """Verifies that bytes are correctly encoded to a data URI string."""
    with patch("find_your_twin.google_drive.get_image_bytes_by_id", return_value=b"abc"):
        result = get_image_base64_by_id(MagicMock(), "fid")
        assert result.startswith("data:image/jpeg;base64,")

def test_get_image_base64_by_id_failure():
    """Verifies that None is returned if byte download fails."""
    with patch("find_your_twin.google_drive.get_image_bytes_by_id", return_value=None):
        assert get_image_base64_by_id(MagicMock(), "fid") is None

# --- Lookup Tests ---

def test_get_file_by_uuid_found():
    """Verifies that the first matching file is returned if found."""
    service = MagicMock()
    # Corrected: Added 'name' so logging doesn't crash
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "1", "name": "found_file.jpg"}]
    }
    assert get_file_by_uuid(service, "uuid")["id"] == "1"

def test_get_file_by_uuid_not_found():
    """Verifies that None is returned if the file list is empty."""
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": []}
    assert get_file_by_uuid(service, "uuid") is None

def test_get_file_by_uuid_error():
    """Verifies exception handling during UUID search."""
    service = MagicMock()
    service.files.return_value.list.side_effect = create_http_error(500)
    assert get_file_by_uuid(service, "uuid") is None

# --- Delete Tests ---

def test_delete_file_by_uuid_success():
    """Verifies successful deletion returns True."""
    service = MagicMock()
    with patch("find_your_twin.google_drive.get_file_by_uuid", return_value={"id": "fid", "name": "del.txt"}):
        assert delete_file_by_uuid(service, "uuid") is True

def test_delete_file_by_uuid_not_found():
    """Verifies False is returned if UUID lookup fails."""
    service = MagicMock()
    with patch("find_your_twin.google_drive.get_file_by_uuid", return_value=None):
        assert delete_file_by_uuid(service, "uuid") is False

def test_delete_file_by_uuid_http_errors():
    """Verifies handling of 404, 403 errors during deletion."""
    service = MagicMock()
    with patch("find_your_twin.google_drive.get_file_by_uuid", return_value={"id": "fid", "name": "del.txt"}):
        # 404
        service.files.return_value.delete.return_value.execute.side_effect = create_http_error(404)
        assert delete_file_by_uuid(service, "uuid") is False
        
        # 403
        service.files.return_value.delete.return_value.execute.side_effect = create_http_error(403)
        assert delete_file_by_uuid(service, "uuid") is False

def test_delete_file_by_uuid_generic_error():
    """Verifies handling of generic exceptions during deletion."""
    service = MagicMock()
    with patch("find_your_twin.google_drive.get_file_by_uuid", return_value={"id": "fid", "name": "del.txt"}):
        service.files.return_value.delete.return_value.execute.side_effect = Exception("Boom")
        assert delete_file_by_uuid(service, "uuid") is False

def test_delete_file_by_id_success():
    """Verifies successful ID deletion returns True."""
    service = MagicMock()
    assert delete_file_by_id(service, "fid") is True

def test_delete_file_by_id_missing_id():
    """Verifies failure if ID is None."""
    assert delete_file_by_id(MagicMock(), None) is False

def test_delete_file_by_id_http_errors():
    """Verifies exception handling during ID deletion."""
    service = MagicMock()
    
    # 404
    service.files.return_value.delete.return_value.execute.side_effect = create_http_error(404)
    assert delete_file_by_id(service, "fid") is False

    # 403
    service.files.return_value.delete.return_value.execute.side_effect = create_http_error(403)
    assert delete_file_by_id(service, "fid") is False

def test_delete_file_by_id_generic_error():
    """Verifies generic exception handling during ID deletion."""
    service = MagicMock()
    service.files.return_value.delete.return_value.execute.side_effect = Exception("Boom")
    assert delete_file_by_id(service, "fid") is False


def test_get_or_create_app_folder_missing_service():
    """Verifies that None is returned immediately if the service object is missing."""
    assert get_or_create_app_folder(None) is None


def test_upload_bytes_to_folder_missing_service_or_id():
    """Verifies that None is returned if the service object or folder_id is missing during bytes upload."""
    # Test missing service
    assert upload_bytes_to_folder(None, "folder_id", "name", b"data", "type") is None

    # Test missing folder_id
    assert upload_bytes_to_folder(MagicMock(), None, "name", b"data", "type") is None
    

def test_get_image_bytes_by_id_missing_service():
    """Verifies that None is returned if the service object is missing."""
    assert get_image_bytes_by_id(None, "file_id_123") is None

def test_get_file_by_uuid_missing_service():
    """Verifies that None is returned if the service object is missing during UUID search."""
    assert get_file_by_uuid(None, "some-uuid") is None

def test_get_file_by_uuid_missing_uuid():
    """Verifies that None is returned if the UUID argument is missing or empty."""
    # Test with None
    assert get_file_by_uuid(MagicMock(), None) is None
    # Test with Empty String
    assert get_file_by_uuid(MagicMock(), "") is None


def test_delete_file_by_uuid_missing_service():
    """Verifies that False is returned if the service object is missing during deletion."""
    assert delete_file_by_uuid(None, "some-uuid") is False

def test_delete_file_by_id_missing_service():
    """Verifies that False is returned if the service object is missing during ID deletion."""
    assert delete_file_by_id(None, "some_file_id") is False


def test_delete_file_by_uuid_missing_uuid():
    """Verifies that False is returned if the UUID argument is missing or empty."""
    # Test with None
    assert delete_file_by_uuid(MagicMock(), None) is False
    # Test with Empty String
    assert delete_file_by_uuid(MagicMock(), "") is False