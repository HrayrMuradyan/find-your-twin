import os
import mimetypes
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
from dotenv import load_dotenv
import io
import base64
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import sys
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import blur_str

load_dotenv()

CLIENT_SECRET_FILE_PATH = PROJECT_ROOT / os.getenv("GOOGLE_CLIENT_SECRET_PATH", "credentials/client_secret.json")
TOKEN_FILE_PATH = PROJECT_ROOT / os.getenv("GOOGLE_TOKEN_PATH", "credentials/token.json")
SCOPES = ['https://www.googleapis.com/auth/drive.file']
APP_FOLDER_NAME = "face-similarity"

def get_drive_service():
    """
    Authenticates and returns a Google Drive API service object.
    """
    creds = None
    # Check if the token file exists
    if TOKEN_FILE_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE_PATH, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials of Google Drive API...")
            try:
                creds.refresh(Request())
                logger.info("Successfuly refreshed the expired credentials of Google Drive API. Ready to use.")

            except Exception as e:
                logger.exception("Error refreshing token: %s", e)
                logger.exception("Could not refresh token. Please re-authenticate.")
                TOKEN_FILE_PATH.unlink()
                creds = None 
        
        # If no valid token, run the auth flow
        if not creds:
            if not CLIENT_SECRET_FILE_PATH.exists():
                logging.error(
                    "Secret file path not found: %s",
                    str(CLIENT_SECRET_FILE_PATH)
                )
                logging.error("Please download it from your Google Cloud project's")
                logging.error("OAuth 2.0 Credentials page and place it in this directory.")
                return None
            
            logging.error(
                "Token not found at %s not found or invalid, starting new auth flow...",
                str(TOKEN_FILE_PATH)
            )
            
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_FILE_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(TOKEN_FILE_PATH, 'w') as token:
            token.write(creds.to_json())
            logging.info(
                "Credentials saved to %s", 
                TOKEN_FILE_PATH.relative_to(PROJECT_ROOT)
            )

    try:
        service = build('drive', 'v3', credentials=creds)
        logging.info("Google Drive service created successfully.")
        return service
    except HttpError as error:
        logging.exception("An error occurred building the service: %s", error)
        return None
    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        return None
    

def list_all_files(service, page_size=100):
    """
    Lists all files in the user's Google Drive.
    """
    if service is None:
        logging.warning("Drive service not initialized.")
        return []

    files = []
    page_token = None

    while True:
        try:
            response = service.files().list(
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, parents, size, appProperties)",
                pageToken=page_token
            ).execute()

            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken')

            if not page_token:
                break
        except Exception as e:
            logger.exception("Error listing files: %s", e)
            break

    return files


def get_or_create_app_folder(service):
    """
    Finds the app folder by name, or creates it if it doesn't exist.
    Returns the folder's ID.
    """
    if not service:
        return None
    
    try:
        # Search for the folder
        # 'q' is the query string. We're looking for a folder with our app's name.
        # The `drive.file` scope means we can only find folders this app has created.
        query = f"mimeType='application/vnd.google-apps.folder' and name='{APP_FOLDER_NAME}' and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        if files:
            folder_id = files[0].get('id')
            logging.info(
                "Found an existing app folder: %s (ID: %s)",
                APP_FOLDER_NAME,
                folder_id
            )

            return folder_id
        else:
            # Folder not found, create it
            logging.info("App folder %s not found. Creating it...", APP_FOLDER_NAME)
            file_metadata = {
                'name': APP_FOLDER_NAME,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logging.info(
                "Successfully created app folder: %s (ID: %s)",
                APP_FOLDER_NAME,
                folder_id
            )
            return folder_id

    except HttpError as error:
        logging.exception("An error occurred searching for or creating the folder: %s", error)
        return None
    


def upload_file_to_folder(service,
                          folder_id,
                          local_file_path, uuid=None):
    """
    Uploads a single file to the specified Google Drive folder,
    optionally adding a custom UUID to its appProperties.
    """
    if not isinstance(local_file_path, (str, Path)):
        raise TypeError(
            "local_file_path should be a string or Path object. You have %s",
            type(local_file_path)
        )
    
    local_file_path = Path(local_file_path)

    if not service or not folder_id:
        logging.warning("Service or folder_id is missing, cannot upload.")
        return None
    
    if not local_file_path.exists():
        logging.warning("Local file not found: %s", str(local_file_path))
        return None

    try:
        file_name = local_file_path.name
        mime_type, _ = mimetypes.guess_type(local_file_path)
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        if uuid:
            logging.info(
                "Adding custom UUID to appProperties: %s",
                blur_str(uuid)
            )
            file_metadata['appProperties'] = {
                'uuid': str(uuid) 
            }
        
        media = MediaFileUpload(local_file_path, mimetype=mime_type)
        
        logging.info("Uploading %s to Drive folder...", blur_str(file_name))
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, appProperties'
        ).execute()
        
        logging.info(
            "File uploaded successfully: %s (ID: %s)",
            blur_str(file.get('name')),
            file.get('id')
        )

        return file.get('id')

    except HttpError as error:
        logging.exception("An error occurred uploading the file: %s", error)
        return None
    
def upload_bytes_to_folder(service,
                           folder_id,
                           file_name,
                           file_bytes,
                           mime_type,
                           uuid_str=None):
    """
    Uploads raw bytes to the specified Google Drive folder.
    """
    if not service or not folder_id:
        logging.warning("Service or folder_id is missing, cannot upload.")
        return None

    try:
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        # Add custom UUID if provided
        if uuid_str:
            file_metadata['appProperties'] = {
                'uuid': uuid_str
            }
            
        # Wrap bytes in an in-memory file-like object
        media = MediaIoBaseUpload(
            io.BytesIO(file_bytes),
            mimetype=mime_type,
            resumable=True
        )
        
        logging.info("Uploading %s (from memory) to Drive folder...", blur_str(file_name))
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()
        
        logging.info(
            "File uploaded successfully: %s (ID: %s)",
            blur_str(file.get('name')),
            file.get('id')
        )

        return file.get('id')

    except HttpError as error:
        logging.exception("An error occurred uploading the file from bytes: %s", error)
        return None
    

def get_image_bytes_by_id(service, file_id):
    """
    Downloads a file's content (like an image) directly into memory
    and returns its raw bytes.
    """
    if not service:
        logging.warning("Service is missing, cannot download.")
        return None
        
    logging.info("Attempting to download file ID: %s into memory...", file_id)
    
    try:
        # Request the media content of the file
        request = service.files().get_media(fileId=file_id)
        
        # Use io.BytesIO() to create an in-memory binary buffer
        fh = io.BytesIO()
        
        # MediaIoBaseDownload downloads the file in chunks to the in-memory buffer
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logging.info("Download to memory %s%%.", int(status.progress() * 100))
        
        logging.info("File downloaded successfully to memory.")
        
        # Go to the beginning of the buffer and get its value
        fh.seek(0)
        return fh.getvalue()
        
    except HttpError as error:
        logging.exception("An error occurred downloading the file: %s", error)
        if error.resp.status == 404:
            logging.error("Error: File not found. Check the file ID.")
        elif error.resp.status == 403:
             logging.error("Error: Permission denied. Does your app have access to this specific file?")
             logging.error("With 'drive.file' scope, your app can only access files it created.")
        return None
    except Exception as e:
        logging.exception("An unexpected error occurred: %s", e)
        return None

def get_image_base64_by_id(service, file_id, mime_type="image/jpeg"):
    """
    Efficiently gets a file from Drive and returns it as a
    Base64 data URI string, ready to be used in an <img> src tag.
    """
    image_bytes = get_image_bytes_by_id(service, file_id)
    
    if image_bytes:
        logging.info("Encoding image bytes to Base64...")
        # Encode the raw bytes into a Base64 string
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Format as a data URI
        data_uri = f"data:{mime_type};base64,{base64_string}"
        logging.info("Successfully created Base64 data URI.")
        return data_uri
    else:
        logging.warning("Could not get image bytes, returning None.")
        return None
    

def get_file_by_uuid(service, uuid):
    """
    Finds a file in Google Drive by its custom appProperties UUID.
    Note: This search is private to your app.
    """
    if not service:
        logging.warning("Service is missing, cannot search.")
        return None
        
    if not uuid:
        logging.warning("UUID is missing, cannot search.")
        return None

    try:
        query = f"appProperties has {{ key='uuid' and value='{uuid}' }} and trashed=false"
        
        logging.info("Searching for file with UUID: %s...", blur_str(uuid))
        
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, appProperties)',  
            pageSize=1  
        ).execute()
        
        files = response.get('files', [])
        
        if files:
            found_file = files[0]
            logging.info("Found file: %s (ID: %s)", blur_str(found_file.get('name')), found_file.get('id'))
            return found_file
        else:
            logging.warning("No file found with UUID: %s", blur_str(uuid))
            return None

    except HttpError as error:
        logging.exception("An error occurred while searching by UUID: %s", error)
        return None
    

def delete_file_by_uuid(service, uuid_str):
    """
    Deletes a file from Google Drive using its custom appProperties UUID.
    Returns True if deleted, False otherwise.
    """
    if not service:
        logging.warning("Service is missing, cannot delete.")
        return False

    if not uuid_str:
        logging.warning("UUID is missing, cannot delete.")
        return False

    # Reuse your existing search helper
    file_info = get_file_by_uuid(service, uuid_str)

    if not file_info:
        logging.warning("No file found with UUID: %s", blur_str(uuid_str))
        return False

    file_id = file_info.get("id")
    file_name = file_info.get("name")

    try:
        logging.info("Deleting file '%s' (ID: %s) ...", file_name, file_id)

        service.files().delete(fileId=file_id).execute()

        logging.info("Successfully deleted file '%s' (UUID=%s).", file_name, blur_str(uuid_str))
        return True

    except HttpError as error:
        logging.exception("An error occurred deleting the file: %s", error)
        if error.resp.status == 404:
            logging.exception("Error: File already deleted or not found.")
        elif error.resp.status == 403:
            logging.exception("Error: Permission denied. Your app may not own this file.")
        return False
    except Exception as e:
        logging.exception("Unexpected error while deleting: %s", e)
        return False

def delete_file_by_id(service, file_id):
    """
    Deletes a file from Google Drive using its file ID.
    Returns True if deleted, False otherwise.
    """
    if not service:
        logging.warning("Service is missing, cannot delete.")
        return False

    if not file_id:
        logging.warning("File ID is missing, cannot delete.")
        return False

    try:
        logging.info("Trying deleting file with ID = %s", file_id)
        service.files().delete(fileId=file_id).execute()

        return True

    except HttpError as error:
        logging.exception("An error occurred deleting the file: %s", error)
        if error.resp.status == 404:
            logging.exception("Error: File not found or already deleted.")
        elif error.resp.status == 403:
            logging.exception("Error: Permission denied. Your app may not own this file.")
        return False

    except Exception as e:
        logging.exception("Unexpected error while deleting: %s", e)
        return False
