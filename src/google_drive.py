import os
import os.path
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


load_dotenv()

CLIENT_SECRET_FILE_PATH = os.getenv("GOOGLE_CLIENT_SECRET_PATH")
TOKEN_FILE_PATH = os.getenv("GOOGLE_TOKEN_PATH")
SCOPES = ['https://www.googleapis.com/auth/drive.file']
APP_FOLDER_NAME = "face-similarity"

def get_drive_service():
    """
    Authenticates and returns a Google Drive API service object.
    """
    creds = None
    # Check if the token file exists
    if os.path.exists(TOKEN_FILE_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE_PATH, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}")
                print("Could not refresh token. Please re-authenticate.")
                os.remove(TOKEN_FILE_PATH) 
                creds = None 
        
        # If no valid token, run the auth flow
        if not creds:
            if not os.path.exists(CLIENT_SECRET_FILE_PATH):
                print(f"CRITICAL ERROR: '{CLIENT_SECRET_FILE_PATH}' not found.")
                print("Please download it from your Google Cloud project's")
                print("OAuth 2.0 Credentials page and place it in this directory.")
                return None
            
            print(f"'{TOKEN_FILE_PATH}' not found or invalid, starting new auth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(TOKEN_FILE_PATH, 'w') as token:
            token.write(creds.to_json())
            print(f"Credentials saved to '{TOKEN_FILE_PATH}'")

    try:
        service = build('drive', 'v3', credentials=creds)
        print("Google Drive service created successfully.")
        return service
    except HttpError as error:
        print(f"An error occurred building the service: {error}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    

def list_all_files(service, page_size=100):
    """
    Lists all files in the user's Google Drive.
    """
    if service is None:
        print("Drive service not initialized.")
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
            print(f"Error listing files: {e}")
            break

    return files


def get_or_create_app_folder(service):
    """
    Finds the app folder by name, or creates it if it doesn't exist.
    Returns the folder's ID.
    """
    if not service:
        return None

    folder_name = APP_FOLDER_NAME
    
    try:
        # Search for the folder
        # 'q' is the query string. We're looking for a folder with our app's name.
        # The `drive.file` scope means we can *only* find folders this app has created.
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = response.get('files', [])

        if files:
            # Folder found
            folder_id = files[0].get('id')
            print(f"Found existing app folder: '{folder_name}' (ID: {folder_id})")
            return folder_id
        else:
            # Folder not found, create it
            print(f"App folder '{folder_name}' not found. Creating it...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            print(f"Successfully created app folder: '{folder_name}' (ID: {folder_id})")
            return folder_id

    except HttpError as error:
        print(f"An error occurred searching for or creating the folder: {error}")
        return None
    


def upload_file_to_folder(service, folder_id, local_file_path, uuid=None):
    """
    Uploads a single file to the specified Google Drive folder,
    optionally adding a custom UUID to its appProperties.
    """
    if not service or not folder_id:
        print("Service or folder_id is missing, cannot upload.")
        return None
    
    if not os.path.exists(local_file_path):
        print(f"Local file not found: {local_file_path}")
        return None

    try:
        file_name = os.path.basename(local_file_path)
        mime_type, _ = mimetypes.guess_type(local_file_path)
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]  # This is crucial for putting it *in* the folder
        }
        
        if uuid:
            print(f"Adding custom UUID to appProperties: {uuid}")
            file_metadata['appProperties'] = {
                'uuid': str(uuid)  # Store the UUID as a string
            }
        
        media = MediaFileUpload(local_file_path, mimetype=mime_type)
        
        print(f"Uploading '{file_name}' to Drive folder...")
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, appProperties' # <-- Request appProperties back
        ).execute()
        
        print(f"File uploaded successfully: '{file.get('name')}' (ID: {file.get('id')})")
        if file.get('appProperties'):
            print(f"  -> with appProperties: {file.get('appProperties')}")
        return file.get('id')

    except HttpError as error:
        print(f"An error occurred uploading the file: {error}")
        return None
    
def upload_bytes_to_folder(service, folder_id, file_name, file_bytes, mime_type, uuid_str=None):
    """
    Uploads raw bytes to the specified Google Drive folder.
    """
    if not service or not folder_id:
        print("Service or folder_id is missing, cannot upload.")
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
        
        print(f"Uploading '{file_name}' (from memory) to Drive folder...")
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name'
        ).execute()
        
        print(f"File uploaded successfully: '{file.get('name')}' (ID: {file.get('id')})")
        return file.get('id')

    except HttpError as error:
        print(f"An error occurred uploading the file from bytes: {error}")
        return None
    

def get_image_bytes_by_id(service, file_id):
    """
    Downloads a file's content (like an image) directly into memory
    and returns its raw bytes.
    """
    if not service:
        print("Service is missing, cannot download.")
        return None
        
    print(f"Attempting to download file ID: {file_id} into memory...")
    
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
            print(f"Download to memory {int(status.progress() * 100)}%.")
        
        print(f"File downloaded successfully to memory.")
        
        # Go to the beginning of the buffer and get its value
        fh.seek(0)
        return fh.getvalue()
        
    except HttpError as error:
        print(f"An error occurred downloading the file: {error}")
        if error.resp.status == 404:
            print("Error: File not found. Check the file ID.")
        elif error.resp.status == 403:
             print("Error: Permission denied. Does your app have access to this specific file?")
             print("With 'drive.file' scope, your app can only access files it created.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_image_base64_by_id(service, file_id, mime_type="image/jpeg"):
    """
    Efficiently gets a file from Drive and returns it as a
    Base64 data URI string, ready to be used in an <img> src tag.
    """
    image_bytes = get_image_bytes_by_id(service, file_id)
    
    if image_bytes:
        print("Encoding image bytes to Base64...")
        # Encode the raw bytes into a Base64 string
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Format as a data URI
        data_uri = f"data:{mime_type};base64,{base64_string}"
        print("Successfully created Base64 data URI.")
        return data_uri
    else:
        print("Could not get image bytes, returning None.")
        return None
    

def get_file_by_uuid(service, uuid):
    """
    Finds a file in Google Drive by its custom appProperties UUID.
    Note: This search is private to your app.
    """
    if not service:
        print("Service is missing, cannot search.")
        return None
        
    if not uuid:
        print("UUID is missing, cannot search.")
        return None

    try:
        # Note: appProperties keys and values must be in single quotes
        # inside the query string.
        query = f"appProperties has {{ key='uuid' and value='{uuid}' }} and trashed=false"
        
        print(f"Searching for file with UUID: {uuid}...")
        
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, appProperties)',  # Get all relevant fields
            pageSize=1  # We only expect one, so limit to 1
        ).execute()
        
        files = response.get('files', [])
        
        if files:
            found_file = files[0]
            print(f"Found file: '{found_file.get('name')}' (ID: {found_file.get('id')})")
            print(f"  -> with appProperties: {found_file.get('appProperties')}")
            return found_file
        else:
            print(f"No file found with UUID: {uuid}")
            return None

    except HttpError as error:
        print(f"An error occurred while searching by UUID: {error}")
        return None
    

def delete_file_by_uuid(service, uuid_str):
    """
    Deletes a file from Google Drive using its custom appProperties UUID.
    Returns True if deleted, False otherwise.
    """
    if not service:
        print("Service is missing, cannot delete.")
        return False

    if not uuid_str:
        print("UUID is missing, cannot delete.")
        return False

    # Reuse your existing search helper
    file_info = get_file_by_uuid(service, uuid_str)

    if not file_info:
        print(f"No file found with UUID: {uuid_str}")
        return False

    file_id = file_info.get("id")
    file_name = file_info.get("name")

    try:
        print(f"Deleting file '{file_name}' (ID: {file_id}) ...")

        service.files().delete(fileId=file_id).execute()

        print(f"Successfully deleted file '{file_name}' (UUID={uuid_str}).")
        return True

    except HttpError as error:
        print(f"An error occurred deleting the file: {error}")
        if error.resp.status == 404:
            print("Error: File already deleted or not found.")
        elif error.resp.status == 403:
            print("Error: Permission denied. Your app may not own this file.")
        return False
    except Exception as e:
        print(f"Unexpected error while deleting: {e}")
        return False

def delete_file_by_id(service, file_id):
    """
    Deletes a file from Google Drive using its file ID.
    Returns True if deleted, False otherwise.
    """
    if not service:
        print("Service is missing, cannot delete.")
        return False

    if not file_id:
        print("File ID is missing, cannot delete.")
        return False

    try:
        service.files().delete(fileId=file_id).execute()

        return True

    except HttpError as error:
        print(f"An error occurred deleting the file: {error}")
        if error.resp.status == 404:
            print("Error: File not found or already deleted.")
        elif error.resp.status == 403:
            print("Error: Permission denied. Your app may not own this file.")
        return False

    except Exception as e:
        print(f"Unexpected error while deleting: {e}")
        return False
