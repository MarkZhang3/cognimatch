import io
import base64
import requests
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

########################
# CONFIG
########################

SERVICE_ACCOUNT_FILE = './service_account.json'
SPREADSHEET_ID = '1z8uPq8oGMhmmYjQEny-vOVh-AqJpEwSrF8rkadzpR38'
RANGE_NAME = 'Form Responses 1!A1:AM'  # Adjust range if needed
FORM_ENDPOINT = "http://34.27.216.29:8000/save_form"  # Replace with your actual endpoint
# Optionally, you could load an endpoint from environment variables, e.g.:
# FORM_ENDPOINT = os.getenv("YOUR_ENDPOINT_VAR")

########################
# SHEETS READING
########################

def get_sheets_service():
    """
    Build and return a Sheets API service client.
    """
    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return build('sheets', 'v4', credentials=creds)

def fetch_form_responses():
    """
    Fetch rows from the Google Sheet. Each row is a list of cell values.
    Returns: a list of rows.
    """
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME
    ).execute()
    rows = result.get('values', [])
    return rows

########################
# DRIVE / FILE HANDLING
########################

def get_drive_service():
    """
    Build and return a Drive API service client.
    """
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def extract_file_id(drive_link):
    """
    Given a typical Drive link, extract the file ID.
    For example:
      https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    We want to return the 'FILE_ID' portion.
    """
    # ONLY handles the following patterns:
    #   https://drive.google.com/file/d/<id>/view?usp=sharing
    #   https://drive.google.com/open?id=<id>
    #   etc.
    if '/d/' in drive_link:
        # e.g. https://drive.google.com/file/d/ABC123/view?usp=sharing
        # split by '/d/' → ["https://drive.google.com/file", "ABC123/view?usp=sharing"]
        part = drive_link.split('/d/')[-1]
        # then split by '/', so "ABC123/view?usp=sharing" → ["ABC123", "view?usp=sharing"]
        file_id = part.split('/')[0]
        print("file id from extract", file_id)
        return file_id
    elif 'open?id=' in drive_link:
        # e.g. https://drive.google.com/open?id=ABC123
        # split by '?id=' → ["https://drive.google.com/open", "ABC123"]
        file_id = drive_link.split('open?id=')[-1]
        print("file id from extract", file_id)
        return file_id
    else:
        # If it doesn't match a known pattern, you might want to print an error or return None
        print(f"Could not parse file ID from link: {drive_link}")
        return None

def fetch_and_encode_file(file_id):
    """
    Fetch file (image or PDF) from Drive using file_id via the Drive API,
    return base64-encoded bytes as a string.
    """
    drive_service = get_drive_service()
    request = drive_service.files().get_media(fileId=file_id)
    
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    file_bytes = fh.getvalue()
    encoded_str = base64.b64encode(file_bytes).decode('utf-8')
    return encoded_str

########################
# SEND TO ENDPOINT
########################

def send_form_data_to_endpoint(row_data):

    response = requests.post(FORM_ENDPOINT, json=row_data)
    
    if response.status_code == 200:
        print("Data sent successfully:", response.text)
    else:
        print("Error sending data:", response.status_code, response.text)

########################
# MAIN
########################

def fetch_and_send():
    rows = fetch_form_responses()
    if not rows:
        print("No data found.")
        return

    # Let's assume the first row is headers, so skip it
    headers = rows[0]
    data_rows = rows[1:]

    # Print them just to see what we have
    print("Headers:", headers)

    img_idx = 13
    for i, row in enumerate(data_rows, start=2):  # start=2 meaning row #2 in the sheet
        # Safety check: row could have fewer columns if not all questions were answered
        # if len(row) < 4:
        #     print(f"Skipping row {i} (not enough columns).")
        #     continue
        drive_links = row[img_idx].split(', ')
        base64_files = []
        for drive_link in drive_links:
            file_id = extract_file_id(drive_link)
            if file_id:
                base64_file = fetch_and_encode_file(file_id)
            else:
                base64_file = None
            base64_files.append(base64_file)

        captions = row[img_idx + 1].split('\n')
        
        
        # data = {
        #     "timestamp": row[0],           
        #     "name": row[1],
        #     "age": row[2],
        #     "brief background": row[3],       
        #     "mcquestion": row[2],
        #     "images (base64)": base64_files, 
        #     "captions": captions,
        # }
        data = {
            headers[j] : row[j] for j in range(len(row)) if headers[j] not in ['Upload Pictures', 'Caption for pictures']
        }

        data['Pictures (base64)'] = base64_files 
        data['Captions'] = captions
        print(row[img_idx + 1], len(row), len(headers))

        to_send = {
            'id': str(i), 
            'form': data
        }

        print(f"===== Sending row {i} data to endpoint ====")
        print(data['Captions'])
        print(data['Additional Notes'])
        send_form_data_to_endpoint(to_send)

if __name__ == "__main__":
    # fetch_and_send()

    convo_endpoint = 'http://34.27.216.29:8000/start_convo'
    to_send = {
        'convo_id': '1',
        'speaker_1_id': '2',
        'speaker_2_id': '3',
    }
    response = requests.post(convo_endpoint, json=to_send)


    print(response)

