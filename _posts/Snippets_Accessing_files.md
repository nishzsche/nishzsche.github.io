---
layout: post
title: "Open files from Google Drive"
---

# Open files from Google Drive

The example below shows how to mount your Google Drive in your virtual machine using an authorization code.


```
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
```

# Open files from GitHub

To open a file from GitHub, you can clone the repo or fetch the file directly.


If you are trying to access files from a private repo, you must use a [GitHub access token](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line).


```
# Clone the entire repo.
!git clone -l -s git://github.com/jakevdp/PythonDataScienceHandbook.git cloned-repo
%cd cloned-repo
!ls
```


```
# Fetch a single <1MB file using the raw GitHub URL.
!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://api.github.com/repos/jakevdp/PythonDataScienceHandbook/contents/notebooks/data/california_cities.csv
```

# Open files from your local file system

`files.upload` returns a dictionary of the files which were uploaded.
The dictionary is keyed by the file name, the value is the data which was uploaded.


```
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```

# Open files from GCS with gsutil

- Use the [Cloud Resource Manager](https://cloud.google.com/resource-manager/) to create a project if you do not already have one.
- [Enable billing](https://support.google.com/cloud/answer/6293499#enable-billing) for the project.
- See [Google Cloud Storage (GCS) Documentation](https://cloud.google.com/storage/) for more info.



```
from google.colab import auth
auth.authenticate_user()

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = '[your Cloud Platform project ID]'
!gcloud config set project {project_id}
```


```
# Download the file from a given Google Cloud Storage bucket.
!gsutil cp gs://{bucket_name}/file_to_download.txt /tmp/gsutil_download.txt

# Print the result to make sure the transfer worked.
!cat /tmp/gsutil_download.txt
```

# Open files from GCS with the Cloud Storage Python API

- Use the [Cloud Resource Manager](https://cloud.google.com/resource-manager/) to create a project if you do not already have one.
- [Enable billing](https://support.google.com/cloud/answer/6293499#enable-billing) for the project.
- See [Google Cloud Storage (GCS) Documentation](https://cloud.google.com/storage/) for more info.

You can also use [gsutil](https://cloud.google.com/storage/docs/gsutil) to interact with [Google Cloud Storage (GCS)](https://cloud.google.com/storage/).
This snippet is based on [a larger example](https://github.com/GoogleCloudPlatform/storage-file-transfer-json-python/blob/master/chunked_transfer.py).


```
# Authenticate to GCS.
from google.colab import auth
auth.authenticate_user()

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = '[your Cloud Platform project ID]'

# Create the service client.
from googleapiclient.discovery import build
gcs_service = build('storage', 'v1')

from apiclient.http import MediaIoBaseDownload

with open('/tmp/gsutil_download.txt', 'wb') as f:
  # Download the file from a given Google Cloud Storage bucket.
  request = gcs_service.objects().get_media(bucket=bucket_name,
                                            object='to_upload.txt')
  media = MediaIoBaseDownload(f, request)

  done = False
  while not done:
    # _ is a placeholder for a progress object that we ignore.
    # (Our file is small, so we skip reporting progress.)
    _, done = media.next_chunk()

print('Download complete')
```


```
# Inspect the file we downloaded to /tmp
!cat /tmp/downloaded_from_gcs.txt
```
