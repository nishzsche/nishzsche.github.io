---
layout: post
title: "IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/notebooks/revisiting_the_titanic_with_pycaret.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'titanic:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F3136%2F26502%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240515%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240515T183613Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D910133f4f00fd2d8ba9ecb704215ce385b1ef98bf6f2deeac50a1f66b80a4b2973cfb7489966e541071773c627d83e7068f07bc9f739112fdba4a5c599c2de0528ba52c97aa32bf2d41c0bf91bf52a6b50fa61c7cd0f6a684d213e816cbf3a712151b530bd1a6525ab2dbe51eaecca817bbc38e6297daa4dca59224b002ac83d249d132a13cc05da1fd572a71f9e6a2ac01dbb12a02446507c05ca645090a226c5ed3bc5bddcd9f538de635912f7a6ff8f433ea74a31f35208b83170d6aaf9cce523529b9da9a7ace521fe25b6adcab37f572d3218aafd2aecbc06277e57ade5334548f8b88135d8dac2536c522889e3935753ecb0899f72f0d534d7671e7e69'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'
Downloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')
```

    Downloading titanic, 34877 bytes compressed
    [==================================================] 34877 bytes downloaded
    Downloaded and uncompressed: titanic
    Data source import complete.



```python
#Scikit-learn version: 1.4.2
#PyCaret version: 3.3.2

!pip install scikit-learn==1.4.2
!pip install pycaret==3.3.2
```

    Collecting scikit-learn==1.4.2
      Downloading scikit_learn-1.4.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.1/12.1 MB[0m [31m32.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.10/dist-packages (from scikit-learn==1.4.2) (1.25.2)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn==1.4.2) (1.11.4)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn==1.4.2) (1.4.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn==1.4.2) (3.5.0)
    Installing collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.2.2
        Uninstalling scikit-learn-1.2.2:
          Successfully uninstalled scikit-learn-1.2.2
    Successfully installed scikit-learn-1.4.2
    Collecting pycaret==3.3.2
      Downloading pycaret-3.3.2-py3-none-any.whl (486 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m486.1/486.1 kB[0m [31m6.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: ipython>=5.5.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (7.34.0)
    Requirement already satisfied: ipywidgets>=7.6.5 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (7.7.1)
    Requirement already satisfied: tqdm>=4.62.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (4.66.4)
    Requirement already satisfied: numpy<1.27,>=1.21 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (1.25.2)
    Requirement already satisfied: pandas<2.2.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (2.0.3)
    Requirement already satisfied: jinja2>=3 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (3.1.4)
    Requirement already satisfied: scipy<=1.11.4,>=1.6.1 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (1.11.4)
    Collecting joblib<1.4,>=1.2.0 (from pycaret==3.3.2)
      Downloading joblib-1.3.2-py3-none-any.whl (302 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.2/302.2 kB[0m [31m9.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scikit-learn>1.4.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (1.4.2)
    Collecting pyod>=1.1.3 (from pycaret==3.3.2)
      Downloading pyod-1.1.3.tar.gz (160 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m160.5/160.5 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    Collecting imbalanced-learn>=0.12.0 (from pycaret==3.3.2)
      Downloading imbalanced_learn-0.12.2-py3-none-any.whl (257 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m258.0/258.0 kB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting category-encoders>=2.4.0 (from pycaret==3.3.2)
      Downloading category_encoders-2.6.3-py2.py3-none-any.whl (81 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.9/81.9 kB[0m [31m6.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: lightgbm>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (4.1.0)
    Requirement already satisfied: numba>=0.55.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (0.58.1)
    Requirement already satisfied: requests>=2.27.1 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (2.31.0)
    Requirement already satisfied: psutil>=5.9.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (5.9.5)
    Requirement already satisfied: markupsafe>=2.0.1 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (2.1.5)
    Requirement already satisfied: importlib-metadata>=4.12.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (7.1.0)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (5.10.4)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (2.2.1)
    Collecting deprecation>=2.1.0 (from pycaret==3.3.2)
      Downloading deprecation-2.1.0-py2.py3-none-any.whl (11 kB)
    Collecting xxhash (from pycaret==3.3.2)
      Downloading xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib<3.8.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (3.7.1)
    Collecting scikit-plot>=0.3.7 (from pycaret==3.3.2)
      Downloading scikit_plot-0.3.7-py3-none-any.whl (33 kB)
    Requirement already satisfied: yellowbrick>=1.4 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (1.5)
    Requirement already satisfied: plotly>=5.14.0 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (5.15.0)
    Collecting kaleido>=0.2.1 (from pycaret==3.3.2)
      Downloading kaleido-0.2.1-py2.py3-none-manylinux1_x86_64.whl (79.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.9/79.9 MB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting schemdraw==0.15 (from pycaret==3.3.2)
      Downloading schemdraw-0.15-py3-none-any.whl (106 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m106.8/106.8 kB[0m [31m11.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting plotly-resampler>=0.8.3.1 (from pycaret==3.3.2)
      Downloading plotly_resampler-0.10.0-py3-none-any.whl (80 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m80.7/80.7 kB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: statsmodels>=0.12.1 in /usr/local/lib/python3.10/dist-packages (from pycaret==3.3.2) (0.14.2)
    Collecting sktime==0.26.0 (from pycaret==3.3.2)
      Downloading sktime-0.26.0-py3-none-any.whl (21.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.8/21.8 MB[0m [31m20.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tbats>=1.1.3 (from pycaret==3.3.2)
      Downloading tbats-1.1.3-py3-none-any.whl (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.0/44.0 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pmdarima>=2.0.4 (from pycaret==3.3.2)
      Downloading pmdarima-2.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m40.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting wurlitzer (from pycaret==3.3.2)
      Downloading wurlitzer-3.1.0-py3-none-any.whl (8.4 kB)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from sktime==0.26.0->pycaret==3.3.2) (24.0)
    Collecting scikit-base<0.8.0 (from sktime==0.26.0->pycaret==3.3.2)
      Downloading scikit_base-0.7.8-py3-none-any.whl (130 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.1/130.1 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.10/dist-packages (from category-encoders>=2.4.0->pycaret==3.3.2) (0.5.6)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from imbalanced-learn>=0.12.0->pycaret==3.3.2) (3.5.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=4.12.0->pycaret==3.3.2) (3.18.1)
    Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (67.7.2)
    Collecting jedi>=0.16 (from ipython>=5.5.0->pycaret==3.3.2)
      Downloading jedi-0.19.1-py2.py3-none-any.whl (1.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m45.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (4.4.2)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (0.7.5)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (5.7.1)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (3.0.43)
    Requirement already satisfied: pygments in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (2.16.1)
    Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (0.2.0)
    Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (0.1.7)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.5.0->pycaret==3.3.2) (4.9.0)
    Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret==3.3.2) (5.5.6)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret==3.3.2) (0.2.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret==3.3.2) (3.6.6)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets>=7.6.5->pycaret==3.3.2) (3.0.10)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (1.2.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (1.4.5)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib<3.8.0->pycaret==3.3.2) (2.8.2)
    Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret==3.3.2) (2.19.1)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret==3.3.2) (4.19.2)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in /usr/local/lib/python3.10/dist-packages (from nbformat>=4.2.0->pycaret==3.3.2) (5.7.2)
    Requirement already satisfied: llvmlite<0.42,>=0.41.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba>=0.55.0->pycaret==3.3.2) (0.41.1)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<2.2.0->pycaret==3.3.2) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<2.2.0->pycaret==3.3.2) (2024.1)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from plotly>=5.14.0->pycaret==3.3.2) (8.3.0)
    Collecting dash>=2.9.0 (from plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading dash-2.17.0-py3-none-any.whl (7.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.5/7.5 MB[0m [31m23.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting orjson<4.0.0,>=3.8.0 (from plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading orjson-3.10.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m142.5/142.5 kB[0m [31m13.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tsdownsample>=0.1.3 (from plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading tsdownsample-0.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m24.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in /usr/local/lib/python3.10/dist-packages (from pmdarima>=2.0.4->pycaret==3.3.2) (3.0.10)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.10/dist-packages (from pmdarima>=2.0.4->pycaret==3.3.2) (2.0.7)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from pyod>=1.1.3->pycaret==3.3.2) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret==3.3.2) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret==3.3.2) (3.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->pycaret==3.3.2) (2024.2.2)
    Requirement already satisfied: Flask<3.1,>=1.0.4 in /usr/local/lib/python3.10/dist-packages (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (2.2.5)
    Requirement already satisfied: Werkzeug<3.1 in /usr/local/lib/python3.10/dist-packages (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (3.0.3)
    Collecting dash-html-components==2.0.0 (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading dash_html_components-2.0.0-py3-none-any.whl (4.1 kB)
    Collecting dash-core-components==2.0.0 (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading dash_core_components-2.0.0-py3-none-any.whl (3.8 kB)
    Collecting dash-table==5.0.0 (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading dash_table-5.0.0-py3-none-any.whl (3.9 kB)
    Requirement already satisfied: typing-extensions>=4.1.1 in /usr/local/lib/python3.10/dist-packages (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (4.11.0)
    Collecting retrying (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2)
      Downloading retrying-1.3.4-py3-none-any.whl (11 kB)
    Requirement already satisfied: nest-asyncio in /usr/local/lib/python3.10/dist-packages (from dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (1.6.0)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.5->pycaret==3.3.2) (6.1.12)
    Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.6.5->pycaret==3.3.2) (6.3.3)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=5.5.0->pycaret==3.3.2) (0.8.4)
    Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret==3.3.2) (23.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret==3.3.2) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret==3.3.2) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=4.2.0->pycaret==3.3.2) (0.18.1)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core!=5.0.*,>=4.12->nbformat>=4.2.0->pycaret==3.3.2) (4.2.1)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.10/dist-packages (from pexpect>4.3->ipython>=5.5.0->pycaret==3.3.2) (0.7.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=5.5.0->pycaret==3.3.2) (0.2.13)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.10/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (6.5.5)
    Requirement already satisfied: itsdangerous>=2.0 in /usr/local/lib/python3.10/dist-packages (from Flask<3.1,>=1.0.4->dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (2.2.0)
    Requirement already satisfied: click>=8.0 in /usr/local/lib/python3.10/dist-packages (from Flask<3.1,>=1.0.4->dash>=2.9.0->plotly-resampler>=0.8.3.1->pycaret==3.3.2) (8.1.7)
    Requirement already satisfied: pyzmq<25,>=17 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (24.0.1)
    Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (23.1.0)
    Requirement already satisfied: nbconvert>=5 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (6.5.4)
    Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.8.3)
    Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.18.1)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.20.0)
    Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.10/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.0.0)
    Requirement already satisfied: jupyter-server>=1.8 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.24.0)
    Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.2.4)
    Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (4.9.4)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (4.12.3)
    Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (6.1.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.7.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.4)
    Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.3.0)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.8.4)
    Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.10.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.5.1)
    Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.3.0)
    Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (21.2.0)
    Requirement already satisfied: anyio<4,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-server>=1.8->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (3.7.1)
    Requirement already satisfied: websocket-client in /usr/local/lib/python3.10/dist-packages (from jupyter-server>=1.8->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.8.0)
    Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.16.0)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (2.5)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert>=5->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (0.5.1)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server>=1.8->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.3.1)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server>=1.8->nbclassic>=0.4.7->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (1.2.1)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.6.5->pycaret==3.3.2) (2.22)
    Building wheels for collected packages: pyod
      Building wheel for pyod (setup.py) ... [?25l[?25hdone
      Created wheel for pyod: filename=pyod-1.1.3-py3-none-any.whl size=190250 sha256=c66adda1626112c863b0bcf55b7c923ca0e58a0121897595dd47ea220c0682aa
      Stored in directory: /root/.cache/pip/wheels/05/f8/db/124d43bec122d6ec0ab3713fadfe25ebed8af52ec561682b4e
    Successfully built pyod
    Installing collected packages: kaleido, dash-table, dash-html-components, dash-core-components, xxhash, wurlitzer, tsdownsample, scikit-base, schemdraw, retrying, orjson, joblib, jedi, deprecation, sktime, scikit-plot, pyod, imbalanced-learn, dash, pmdarima, plotly-resampler, category-encoders, tbats, pycaret
      Attempting uninstall: joblib
        Found existing installation: joblib 1.4.2
        Uninstalling joblib-1.4.2:
          Successfully uninstalled joblib-1.4.2
      Attempting uninstall: imbalanced-learn
        Found existing installation: imbalanced-learn 0.10.1
        Uninstalling imbalanced-learn-0.10.1:
          Successfully uninstalled imbalanced-learn-0.10.1
    Successfully installed category-encoders-2.6.3 dash-2.17.0 dash-core-components-2.0.0 dash-html-components-2.0.0 dash-table-5.0.0 deprecation-2.1.0 imbalanced-learn-0.12.2 jedi-0.19.1 joblib-1.3.2 kaleido-0.2.1 orjson-3.10.3 plotly-resampler-0.10.0 pmdarima-2.0.4 pycaret-3.3.2 pyod-1.1.3 retrying-1.3.4 schemdraw-0.15 scikit-base-0.7.8 scikit-plot-0.3.7 sktime-0.26.0 tbats-1.1.3 tsdownsample-0.1.3 wurlitzer-3.1.0 xxhash-3.4.1



```python
! ls /kaggle/input/titanic
```

    gender_submission.csv  test.csv  train.csv



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
data_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_files.append(os.path.join(dirname, filename))
print(data_files)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    ['/kaggle/input/titanic/train.csv', '/kaggle/input/titanic/test.csv', '/kaggle/input/titanic/gender_submission.csv']


# References

1. [Britannica](https://www.britannica.com/topic/Titanic)



# Exploratory Data Analysis


---

![route.png](data:image/png;base64,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)
## Files

1.   `df_train` - development dataset
2.   `df_test` - submission dataset
3.   `df_gender` - sample submission dataset where all female passengers are assumed to have survived - 76% accuracy recorded with this naive approach!




```python
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(df_train.shape, df_test.shape, df_gender.shape)
```

    (891, 12) (418, 11) (418, 2)



```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB


## Observations

-   ID column : `PassengerID`
-   Target (DV) column : `Survived` {0|1}
-   Non-Null IDV columns
  - `Class`
  - `Name`
  - `Sex`
  - `SibSp`
  - `Parch`
  - `Ticket`
  - `Fare`
  - `Embarked`
-   IDV columns with some Null values
  - `Age`
  - `Cabin`


```python
df_train.describe()
```





  <div id="df-de8ab9cf-4a30-4d59-aaf8-5e7bb5130b9c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-de8ab9cf-4a30-4d59-aaf8-5e7bb5130b9c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-de8ab9cf-4a30-4d59-aaf8-5e7bb5130b9c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-de8ab9cf-4a30-4d59-aaf8-5e7bb5130b9c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1a59052d-755a-45d8-b371-cdb944583b7e">
  <button class="colab-df-quickchart" onclick="quickchart('df-1a59052d-755a-45d8-b371-cdb944583b7e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1a59052d-755a-45d8-b371-cdb944583b7e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python

```

## Columns without any null values

### Pclass

> Ticket Class


```python
df_train.Pclass.sample(10)
```




    286    3
    353    3
    693    3
    152    3
    871    1
    183    2
    199    2
    162    3
    453    1
    455    3
    Name: Pclass, dtype: int64



What is the distribution of the class of tickets?


```python
pd.concat([df_train.Pclass.value_counts(),
           df_train.Pclass.value_counts(normalize=True)], axis = 1)
```





  <div id="df-47069900-ffc7-4176-be94-c1c457070345" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>0.551066</td>
    </tr>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>0.242424</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>0.206510</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-47069900-ffc7-4176-be94-c1c457070345')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-47069900-ffc7-4176-be94-c1c457070345 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-47069900-ffc7-4176-be94-c1c457070345');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e588c876-dc97-49eb-b1ea-ed58dd424f65">
  <button class="colab-df-quickchart" onclick="quickchart('df-e588c876-dc97-49eb-b1ea-ed58dd424f65')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e588c876-dc97-49eb-b1ea-ed58dd424f65 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
class_survivors = df_train.groupby('Pclass').agg({'Survived':['count', 'sum']})

class_survivors
```





  <div id="df-77a90030-9c02-4bdd-93cd-1096e9aca580" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Survived</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>119</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-77a90030-9c02-4bdd-93cd-1096e9aca580')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-77a90030-9c02-4bdd-93cd-1096e9aca580 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-77a90030-9c02-4bdd-93cd-1096e9aca580');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c9b6fae5-a74e-47b4-88a7-f6657c0c9db2">
  <button class="colab-df-quickchart" onclick="quickchart('df-c9b6fae5-a74e-47b4-88a7-f6657c0c9db2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c9b6fae5-a74e-47b4-88a7-f6657c0c9db2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_c96068e4-7db7-4543-a665-fdb3087030da">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('class_survivors')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c96068e4-7db7-4543-a665-fdb3087030da button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('class_survivors');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
class_survivors[('Survived', 'sum')]/class_survivors[('Survived', 'count')]
```




    Pclass
    1    0.629630
    2    0.472826
    3    0.242363
    dtype: float64



### Name


```python
df_train.Name.sample(10)
```




    5                                       Moran, Mr. James
    477                            Braund, Mr. Lewis Richard
    434                            Silvey, Mr. William Baird
    835                          Compton, Miss. Sara Rebecca
    862    Swift, Mrs. Frederick Joel (Margaret Welles Ba...
    709    Moubarek, Master. Halim Gonios ("William George")
    210                                       Ali, Mr. Ahmed
    409                                   Lefebre, Miss. Ida
    329                         Hippach, Miss. Jean Gertrude
    599         Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")
    Name: Name, dtype: object



**Note**
1. Title could be extracted from the name.
2. Is there some significance in survival context to names with a braces, an alternative name?
3. Are there important family names that can extracted to signify importance for survival?

### Sex


```python
df_train.Sex.sample(10)
```




    334    female
    29       male
    623      male
    616      male
    398      male
    0        male
    258    female
    388      male
    811      male
    405      male
    Name: Sex, dtype: object



What is the distribution of passenger sex?


```python
pd.concat([df_train.Sex.value_counts(),
           df_train.Sex.value_counts(normalize=True)], axis = 1)
```





  <div id="df-1421e154-b017-413a-b04f-30e2abd29d4c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>male</th>
      <td>577</td>
      <td>0.647587</td>
    </tr>
    <tr>
      <th>female</th>
      <td>314</td>
      <td>0.352413</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1421e154-b017-413a-b04f-30e2abd29d4c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1421e154-b017-413a-b04f-30e2abd29d4c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1421e154-b017-413a-b04f-30e2abd29d4c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f7d4e479-aa9b-418a-b869-ee6b6f690d9a">
  <button class="colab-df-quickchart" onclick="quickchart('df-f7d4e479-aa9b-418a-b869-ee6b6f690d9a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f7d4e479-aa9b-418a-b869-ee6b6f690d9a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train[df_train.Age.isna()].sample(10)
```





  <div id="df-c4be0724-d50d-4785-a290-1a0ec310104a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>229</th>
      <td>230</td>
      <td>0</td>
      <td>3</td>
      <td>Lefebre, Miss. Mathilde</td>
      <td>female</td>
      <td>NaN</td>
      <td>3</td>
      <td>1</td>
      <td>4133</td>
      <td>25.4667</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>656</th>
      <td>657</td>
      <td>0</td>
      <td>3</td>
      <td>Radeff, Mr. Alexander</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349223</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>475</th>
      <td>476</td>
      <td>0</td>
      <td>1</td>
      <td>Clifford, Mr. George Quincy</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>110465</td>
      <td>52.0000</td>
      <td>A14</td>
      <td>S</td>
    </tr>
    <tr>
      <th>180</th>
      <td>181</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Constance Gladys</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>760</th>
      <td>761</td>
      <td>0</td>
      <td>3</td>
      <td>Garfirth, Mr. John</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>358585</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>783</th>
      <td>784</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Mr. Andrew G</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>87</th>
      <td>88</td>
      <td>0</td>
      <td>3</td>
      <td>Slocovski, Mr. Selman Francis</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392086</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>55</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>Woolner, Mr. Hugh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>19947</td>
      <td>35.5000</td>
      <td>C52</td>
      <td>S</td>
    </tr>
    <tr>
      <th>65</th>
      <td>66</td>
      <td>1</td>
      <td>3</td>
      <td>Moubarek, Master. Gerios</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2661</td>
      <td>15.2458</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c4be0724-d50d-4785-a290-1a0ec310104a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c4be0724-d50d-4785-a290-1a0ec310104a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c4be0724-d50d-4785-a290-1a0ec310104a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ba74676b-8099-43da-84e0-eb015d376bdf">
  <button class="colab-df-quickchart" onclick="quickchart('df-ba74676b-8099-43da-84e0-eb015d376bdf')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ba74676b-8099-43da-84e0-eb015d376bdf button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### SibSp

> number of siblings / spouses aboard the Titanic


```python
df_train.SibSp.sample(10)
```




    170    0
    410    0
    533    0
    316    1
    7      3
    395    0
    635    0
    855    0
    844    0
    2      0
    Name: SibSp, dtype: int64




```python
pd.concat([df_train.SibSp.value_counts(),
           df_train.SibSp.value_counts(normalize=True)], axis = 1)
```





  <div id="df-37e7e4cd-a977-41b3-b29c-28f4b7bebfec" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>SibSp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>608</td>
      <td>0.682379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>209</td>
      <td>0.234568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>0.031425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>0.020202</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>0.017957</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>0.007856</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.005612</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-37e7e4cd-a977-41b3-b29c-28f4b7bebfec')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-37e7e4cd-a977-41b3-b29c-28f4b7bebfec button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-37e7e4cd-a977-41b3-b29c-28f4b7bebfec');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2b416fa7-250a-45c5-b42e-e6191440a7d4">
  <button class="colab-df-quickchart" onclick="quickchart('df-2b416fa7-250a-45c5-b42e-e6191440a7d4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2b416fa7-250a-45c5-b42e-e6191440a7d4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Note**

1. About 70% of the passengers have no siblings or spouses accompanying them.
2. If `SibSp` == 1, is it a spouse more likely then a sibling? If so, would they survive together or die together unlike Jack and Rose :D

### Parch

> Number of parents / children aboard the Titanic


```python
df_train.Parch.sample(10)
```




    658    0
    150    0
    530    1
    338    0
    834    0
    308    0
    866    0
    475    0
    45     0
    435    2
    Name: Parch, dtype: int64




```python
pd.concat([df_train.Parch.value_counts(),
           df_train.Parch.value_counts(normalize=True)], axis = 1)
```





  <div id="df-bf65b4a4-92e5-4633-a025-2778238dfed9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Parch</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>678</td>
      <td>0.760943</td>
    </tr>
    <tr>
      <th>1</th>
      <td>118</td>
      <td>0.132435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80</td>
      <td>0.089787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.005612</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.005612</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.004489</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.001122</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bf65b4a4-92e5-4633-a025-2778238dfed9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bf65b4a4-92e5-4633-a025-2778238dfed9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bf65b4a4-92e5-4633-a025-2778238dfed9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-98c7ae84-e0dd-453a-a119-d390b49e805d">
  <button class="colab-df-quickchart" onclick="quickchart('df-98c7ae84-e0dd-453a-a119-d390b49e805d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-98c7ae84-e0dd-453a-a119-d390b49e805d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### Ticket

> Ticket number



```python
df_train.Ticket.sample(10)
```




    527           PC 17483
    51          A/4. 39886
    485               4133
    755             250649
    464           A/S 2816
    404             315096
    776             383121
    95              374910
    594        SC/AH 29037
    157    SOTON/OQ 392090
    Name: Ticket, dtype: object




```python
pd.concat([df_train.Parch.value_counts(),
           df_train.Parch.value_counts(normalize=True)], axis = 1)
```





  <div id="df-e63ef177-ee9e-443b-9bcd-b1be17a94d3d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Parch</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>678</td>
      <td>0.760943</td>
    </tr>
    <tr>
      <th>1</th>
      <td>118</td>
      <td>0.132435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80</td>
      <td>0.089787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.005612</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.005612</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.004489</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.001122</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e63ef177-ee9e-443b-9bcd-b1be17a94d3d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e63ef177-ee9e-443b-9bcd-b1be17a94d3d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e63ef177-ee9e-443b-9bcd-b1be17a94d3d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f72a23f8-8ea4-4cda-b91a-1494a23b7818">
  <button class="colab-df-quickchart" onclick="quickchart('df-f72a23f8-8ea4-4cda-b91a-1494a23b7818')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f72a23f8-8ea4-4cda-b91a-1494a23b7818 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train[['Embarked','Pclass', 'Ticket', 'Cabin']].sample(10)
```





  <div id="df-67421aa5-0271-41f5-8be5-30f1162c723e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Pclass</th>
      <th>Ticket</th>
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>238</th>
      <td>S</td>
      <td>2</td>
      <td>28665</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>280</th>
      <td>Q</td>
      <td>3</td>
      <td>336439</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>533</th>
      <td>C</td>
      <td>3</td>
      <td>2668</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>493</th>
      <td>C</td>
      <td>1</td>
      <td>PC 17609</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>813</th>
      <td>S</td>
      <td>3</td>
      <td>347082</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>438</th>
      <td>S</td>
      <td>1</td>
      <td>19950</td>
      <td>C23 C25 C27</td>
    </tr>
    <tr>
      <th>321</th>
      <td>S</td>
      <td>3</td>
      <td>349219</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>790</th>
      <td>Q</td>
      <td>3</td>
      <td>12460</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>S</td>
      <td>1</td>
      <td>19972</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>672</th>
      <td>S</td>
      <td>2</td>
      <td>C.A. 24580</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-67421aa5-0271-41f5-8be5-30f1162c723e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-67421aa5-0271-41f5-8be5-30f1162c723e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-67421aa5-0271-41f5-8be5-30f1162c723e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-04c6089f-ab30-499a-afc5-59d19be9b9d1">
  <button class="colab-df-quickchart" onclick="quickchart('df-04c6089f-ab30-499a-afc5-59d19be9b9d1')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-04c6089f-ab30-499a-afc5-59d19be9b9d1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Note**

1. `Ticket` could help fill in `Cabin` data by way of signalling it?
2. It could also be dependent on `Pclass` and `Embarked`

### Fare

> Passenger fare



```python
df_train.Fare.sample(10)
```




    511     8.0500
    753     7.8958
    460    26.5500
    9      30.0708
    99     26.0000
    347    16.1000
    827    37.0042
    351    35.0000
    177    28.7125
    825     6.9500
    Name: Fare, dtype: float64




```python
df_train.groupby(['Pclass', 'Embarked']).agg({'Fare':'mean'})
```





  <div id="df-40d390c8-f304-47cc-b8db-9047976f19a3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th>Embarked</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>C</th>
      <td>104.718529</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>S</th>
      <td>70.364862</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>C</th>
      <td>25.358335</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>12.350000</td>
    </tr>
    <tr>
      <th>S</th>
      <td>20.327439</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">3</th>
      <th>C</th>
      <td>11.214083</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>11.183393</td>
    </tr>
    <tr>
      <th>S</th>
      <td>14.644083</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-40d390c8-f304-47cc-b8db-9047976f19a3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-40d390c8-f304-47cc-b8db-9047976f19a3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-40d390c8-f304-47cc-b8db-9047976f19a3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-44c29a7c-7e72-46fc-b91c-c757b8517925">
  <button class="colab-df-quickchart" onclick="quickchart('df-44c29a7c-7e72-46fc-b91c-c757b8517925')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-44c29a7c-7e72-46fc-b91c-c757b8517925 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Note**

1. As expected the class order in terms of fare is 1, 2 and then 3 but where the passenger embarks also has an impact on the fare for the travel.
2. However, no new information in our context could be extracted from this column?

### Embarked

> Port of Embarkation; C -Cherbourg, Q - Queenstown, S - Southampton


```python
pd.concat([df_train.Embarked.value_counts(),
           df_train.Embarked.value_counts(normalize = True)], axis = 1)
```





  <div id="df-1ef8442d-553e-4631-a784-553737726cf9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Embarked</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>S</th>
      <td>644</td>
      <td>0.724409</td>
    </tr>
    <tr>
      <th>C</th>
      <td>168</td>
      <td>0.188976</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>77</td>
      <td>0.086614</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1ef8442d-553e-4631-a784-553737726cf9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1ef8442d-553e-4631-a784-553737726cf9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1ef8442d-553e-4631-a784-553737726cf9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-421bb55a-0685-4a86-881c-94ebd89f5020">
  <button class="colab-df-quickchart" onclick="quickchart('df-421bb55a-0685-4a86-881c-94ebd89f5020')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-421bb55a-0685-4a86-881c-94ebd89f5020 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Note**

1. Don't see relevance in survival but could be useful for other columns?


## Columns with null values

### Cabin

> Cabin number


```python
df_train.Cabin.sample(10)
```




    57     NaN
    72     NaN
    348    NaN
    631    NaN
    422    NaN
    689     B5
    734    NaN
    301    NaN
    615    NaN
    754    NaN
    Name: Cabin, dtype: object




```python
pd.concat([df_train.Cabin.isna().value_counts(),
           df_train.Cabin.isna().value_counts(normalize=True)], axis = 1)
```





  <div id="df-404833db-0eba-4932-912f-f47fc977913f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>proportion</th>
    </tr>
    <tr>
      <th>Cabin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True</th>
      <td>687</td>
      <td>0.771044</td>
    </tr>
    <tr>
      <th>False</th>
      <td>204</td>
      <td>0.228956</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-404833db-0eba-4932-912f-f47fc977913f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-404833db-0eba-4932-912f-f47fc977913f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-404833db-0eba-4932-912f-f47fc977913f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a093471d-8836-4601-8c9d-7223ad1d7006">
  <button class="colab-df-quickchart" onclick="quickchart('df-a093471d-8836-4601-8c9d-7223ad1d7006')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a093471d-8836-4601-8c9d-7223ad1d7006 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train[df_train.Cabin.notnull()].groupby('Pclass')['Cabin'].agg(' '.join)
```




    Pclass
    1    C85 C123 E46 C103 A6 C23 C25 C27 B78 D33 B30 C...
    2    D56 F33 E101 F2 F4 F2 D E101 D F2 F33 D F33 F4...
    3    G6 F G73 F E69 G6 G6 G6 E10 F G63 F G73 E121 F...
    Name: Cabin, dtype: object



**Note**

1. Cabins could hold key information relating to who survived by virtue of how accessible they are to the deck.
2. However, since large portions of the information is unavailable (~78%) It cannot be used unless encoded through any other variable.
3. Class could signify some cabins in terms of allocation, the definition of class is separation, but what use is the attribute if class alone can define it. Needs exploration!

### Age


```python
df_train.Age.sample(10)
```




    354     NaN
    210    24.0
    711     NaN
    11     58.0
    395    22.0
    629     NaN
    722    34.0
    193     3.0
    300     NaN
    794    25.0
    Name: Age, dtype: float64



# Data Preprocessing

## Feature Engineering

### Title


```python
df_train['Name'].apply(lambda x : x.split('.')[0].split(' ')[-1].strip()).value_counts()
```




    Name
    Mr          517
    Miss        182
    Mrs         125
    Master       40
    Dr            7
    Rev           6
    Mlle          2
    Major         2
    Col           2
    Countess      1
    Capt          1
    Ms            1
    Sir           1
    Lady          1
    Mme           1
    Don           1
    Jonkheer      1
    Name: count, dtype: int64




```python
df_train['Title'] = df_train['Name'].apply(lambda x : x.split('.')[0].split(' ')[-1].strip())
df_test['Title'] = df_test['Name'].apply(lambda x : x.split('.')[0].split(' ')[-1].strip())

df_train.sample(10)
```





  <div id="df-b6345dc0-0ebb-4e70-a045-cf2d71db8cc0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>637</th>
      <td>638</td>
      <td>0</td>
      <td>2</td>
      <td>Collyer, Mr. Harvey</td>
      <td>male</td>
      <td>31.0</td>
      <td>1</td>
      <td>1</td>
      <td>C.A. 31921</td>
      <td>26.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>1</td>
      <td>3</td>
      <td>Mullens, Miss. Katherine "Katie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>35852</td>
      <td>7.7333</td>
      <td>NaN</td>
      <td>Q</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0000</td>
      <td>A36</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>324</th>
      <td>325</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. George John Jr</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>835</th>
      <td>836</td>
      <td>1</td>
      <td>1</td>
      <td>Compton, Miss. Sara Rebecca</td>
      <td>female</td>
      <td>39.0</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17756</td>
      <td>83.1583</td>
      <td>E49</td>
      <td>C</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>513</th>
      <td>514</td>
      <td>1</td>
      <td>1</td>
      <td>Rothschild, Mrs. Martin (Elizabeth L. Barrett)</td>
      <td>female</td>
      <td>54.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17603</td>
      <td>59.4000</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>291</th>
      <td>292</td>
      <td>1</td>
      <td>1</td>
      <td>Bishop, Mrs. Dickinson H (Helen Walton)</td>
      <td>female</td>
      <td>19.0</td>
      <td>1</td>
      <td>0</td>
      <td>11967</td>
      <td>91.0792</td>
      <td>B49</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>767</th>
      <td>768</td>
      <td>0</td>
      <td>3</td>
      <td>Mangan, Miss. Mary</td>
      <td>female</td>
      <td>30.5</td>
      <td>0</td>
      <td>0</td>
      <td>364850</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>851</th>
      <td>852</td>
      <td>0</td>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>117</th>
      <td>118</td>
      <td>0</td>
      <td>2</td>
      <td>Turpin, Mr. William John Robert</td>
      <td>male</td>
      <td>29.0</td>
      <td>1</td>
      <td>0</td>
      <td>11668</td>
      <td>21.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b6345dc0-0ebb-4e70-a045-cf2d71db8cc0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b6345dc0-0ebb-4e70-a045-cf2d71db8cc0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b6345dc0-0ebb-4e70-a045-cf2d71db8cc0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bbeaec8b-93f2-4a9c-8c54-752e460f39eb">
  <button class="colab-df-quickchart" onclick="quickchart('df-bbeaec8b-93f2-4a9c-8c54-752e460f39eb')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bbeaec8b-93f2-4a9c-8c54-752e460f39eb button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train.groupby("Title").agg({'Age':['count', 'mean']}).sort_values(('Age', 'mean'), ascending = False)
```





  <div id="df-934c3bb7-8eee-4269-830d-825d5355a363" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Age</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>1</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>2</td>
      <td>58.000000</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>1</td>
      <td>49.000000</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>2</td>
      <td>48.500000</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>6</td>
      <td>43.166667</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>6</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>1</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>1</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>108</td>
      <td>35.898148</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>398</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>28.000000</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>146</td>
      <td>21.773973</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>36</td>
      <td>4.574167</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-934c3bb7-8eee-4269-830d-825d5355a363')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-934c3bb7-8eee-4269-830d-825d5355a363 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-934c3bb7-8eee-4269-830d-825d5355a363');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a930b550-1205-4971-a398-d04dcf57c756">
  <button class="colab-df-quickchart" onclick="quickchart('df-a930b550-1205-4971-a398-d04dcf57c756')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a930b550-1205-4971-a398-d04dcf57c756 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train[df_train.Age.isna()].Title.value_counts()
```




    Title
    Mr        119
    Miss       36
    Mrs        17
    Master      4
    Dr          1
    Name: count, dtype: int64




```python
df_test[df_test.Age.isna()].Title.value_counts()
```




    Title
    Mr        57
    Miss      14
    Mrs       10
    Master     4
    Ms         1
    Name: count, dtype: int64



**Note** Use the average age by `Title` to encode missing value in the `Age` column. This method is preferred over mean encoding on the overall dataset.

### Title mean age

> Average age of all the passengers with a specific title.
**Note:** Could be useful for age encoding


```python
df_title_mean_age = df_train.groupby("Title").agg({'Age':'mean'}).reset_index().rename({'Age': 'Title_Mean_Age'}, axis = 1)

df_title_mean_age.sample(5)
```





  <div id="df-945b6bfe-3c44-429b-842f-b24e13ee5a25" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Title_Mean_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>Sir</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Countess</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Major</td>
      <td>48.5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Capt</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lady</td>
      <td>48.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-945b6bfe-3c44-429b-842f-b24e13ee5a25')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-945b6bfe-3c44-429b-842f-b24e13ee5a25 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-945b6bfe-3c44-429b-842f-b24e13ee5a25');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-88d98c26-4857-476d-b036-8b048f415327">
  <button class="colab-df-quickchart" onclick="quickchart('df-88d98c26-4857-476d-b036-8b048f415327')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-88d98c26-4857-476d-b036-8b048f415327 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train = df_train.merge(df_title_mean_age, how = 'left', on = 'Title')
df_test = df_test.merge(df_title_mean_age, how = 'left', on = 'Title')

df_train.sample(10)
```





  <div id="df-df0aa658-d6dd-4dba-9ad3-1754d223db5f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>Title_Mean_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>634</th>
      <td>635</td>
      <td>0</td>
      <td>3</td>
      <td>Skoog, Miss. Mabel</td>
      <td>female</td>
      <td>9.0</td>
      <td>3</td>
      <td>2</td>
      <td>347088</td>
      <td>27.9000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
      <td>21.773973</td>
    </tr>
    <tr>
      <th>232</th>
      <td>233</td>
      <td>0</td>
      <td>2</td>
      <td>Sjostedt, Mr. Ernst Adolf</td>
      <td>male</td>
      <td>59.0</td>
      <td>0</td>
      <td>0</td>
      <td>237442</td>
      <td>13.5000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>721</th>
      <td>722</td>
      <td>0</td>
      <td>3</td>
      <td>Jensen, Mr. Svend Lauritz</td>
      <td>male</td>
      <td>17.0</td>
      <td>1</td>
      <td>0</td>
      <td>350048</td>
      <td>7.0542</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>266</th>
      <td>267</td>
      <td>0</td>
      <td>3</td>
      <td>Panula, Mr. Ernesti Arvid</td>
      <td>male</td>
      <td>16.0</td>
      <td>4</td>
      <td>1</td>
      <td>3101295</td>
      <td>39.6875</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>650</th>
      <td>651</td>
      <td>0</td>
      <td>3</td>
      <td>Mitkoff, Mr. Mito</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>349221</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>524</th>
      <td>525</td>
      <td>0</td>
      <td>3</td>
      <td>Kassem, Mr. Fared</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2700</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
      <td>Miss</td>
      <td>21.773973</td>
    </tr>
    <tr>
      <th>116</th>
      <td>117</td>
      <td>0</td>
      <td>3</td>
      <td>Connors, Mr. Patrick</td>
      <td>male</td>
      <td>70.5</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
    <tr>
      <th>865</th>
      <td>866</td>
      <td>1</td>
      <td>2</td>
      <td>Bystrom, Mrs. (Karolina)</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>236852</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
      <td>35.898148</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-df0aa658-d6dd-4dba-9ad3-1754d223db5f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-df0aa658-d6dd-4dba-9ad3-1754d223db5f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-df0aa658-d6dd-4dba-9ad3-1754d223db5f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7d8b8a50-c98c-4757-a8a0-75be0a108694">
  <button class="colab-df-quickchart" onclick="quickchart('df-7d8b8a50-c98c-4757-a8a0-75be0a108694')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7d8b8a50-c98c-4757-a8a0-75be0a108694 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train[df_train.Age.isna()].sample()
```





  <div id="df-63acaa1f-bf65-455a-ae7e-7f62384ff24a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>Title_Mean_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>776</th>
      <td>777</td>
      <td>0</td>
      <td>3</td>
      <td>Tobin, Mr. Roger</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>383121</td>
      <td>7.75</td>
      <td>F38</td>
      <td>Q</td>
      <td>Mr</td>
      <td>32.36809</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-63acaa1f-bf65-455a-ae7e-7f62384ff24a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-63acaa1f-bf65-455a-ae7e-7f62384ff24a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-63acaa1f-bf65-455a-ae7e-7f62384ff24a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>





```python
df_train['Age'] = df_train['Age'].fillna(df_train['Title_Mean_Age'])
df_test['Age'] = df_test['Age'].fillna(df_test['Title_Mean_Age'])

df_test.shape
```




    (418, 13)



### Alone

> Flag to check if no siblings or spouses or children or parents are accompanying. Note: This is ignorant of friends or accomplices


```python
print(f"The number of passengers who are travelling alone is \
{df_train[(df_train.SibSp==0)&(df_train.Parch==0)].shape[0]} \
which is about {df_train[(df_train.SibSp==0)&(df_train.Parch==0)].shape[0]/df_train.shape[0]*100:.2f}% of the total passengers.")
```

    The number of passengers who are travelling alone is 537 which is about 60.27% of the total passengers.



```python
df_train['Alone'] = (df_train.SibSp==0)&(df_train.Parch==0)
df_test['Alone'] = (df_test.SibSp==0)&(df_test.Parch==0)

df_train.sample(10)
```





  <div id="df-7ee382c7-f780-43eb-8ea1-5f6008257675" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>Title_Mean_Age</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>573</th>
      <td>574</td>
      <td>1</td>
      <td>3</td>
      <td>Kelly, Miss. Mary</td>
      <td>female</td>
      <td>21.773973</td>
      <td>0</td>
      <td>0</td>
      <td>14312</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>Miss</td>
      <td>21.773973</td>
      <td>True</td>
    </tr>
    <tr>
      <th>495</th>
      <td>496</td>
      <td>0</td>
      <td>3</td>
      <td>Yousseff, Mr. Gerious</td>
      <td>male</td>
      <td>32.368090</td>
      <td>0</td>
      <td>0</td>
      <td>2627</td>
      <td>14.4583</td>
      <td>NaN</td>
      <td>C</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>True</td>
    </tr>
    <tr>
      <th>472</th>
      <td>473</td>
      <td>1</td>
      <td>2</td>
      <td>West, Mrs. Edwy Arthur (Ada Mary Worth)</td>
      <td>female</td>
      <td>33.000000</td>
      <td>1</td>
      <td>2</td>
      <td>C.A. 34651</td>
      <td>27.7500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mrs</td>
      <td>35.898148</td>
      <td>False</td>
    </tr>
    <tr>
      <th>741</th>
      <td>742</td>
      <td>0</td>
      <td>1</td>
      <td>Cavendish, Mr. Tyrell William</td>
      <td>male</td>
      <td>36.000000</td>
      <td>1</td>
      <td>0</td>
      <td>19877</td>
      <td>78.8500</td>
      <td>C46</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>False</td>
    </tr>
    <tr>
      <th>345</th>
      <td>346</td>
      <td>1</td>
      <td>2</td>
      <td>Brown, Miss. Amelia "Mildred"</td>
      <td>female</td>
      <td>24.000000</td>
      <td>0</td>
      <td>0</td>
      <td>248733</td>
      <td>13.0000</td>
      <td>F33</td>
      <td>S</td>
      <td>Miss</td>
      <td>21.773973</td>
      <td>True</td>
    </tr>
    <tr>
      <th>566</th>
      <td>567</td>
      <td>0</td>
      <td>3</td>
      <td>Stoytcheff, Mr. Ilia</td>
      <td>male</td>
      <td>19.000000</td>
      <td>0</td>
      <td>0</td>
      <td>349205</td>
      <td>7.8958</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>True</td>
    </tr>
    <tr>
      <th>785</th>
      <td>786</td>
      <td>0</td>
      <td>3</td>
      <td>Harmer, Mr. Abraham (David Lishin)</td>
      <td>male</td>
      <td>25.000000</td>
      <td>0</td>
      <td>0</td>
      <td>374887</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>True</td>
    </tr>
    <tr>
      <th>174</th>
      <td>175</td>
      <td>0</td>
      <td>1</td>
      <td>Smith, Mr. James Clinch</td>
      <td>male</td>
      <td>56.000000</td>
      <td>0</td>
      <td>0</td>
      <td>17764</td>
      <td>30.6958</td>
      <td>A7</td>
      <td>C</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>True</td>
    </tr>
    <tr>
      <th>474</th>
      <td>475</td>
      <td>0</td>
      <td>3</td>
      <td>Strandberg, Miss. Ida Sofia</td>
      <td>female</td>
      <td>22.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7553</td>
      <td>9.8375</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
      <td>21.773973</td>
      <td>True</td>
    </tr>
    <tr>
      <th>512</th>
      <td>513</td>
      <td>1</td>
      <td>1</td>
      <td>McGough, Mr. James Robert</td>
      <td>male</td>
      <td>36.000000</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17473</td>
      <td>26.2875</td>
      <td>E25</td>
      <td>S</td>
      <td>Mr</td>
      <td>32.368090</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7ee382c7-f780-43eb-8ea1-5f6008257675')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7ee382c7-f780-43eb-8ea1-5f6008257675 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7ee382c7-f780-43eb-8ea1-5f6008257675');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-56de399b-ccec-4681-875a-795352f5ee3b">
  <button class="colab-df-quickchart" onclick="quickchart('df-56de399b-ccec-4681-875a-795352f5ee3b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-56de399b-ccec-4681-875a-795352f5ee3b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train.Alone.value_counts()
```




    Alone
    True     537
    False    354
    Name: count, dtype: int64




```python
df_train.Survived.value_counts()
```




    Survived
    0    549
    1    342
    Name: count, dtype: int64




```python
df_train.groupby('Alone').agg({'Survived':'sum'}).rename({'Survived':'count'}, axis = 1)
```





  <div id="df-d1b6d5ae-0ca5-42f5-b066-3ded1b05b8a9" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Alone</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>179</td>
    </tr>
    <tr>
      <th>True</th>
      <td>163</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d1b6d5ae-0ca5-42f5-b066-3ded1b05b8a9')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d1b6d5ae-0ca5-42f5-b066-3ded1b05b8a9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d1b6d5ae-0ca5-42f5-b066-3ded1b05b8a9');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a87a3181-0836-4559-8c26-ffb46f185480">
  <button class="colab-df-quickchart" onclick="quickchart('df-a87a3181-0836-4559-8c26-ffb46f185480')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a87a3181-0836-4559-8c26-ffb46f185480 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
pd.DataFrame(df_train.Alone.value_counts())
```





  <div id="df-94f0b24f-7383-4750-9555-4d208c3d567f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Alone</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True</th>
      <td>537</td>
    </tr>
    <tr>
      <th>False</th>
      <td>354</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-94f0b24f-7383-4750-9555-4d208c3d567f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-94f0b24f-7383-4750-9555-4d208c3d567f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-94f0b24f-7383-4750-9555-4d208c3d567f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a0015a90-ad7d-4082-b9f5-ac3208e1d7ce">
  <button class="colab-df-quickchart" onclick="quickchart('df-a0015a90-ad7d-4082-b9f5-ac3208e1d7ce')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a0015a90-ad7d-4082-b9f5-ac3208e1d7ce button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_train.groupby('Alone').agg({'Survived':'sum'}).rename({'Survived':'count'}, axis = 1)/pd.DataFrame(df_train.Alone.value_counts())
```





  <div id="df-0be27aae-b1d9-46de-9ce0-3efafcd8a146" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>Alone</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>0.505650</td>
    </tr>
    <tr>
      <th>True</th>
      <td>0.303538</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0be27aae-b1d9-46de-9ce0-3efafcd8a146')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0be27aae-b1d9-46de-9ce0-3efafcd8a146 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0be27aae-b1d9-46de-9ce0-3efafcd8a146');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f906e81f-acce-4cd1-9b97-b2fc40ef414d">
  <button class="colab-df-quickchart" onclick="quickchart('df-f906e81f-acce-4cd1-9b97-b2fc40ef414d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f906e81f-acce-4cd1-9b97-b2fc40ef414d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Note**

1. This could be an important feature as the survival rate drops by 20% at least for a passenger who is travelling alone!

### Age group
Use the Age column after filling null values with `Title mean age`


```python
df_test.Age.info()
```

    <class 'pandas.core.series.Series'>
    RangeIndex: 418 entries, 0 to 417
    Series name: Age
    Non-Null Count  Dtype  
    --------------  -----  
    418 non-null    float64
    dtypes: float64(1)
    memory usage: 3.4 KB



```python
# Define bins for the age ranges according to biological markers
bins = [0, 6, 13, 20, 36, 56, 76, float('inf')]  # float('inf') for ages above 75

# Labels for the age groups
labels = ['0-5', '6-12', '13-19', '20-35', '36-55', '56-75', '76+']

# Create age categories
df_train['Age_Group'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=False)
df_test['Age_Group'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=False)
df_train.Age_Group.value_counts()
```




    Age_Group
    20-35    505
    36-55    179
    13-19     95
    0-5       48
    56-75     38
    6-12      25
    76+        1
    Name: count, dtype: int64




```python
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'Alone', 'Age_Group']
cat_features = ['Sex', 'Title', 'Alone', 'Age_Group']

df_train[features].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column     Non-Null Count  Dtype   
    ---  ------     --------------  -----   
     0   Pclass     891 non-null    int64   
     1   Sex        891 non-null    object  
     2   SibSp      891 non-null    int64   
     3   Parch      891 non-null    int64   
     4   Title      891 non-null    object  
     5   Alone      891 non-null    bool    
     6   Age_Group  891 non-null    category
    dtypes: bool(1), category(1), int64(3), object(2)
    memory usage: 37.0+ KB


### Family Size


```python

```

## Encoding


```python
# one hot encoding categorical columns

from sklearn.preprocessing import OneHotEncoder
```


```python
encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
```


```python
encoder.fit(df_train[cat_features])
train_encoded = encoder.transform(df_train[cat_features])
test_encoded = encoder.transform(df_test[cat_features])
# Convert encoded data back to DataFrame
df_train_encoded = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out())
df_test_encoded = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out())
```


```python
df_train_encoded.columns
```




    Index(['Sex_female', 'Sex_male', 'Title_Capt', 'Title_Col', 'Title_Countess',
           'Title_Don', 'Title_Dr', 'Title_Jonkheer', 'Title_Lady', 'Title_Major',
           'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme', 'Title_Mr',
           'Title_Mrs', 'Title_Ms', 'Title_Rev', 'Title_Sir', 'Alone_False',
           'Alone_True', 'Age_Group_0-5', 'Age_Group_13-19', 'Age_Group_20-35',
           'Age_Group_36-55', 'Age_Group_56-75', 'Age_Group_6-12',
           'Age_Group_76+'],
          dtype='object')




```python
df_train_encoded['Survived'] = df_train['Survived']

df_train_encoded.columns
```




    Index(['Sex_female', 'Sex_male', 'Title_Capt', 'Title_Col', 'Title_Countess',
           'Title_Don', 'Title_Dr', 'Title_Jonkheer', 'Title_Lady', 'Title_Major',
           'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme', 'Title_Mr',
           'Title_Mrs', 'Title_Ms', 'Title_Rev', 'Title_Sir', 'Alone_False',
           'Alone_True', 'Age_Group_0-5', 'Age_Group_13-19', 'Age_Group_20-35',
           'Age_Group_36-55', 'Age_Group_56-75', 'Age_Group_6-12', 'Age_Group_76+',
           'Survived'],
          dtype='object')




```python
df_train_encoded.sample(10)
```





  <div id="df-faca711b-5f6b-4aa6-bd1b-402bf6523e86" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Title_Capt</th>
      <th>Title_Col</th>
      <th>Title_Countess</th>
      <th>Title_Don</th>
      <th>Title_Dr</th>
      <th>Title_Jonkheer</th>
      <th>Title_Lady</th>
      <th>Title_Major</th>
      <th>...</th>
      <th>Alone_False</th>
      <th>Alone_True</th>
      <th>Age_Group_0-5</th>
      <th>Age_Group_13-19</th>
      <th>Age_Group_20-35</th>
      <th>Age_Group_36-55</th>
      <th>Age_Group_56-75</th>
      <th>Age_Group_6-12</th>
      <th>Age_Group_76+</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>638</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>510</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>393</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>857</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>189</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>816</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 29 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-faca711b-5f6b-4aa6-bd1b-402bf6523e86')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-faca711b-5f6b-4aa6-bd1b-402bf6523e86 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-faca711b-5f6b-4aa6-bd1b-402bf6523e86');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4f765204-60f0-4037-9a03-0a7a28494a0e">
  <button class="colab-df-quickchart" onclick="quickchart('df-4f765204-60f0-4037-9a03-0a7a28494a0e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4f765204-60f0-4037-9a03-0a7a28494a0e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# Model Building

## Auto-ML using PyCaret


```python
from pycaret.classification import setup, compare_models

# Assuming 'data' is your DataFrame and 'target' is the name of the target column
setup_data = setup(data=df_train_encoded, target='Survived', fold = 3, session_id=123)

```


<style type="text/css">
#T_db3b8_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_db3b8" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_db3b8_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_db3b8_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_db3b8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_db3b8_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_db3b8_row0_col1" class="data row0 col1" >123</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_db3b8_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_db3b8_row1_col1" class="data row1 col1" >Survived</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_db3b8_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_db3b8_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_db3b8_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_db3b8_row3_col1" class="data row3 col1" >(891, 29)</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_db3b8_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_db3b8_row4_col1" class="data row4 col1" >(891, 29)</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_db3b8_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_db3b8_row5_col1" class="data row5 col1" >(623, 29)</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_db3b8_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_db3b8_row6_col1" class="data row6 col1" >(268, 29)</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_db3b8_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_db3b8_row7_col1" class="data row7 col1" >28</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_db3b8_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_db3b8_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_db3b8_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_db3b8_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_db3b8_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_db3b8_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_db3b8_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_db3b8_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_db3b8_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_db3b8_row12_col1" class="data row12 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_db3b8_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_db3b8_row13_col1" class="data row13 col1" >3</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_db3b8_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_db3b8_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_db3b8_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_db3b8_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_db3b8_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_db3b8_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_db3b8_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_db3b8_row17_col1" class="data row17 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_db3b8_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_db3b8_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_db3b8_row18_col1" class="data row18 col1" >2335</td>
    </tr>
  </tbody>
</table>




```python
best_model = compare_models()
```






<style type="text/css">
#T_b1983 th {
  text-align: left;
}
#T_b1983_row0_col0, #T_b1983_row0_col2, #T_b1983_row0_col4, #T_b1983_row1_col0, #T_b1983_row1_col2, #T_b1983_row1_col3, #T_b1983_row1_col5, #T_b1983_row1_col6, #T_b1983_row1_col7, #T_b1983_row2_col0, #T_b1983_row2_col1, #T_b1983_row2_col2, #T_b1983_row2_col3, #T_b1983_row2_col4, #T_b1983_row2_col5, #T_b1983_row2_col6, #T_b1983_row2_col7, #T_b1983_row3_col0, #T_b1983_row3_col1, #T_b1983_row3_col2, #T_b1983_row3_col3, #T_b1983_row3_col4, #T_b1983_row3_col5, #T_b1983_row3_col6, #T_b1983_row3_col7, #T_b1983_row4_col0, #T_b1983_row4_col1, #T_b1983_row4_col2, #T_b1983_row4_col3, #T_b1983_row4_col4, #T_b1983_row4_col5, #T_b1983_row4_col6, #T_b1983_row4_col7, #T_b1983_row5_col0, #T_b1983_row5_col1, #T_b1983_row5_col2, #T_b1983_row5_col3, #T_b1983_row5_col4, #T_b1983_row5_col5, #T_b1983_row5_col6, #T_b1983_row5_col7, #T_b1983_row6_col0, #T_b1983_row6_col1, #T_b1983_row6_col2, #T_b1983_row6_col3, #T_b1983_row6_col4, #T_b1983_row6_col5, #T_b1983_row6_col6, #T_b1983_row6_col7, #T_b1983_row7_col0, #T_b1983_row7_col1, #T_b1983_row7_col2, #T_b1983_row7_col3, #T_b1983_row7_col4, #T_b1983_row7_col5, #T_b1983_row7_col6, #T_b1983_row7_col7, #T_b1983_row8_col0, #T_b1983_row8_col1, #T_b1983_row8_col2, #T_b1983_row8_col3, #T_b1983_row8_col4, #T_b1983_row8_col5, #T_b1983_row8_col6, #T_b1983_row8_col7, #T_b1983_row9_col0, #T_b1983_row9_col1, #T_b1983_row9_col3, #T_b1983_row9_col4, #T_b1983_row9_col5, #T_b1983_row9_col6, #T_b1983_row9_col7, #T_b1983_row10_col0, #T_b1983_row10_col1, #T_b1983_row10_col2, #T_b1983_row10_col3, #T_b1983_row10_col4, #T_b1983_row10_col5, #T_b1983_row10_col6, #T_b1983_row10_col7, #T_b1983_row11_col0, #T_b1983_row11_col1, #T_b1983_row11_col2, #T_b1983_row11_col3, #T_b1983_row11_col4, #T_b1983_row11_col5, #T_b1983_row11_col6, #T_b1983_row11_col7, #T_b1983_row12_col0, #T_b1983_row12_col1, #T_b1983_row12_col2, #T_b1983_row12_col3, #T_b1983_row12_col4, #T_b1983_row12_col5, #T_b1983_row12_col6, #T_b1983_row12_col7, #T_b1983_row13_col0, #T_b1983_row13_col1, #T_b1983_row13_col2, #T_b1983_row13_col3, #T_b1983_row13_col4, #T_b1983_row13_col5, #T_b1983_row13_col6, #T_b1983_row13_col7, #T_b1983_row14_col0, #T_b1983_row14_col1, #T_b1983_row14_col2, #T_b1983_row14_col3, #T_b1983_row14_col4, #T_b1983_row14_col5, #T_b1983_row14_col6, #T_b1983_row14_col7 {
  text-align: left;
}
#T_b1983_row0_col1, #T_b1983_row0_col3, #T_b1983_row0_col5, #T_b1983_row0_col6, #T_b1983_row0_col7, #T_b1983_row1_col1, #T_b1983_row1_col4, #T_b1983_row9_col2 {
  text-align: left;
  background-color: yellow;
}
#T_b1983_row0_col8, #T_b1983_row1_col8, #T_b1983_row2_col8, #T_b1983_row3_col8, #T_b1983_row4_col8, #T_b1983_row5_col8, #T_b1983_row6_col8, #T_b1983_row7_col8, #T_b1983_row8_col8, #T_b1983_row9_col8, #T_b1983_row11_col8, #T_b1983_row12_col8, #T_b1983_row14_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_b1983_row10_col8, #T_b1983_row13_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_b1983" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b1983_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_b1983_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_b1983_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_b1983_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_b1983_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_b1983_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_b1983_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_b1983_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_b1983_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b1983_level0_row0" class="row_heading level0 row0" >lr</th>
      <td id="T_b1983_row0_col0" class="data row0 col0" >Logistic Regression</td>
      <td id="T_b1983_row0_col1" class="data row0 col1" >0.8025</td>
      <td id="T_b1983_row0_col2" class="data row0 col2" >0.7862</td>
      <td id="T_b1983_row0_col3" class="data row0 col3" >0.7446</td>
      <td id="T_b1983_row0_col4" class="data row0 col4" >0.7421</td>
      <td id="T_b1983_row0_col5" class="data row0 col5" >0.7433</td>
      <td id="T_b1983_row0_col6" class="data row0 col6" >0.5829</td>
      <td id="T_b1983_row0_col7" class="data row0 col7" >0.5829</td>
      <td id="T_b1983_row0_col8" class="data row0 col8" >1.7533</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row1" class="row_heading level0 row1" >ridge</th>
      <td id="T_b1983_row1_col0" class="data row1 col0" >Ridge Classifier</td>
      <td id="T_b1983_row1_col1" class="data row1 col1" >0.8025</td>
      <td id="T_b1983_row1_col2" class="data row1 col2" >0.7878</td>
      <td id="T_b1983_row1_col3" class="data row1 col3" >0.7363</td>
      <td id="T_b1983_row1_col4" class="data row1 col4" >0.7463</td>
      <td id="T_b1983_row1_col5" class="data row1 col5" >0.7411</td>
      <td id="T_b1983_row1_col6" class="data row1 col6" >0.5815</td>
      <td id="T_b1983_row1_col7" class="data row1 col7" >0.5817</td>
      <td id="T_b1983_row1_col8" class="data row1 col8" >0.0500</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row2" class="row_heading level0 row2" >lda</th>
      <td id="T_b1983_row2_col0" class="data row2 col0" >Linear Discriminant Analysis</td>
      <td id="T_b1983_row2_col1" class="data row2 col1" >0.8009</td>
      <td id="T_b1983_row2_col2" class="data row2 col2" >0.7820</td>
      <td id="T_b1983_row2_col3" class="data row2 col3" >0.7321</td>
      <td id="T_b1983_row2_col4" class="data row2 col4" >0.7450</td>
      <td id="T_b1983_row2_col5" class="data row2 col5" >0.7383</td>
      <td id="T_b1983_row2_col6" class="data row2 col6" >0.5777</td>
      <td id="T_b1983_row2_col7" class="data row2 col7" >0.5779</td>
      <td id="T_b1983_row2_col8" class="data row2 col8" >0.0767</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row3" class="row_heading level0 row3" >ada</th>
      <td id="T_b1983_row3_col0" class="data row3 col0" >Ada Boost Classifier</td>
      <td id="T_b1983_row3_col1" class="data row3 col1" >0.7945</td>
      <td id="T_b1983_row3_col2" class="data row3 col2" >0.7753</td>
      <td id="T_b1983_row3_col3" class="data row3 col3" >0.7279</td>
      <td id="T_b1983_row3_col4" class="data row3 col4" >0.7343</td>
      <td id="T_b1983_row3_col5" class="data row3 col5" >0.7309</td>
      <td id="T_b1983_row3_col6" class="data row3 col6" >0.5647</td>
      <td id="T_b1983_row3_col7" class="data row3 col7" >0.5649</td>
      <td id="T_b1983_row3_col8" class="data row3 col8" >0.1600</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row4" class="row_heading level0 row4" >lightgbm</th>
      <td id="T_b1983_row4_col0" class="data row4 col0" >Light Gradient Boosting Machine</td>
      <td id="T_b1983_row4_col1" class="data row4 col1" >0.7881</td>
      <td id="T_b1983_row4_col2" class="data row4 col2" >0.7869</td>
      <td id="T_b1983_row4_col3" class="data row4 col3" >0.7111</td>
      <td id="T_b1983_row4_col4" class="data row4 col4" >0.7297</td>
      <td id="T_b1983_row4_col5" class="data row4 col5" >0.7202</td>
      <td id="T_b1983_row4_col6" class="data row4 col6" >0.5497</td>
      <td id="T_b1983_row4_col7" class="data row4 col7" >0.5499</td>
      <td id="T_b1983_row4_col8" class="data row4 col8" >0.5033</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row5" class="row_heading level0 row5" >gbc</th>
      <td id="T_b1983_row5_col0" class="data row5 col0" >Gradient Boosting Classifier</td>
      <td id="T_b1983_row5_col1" class="data row5 col1" >0.7865</td>
      <td id="T_b1983_row5_col2" class="data row5 col2" >0.7766</td>
      <td id="T_b1983_row5_col3" class="data row5 col3" >0.7069</td>
      <td id="T_b1983_row5_col4" class="data row5 col4" >0.7283</td>
      <td id="T_b1983_row5_col5" class="data row5 col5" >0.7174</td>
      <td id="T_b1983_row5_col6" class="data row5 col6" >0.5458</td>
      <td id="T_b1983_row5_col7" class="data row5 col7" >0.5461</td>
      <td id="T_b1983_row5_col8" class="data row5 col8" >0.3533</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row6" class="row_heading level0 row6" >rf</th>
      <td id="T_b1983_row6_col0" class="data row6 col0" >Random Forest Classifier</td>
      <td id="T_b1983_row6_col1" class="data row6 col1" >0.7849</td>
      <td id="T_b1983_row6_col2" class="data row6 col2" >0.7790</td>
      <td id="T_b1983_row6_col3" class="data row6 col3" >0.6944</td>
      <td id="T_b1983_row6_col4" class="data row6 col4" >0.7311</td>
      <td id="T_b1983_row6_col5" class="data row6 col5" >0.7123</td>
      <td id="T_b1983_row6_col6" class="data row6 col6" >0.5407</td>
      <td id="T_b1983_row6_col7" class="data row6 col7" >0.5411</td>
      <td id="T_b1983_row6_col8" class="data row6 col8" >0.2467</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row7" class="row_heading level0 row7" >et</th>
      <td id="T_b1983_row7_col0" class="data row7 col0" >Extra Trees Classifier</td>
      <td id="T_b1983_row7_col1" class="data row7 col1" >0.7801</td>
      <td id="T_b1983_row7_col2" class="data row7 col2" >0.7732</td>
      <td id="T_b1983_row7_col3" class="data row7 col3" >0.6777</td>
      <td id="T_b1983_row7_col4" class="data row7 col4" >0.7299</td>
      <td id="T_b1983_row7_col5" class="data row7 col5" >0.7027</td>
      <td id="T_b1983_row7_col6" class="data row7 col6" >0.5286</td>
      <td id="T_b1983_row7_col7" class="data row7 col7" >0.5297</td>
      <td id="T_b1983_row7_col8" class="data row7 col8" >0.3300</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row8" class="row_heading level0 row8" >xgboost</th>
      <td id="T_b1983_row8_col0" class="data row8 col0" >Extreme Gradient Boosting</td>
      <td id="T_b1983_row8_col1" class="data row8 col1" >0.7800</td>
      <td id="T_b1983_row8_col2" class="data row8 col2" >0.7814</td>
      <td id="T_b1983_row8_col3" class="data row8 col3" >0.6944</td>
      <td id="T_b1983_row8_col4" class="data row8 col4" >0.7219</td>
      <td id="T_b1983_row8_col5" class="data row8 col5" >0.7079</td>
      <td id="T_b1983_row8_col6" class="data row8 col6" >0.5316</td>
      <td id="T_b1983_row8_col7" class="data row8 col7" >0.5319</td>
      <td id="T_b1983_row8_col8" class="data row8 col8" >0.1833</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row9" class="row_heading level0 row9" >svm</th>
      <td id="T_b1983_row9_col0" class="data row9 col0" >SVM - Linear Kernel</td>
      <td id="T_b1983_row9_col1" class="data row9 col1" >0.7785</td>
      <td id="T_b1983_row9_col2" class="data row9 col2" >0.7883</td>
      <td id="T_b1983_row9_col3" class="data row9 col3" >0.7405</td>
      <td id="T_b1983_row9_col4" class="data row9 col4" >0.7027</td>
      <td id="T_b1983_row9_col5" class="data row9 col5" >0.7193</td>
      <td id="T_b1983_row9_col6" class="data row9 col6" >0.5369</td>
      <td id="T_b1983_row9_col7" class="data row9 col7" >0.5393</td>
      <td id="T_b1983_row9_col8" class="data row9 col8" >0.0500</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row10" class="row_heading level0 row10" >dt</th>
      <td id="T_b1983_row10_col0" class="data row10 col0" >Decision Tree Classifier</td>
      <td id="T_b1983_row10_col1" class="data row10 col1" >0.7769</td>
      <td id="T_b1983_row10_col2" class="data row10 col2" >0.7710</td>
      <td id="T_b1983_row10_col3" class="data row10 col3" >0.6777</td>
      <td id="T_b1983_row10_col4" class="data row10 col4" >0.7235</td>
      <td id="T_b1983_row10_col5" class="data row10 col5" >0.6996</td>
      <td id="T_b1983_row10_col6" class="data row10 col6" >0.5225</td>
      <td id="T_b1983_row10_col7" class="data row10 col7" >0.5234</td>
      <td id="T_b1983_row10_col8" class="data row10 col8" >0.0400</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row11" class="row_heading level0 row11" >knn</th>
      <td id="T_b1983_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_b1983_row11_col1" class="data row11 col1" >0.7672</td>
      <td id="T_b1983_row11_col2" class="data row11 col2" >0.7734</td>
      <td id="T_b1983_row11_col3" class="data row11 col3" >0.6234</td>
      <td id="T_b1983_row11_col4" class="data row11 col4" >0.7317</td>
      <td id="T_b1983_row11_col5" class="data row11 col5" >0.6715</td>
      <td id="T_b1983_row11_col6" class="data row11 col6" >0.4934</td>
      <td id="T_b1983_row11_col7" class="data row11 col7" >0.4984</td>
      <td id="T_b1983_row11_col8" class="data row11 col8" >0.0733</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row12" class="row_heading level0 row12" >qda</th>
      <td id="T_b1983_row12_col0" class="data row12 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_b1983_row12_col1" class="data row12 col1" >0.7192</td>
      <td id="T_b1983_row12_col2" class="data row12 col2" >0.7699</td>
      <td id="T_b1983_row12_col3" class="data row12 col3" >0.7361</td>
      <td id="T_b1983_row12_col4" class="data row12 col4" >0.6423</td>
      <td id="T_b1983_row12_col5" class="data row12 col5" >0.6740</td>
      <td id="T_b1983_row12_col6" class="data row12 col6" >0.4358</td>
      <td id="T_b1983_row12_col7" class="data row12 col7" >0.4498</td>
      <td id="T_b1983_row12_col8" class="data row12 col8" >0.0500</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row13" class="row_heading level0 row13" >dummy</th>
      <td id="T_b1983_row13_col0" class="data row13 col0" >Dummy Classifier</td>
      <td id="T_b1983_row13_col1" class="data row13 col1" >0.6164</td>
      <td id="T_b1983_row13_col2" class="data row13 col2" >0.5000</td>
      <td id="T_b1983_row13_col3" class="data row13 col3" >0.0000</td>
      <td id="T_b1983_row13_col4" class="data row13 col4" >0.0000</td>
      <td id="T_b1983_row13_col5" class="data row13 col5" >0.0000</td>
      <td id="T_b1983_row13_col6" class="data row13 col6" >0.0000</td>
      <td id="T_b1983_row13_col7" class="data row13 col7" >0.0000</td>
      <td id="T_b1983_row13_col8" class="data row13 col8" >0.0400</td>
    </tr>
    <tr>
      <th id="T_b1983_level0_row14" class="row_heading level0 row14" >nb</th>
      <td id="T_b1983_row14_col0" class="data row14 col0" >Naive Bayes</td>
      <td id="T_b1983_row14_col1" class="data row14 col1" >0.4866</td>
      <td id="T_b1983_row14_col2" class="data row14 col2" >0.7688</td>
      <td id="T_b1983_row14_col3" class="data row14 col3" >0.6544</td>
      <td id="T_b1983_row14_col4" class="data row14 col4" >0.5382</td>
      <td id="T_b1983_row14_col5" class="data row14 col5" >0.4083</td>
      <td id="T_b1983_row14_col6" class="data row14 col6" >0.0352</td>
      <td id="T_b1983_row14_col7" class="data row14 col7" >0.0885</td>
      <td id="T_b1983_row14_col8" class="data row14 col8" >0.0433</td>
    </tr>
  </tbody>
</table>




    Processing:   0%|          | 0/65 [00:00<?, ?it/s]







```python
from pycaret.classification import tune_model

tuned_model = tune_model(best_model, fold=10)
```






<style type="text/css">
#T_6596c_row10_col0, #T_6596c_row10_col1, #T_6596c_row10_col2, #T_6596c_row10_col3, #T_6596c_row10_col4, #T_6596c_row10_col5, #T_6596c_row10_col6 {
  background: yellow;
}
</style>
<table id="T_6596c" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6596c_level0_col0" class="col_heading level0 col0" >Accuracy</th>
      <th id="T_6596c_level0_col1" class="col_heading level0 col1" >AUC</th>
      <th id="T_6596c_level0_col2" class="col_heading level0 col2" >Recall</th>
      <th id="T_6596c_level0_col3" class="col_heading level0 col3" >Prec.</th>
      <th id="T_6596c_level0_col4" class="col_heading level0 col4" >F1</th>
      <th id="T_6596c_level0_col5" class="col_heading level0 col5" >Kappa</th>
      <th id="T_6596c_level0_col6" class="col_heading level0 col6" >MCC</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6596c_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_6596c_row0_col0" class="data row0 col0" >0.7937</td>
      <td id="T_6596c_row0_col1" class="data row0 col1" >0.7618</td>
      <td id="T_6596c_row0_col2" class="data row0 col2" >0.7083</td>
      <td id="T_6596c_row0_col3" class="data row0 col3" >0.7391</td>
      <td id="T_6596c_row0_col4" class="data row0 col4" >0.7234</td>
      <td id="T_6596c_row0_col5" class="data row0 col5" >0.5590</td>
      <td id="T_6596c_row0_col6" class="data row0 col6" >0.5593</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_6596c_row1_col0" class="data row1 col0" >0.8571</td>
      <td id="T_6596c_row1_col1" class="data row1 col1" >0.8109</td>
      <td id="T_6596c_row1_col2" class="data row1 col2" >0.7917</td>
      <td id="T_6596c_row1_col3" class="data row1 col3" >0.8261</td>
      <td id="T_6596c_row1_col4" class="data row1 col4" >0.8085</td>
      <td id="T_6596c_row1_col5" class="data row1 col5" >0.6947</td>
      <td id="T_6596c_row1_col6" class="data row1 col6" >0.6951</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_6596c_row2_col0" class="data row2 col0" >0.9048</td>
      <td id="T_6596c_row2_col1" class="data row2 col1" >0.9054</td>
      <td id="T_6596c_row2_col2" class="data row2 col2" >0.9583</td>
      <td id="T_6596c_row2_col3" class="data row2 col3" >0.8214</td>
      <td id="T_6596c_row2_col4" class="data row2 col4" >0.8846</td>
      <td id="T_6596c_row2_col5" class="data row2 col5" >0.8043</td>
      <td id="T_6596c_row2_col6" class="data row2 col6" >0.8113</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_6596c_row3_col0" class="data row3 col0" >0.7903</td>
      <td id="T_6596c_row3_col1" class="data row3 col1" >0.7592</td>
      <td id="T_6596c_row3_col2" class="data row3 col2" >0.6522</td>
      <td id="T_6596c_row3_col3" class="data row3 col3" >0.7500</td>
      <td id="T_6596c_row3_col4" class="data row3 col4" >0.6977</td>
      <td id="T_6596c_row3_col5" class="data row3 col5" >0.5384</td>
      <td id="T_6596c_row3_col6" class="data row3 col6" >0.5415</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_6596c_row4_col0" class="data row4 col0" >0.7419</td>
      <td id="T_6596c_row4_col1" class="data row4 col1" >0.7577</td>
      <td id="T_6596c_row4_col2" class="data row4 col2" >0.5833</td>
      <td id="T_6596c_row4_col3" class="data row4 col3" >0.7000</td>
      <td id="T_6596c_row4_col4" class="data row4 col4" >0.6364</td>
      <td id="T_6596c_row4_col5" class="data row4 col5" >0.4389</td>
      <td id="T_6596c_row4_col6" class="data row4 col6" >0.4433</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_6596c_row5_col0" class="data row5 col0" >0.7419</td>
      <td id="T_6596c_row5_col1" class="data row5 col1" >0.7621</td>
      <td id="T_6596c_row5_col2" class="data row5 col2" >0.7083</td>
      <td id="T_6596c_row5_col3" class="data row5 col3" >0.6538</td>
      <td id="T_6596c_row5_col4" class="data row5 col4" >0.6800</td>
      <td id="T_6596c_row5_col5" class="data row5 col5" >0.4644</td>
      <td id="T_6596c_row5_col6" class="data row5 col6" >0.4654</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_6596c_row6_col0" class="data row6 col0" >0.8871</td>
      <td id="T_6596c_row6_col1" class="data row6 col1" >0.8805</td>
      <td id="T_6596c_row6_col2" class="data row6 col2" >0.8333</td>
      <td id="T_6596c_row6_col3" class="data row6 col3" >0.8696</td>
      <td id="T_6596c_row6_col4" class="data row6 col4" >0.8511</td>
      <td id="T_6596c_row6_col5" class="data row6 col5" >0.7602</td>
      <td id="T_6596c_row6_col6" class="data row6 col6" >0.7607</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_6596c_row7_col0" class="data row7 col0" >0.7258</td>
      <td id="T_6596c_row7_col1" class="data row7 col1" >0.7484</td>
      <td id="T_6596c_row7_col2" class="data row7 col2" >0.7083</td>
      <td id="T_6596c_row7_col3" class="data row7 col3" >0.6296</td>
      <td id="T_6596c_row7_col4" class="data row7 col4" >0.6667</td>
      <td id="T_6596c_row7_col5" class="data row7 col5" >0.4352</td>
      <td id="T_6596c_row7_col6" class="data row7 col6" >0.4373</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_6596c_row8_col0" class="data row8 col0" >0.7097</td>
      <td id="T_6596c_row8_col1" class="data row8 col1" >0.6727</td>
      <td id="T_6596c_row8_col2" class="data row8 col2" >0.5417</td>
      <td id="T_6596c_row8_col3" class="data row8 col3" >0.6500</td>
      <td id="T_6596c_row8_col4" class="data row8 col4" >0.5909</td>
      <td id="T_6596c_row8_col5" class="data row8 col5" >0.3688</td>
      <td id="T_6596c_row8_col6" class="data row8 col6" >0.3725</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_6596c_row9_col0" class="data row9 col0" >0.8387</td>
      <td id="T_6596c_row9_col1" class="data row9 col1" >0.8355</td>
      <td id="T_6596c_row9_col2" class="data row9 col2" >0.8750</td>
      <td id="T_6596c_row9_col3" class="data row9 col3" >0.7500</td>
      <td id="T_6596c_row9_col4" class="data row9 col4" >0.8077</td>
      <td id="T_6596c_row9_col5" class="data row9 col5" >0.6702</td>
      <td id="T_6596c_row9_col6" class="data row9 col6" >0.6761</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_6596c_row10_col0" class="data row10 col0" >0.7991</td>
      <td id="T_6596c_row10_col1" class="data row10 col1" >0.7894</td>
      <td id="T_6596c_row10_col2" class="data row10 col2" >0.7361</td>
      <td id="T_6596c_row10_col3" class="data row10 col3" >0.7390</td>
      <td id="T_6596c_row10_col4" class="data row10 col4" >0.7347</td>
      <td id="T_6596c_row10_col5" class="data row10 col5" >0.5734</td>
      <td id="T_6596c_row10_col6" class="data row10 col6" >0.5762</td>
    </tr>
    <tr>
      <th id="T_6596c_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_6596c_row11_col0" class="data row11 col0" >0.0662</td>
      <td id="T_6596c_row11_col1" class="data row11 col1" >0.0656</td>
      <td id="T_6596c_row11_col2" class="data row11 col2" >0.1232</td>
      <td id="T_6596c_row11_col3" class="data row11 col3" >0.0777</td>
      <td id="T_6596c_row11_col4" class="data row11 col4" >0.0929</td>
      <td id="T_6596c_row11_col5" class="data row11 col5" >0.1431</td>
      <td id="T_6596c_row11_col6" class="data row11 col6" >0.1434</td>
    </tr>
  </tbody>
</table>




    Processing:   0%|          | 0/7 [00:00<?, ?it/s]


    Fitting 10 folds for each of 10 candidates, totalling 100 fits







```python
predictions = tuned_model.predict(df_test_encoded)
```


```python
df_test['Survived'] = predictions
df_submission = df_test[['PassengerId', 'Survived']]
```


```python
df_submission.shape
```




    (418, 2)



## Deep learning


```python
df_submission.to_csv('/kaggle/working/submission.csv', index=False)
```
