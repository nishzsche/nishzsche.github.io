---
layout: post
title: "Using BigQuery with Pandas API"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/Snippets_BigQuery.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Using BigQuery with Pandas API


1.   Use the [Cloud Resource Manager](https://console.cloud.google.com/cloud-resource-manager) to Create a Cloud Platform project if you do not already have one.
2.   [Enable billing](https://support.google.com/cloud/answer/6293499#enable-billing) for the project.
3.   [Enable BigQuery](https://console.cloud.google.com/flows/enableapi?apiid=bigquery) APIs for the project.



```
from google.colab import auth
auth.authenticate_user()
```


```
import pandas as pd

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = '[your Cloud Platform project ID]'
sample_count = 2000

row_count = pd.io.gbq.read_gbq('''
  SELECT
    COUNT(*) as total
  FROM `bigquery-public-data.samples.gsod`
''', project_id=project_id).total[0]

df = pd.io.gbq.read_gbq(f'''
  SELECT
    *
  FROM
    `bigquery-public-data.samples.gsod`
  WHERE RAND() < {sample_count}/{row_count}
''', project_id=project_id)

print(f'Full dataset has {row_count} rows')
```


```
df.describe()
```

## More info

- The [GSOD sample table](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&t=gsod&page=table) contains weather information collected by NOAA, such as precipitation amounts and wind speeds from late 1929 to early 2010.
- [Pandas GBQ Documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_gbq.html)


# Using BigQuery with Cloud API


1.   Use the [Cloud Resource Manager](https://console.cloud.google.com/cloud-resource-manager) to Create a Cloud Platform project if you do not already have one.
2.   [Enable billing](https://support.google.com/cloud/answer/6293499#enable-billing) for the project.
3.   [Enable BigQuery](https://console.cloud.google.com/flows/enableapi?apiid=bigquery) APIs for the project.

[BigQuery Documentation](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/usage.html)


```
from google.colab import auth
auth.authenticate_user()
```


```
from google.cloud import bigquery

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = '[your Cloud Platform project ID]'
client = bigquery.Client(project=project_id)

for dataset in client.list_datasets():
  print(dataset.dataset_id)
```

# Pandas: display dataframes as interactive tables

The `google.colab.data_table` package provides an interactive display of tabular data within colab. It can be enabled by loading the extension:


```
%load_ext google.colab.data_table
```

With this enabled, dataframes are shown as rich, interactive tables:


```
from vega_datasets import data
data.cars()
```

To restore the standard static display, unload the extension:


```
%unload_ext google.colab.data_table
```


```
data.cars()
```

# Use BigQuery DataFrames

Use the ```bigframes``` package
, which provides a Pythonic DataFrame and machine learning (ML) API powered by the BigQuery engine. ``bigframes.pandas`` implements a pandas-like API on top of BigQuery.

[BigQuery DataFrames Documentation](https://cloud.google.com/python/docs/reference/bigframes/latest)


```
from google.colab import auth
auth.authenticate_user()
```

## Declare the Cloud project ID which will be used throughout this notebook


```
project_id = '[your project ID]'
```

## Set up


```
import bigframes.pandas as bpd
from google.cloud import bigquery

# Set BigQuery DataFrames options
bpd.options.bigquery.project = project_id
bpd.options.bigquery.location = "US"
```

## Load data from a query input


```
from google.colab import syntax
query = syntax.sql('''
    SELECT *
    FROM `bigquery-public-data.ml_datasets.penguins`
    LIMIT 20
''')

# Load data from a BigQuery table using BigFrames DataFrames:
bq_df = bpd.read_gbq(query)
```

## Describe the sampled data


```
bq_df.describe()
```

## View the first 10 rows


```
bq_df.head(10)
```
