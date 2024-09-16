---
layout: post
title: "gcp_auth"
---

#Authenticate to GCP

Colab allows you to perform operations on Google Cloud Platform via GCP APIs. You must first authenticate to a project you have the appropriate permissions to. Learn more about GCP projects [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects).


```python
from google.colab import auth
PROJECT_ID = "" # @param {type: "string"}
auth.authenticate_user(project_id=PROJECT_ID)
```
