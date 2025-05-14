---
layout: post
title: "2016"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/TN_election_results_comparison_DMK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!ls drive/MyDrive/Data/TN\ Election\ Results
```

    '2016 Detailed Results.xlsx'
    '2016 List of Successful Candidates.xlsx'
    '2021 5- Performance of Political Parties.xlsx'
    '2021 Detailed Results.xlsx'



```python
import pandas as pd
import os
```


```python

pd.read_xls?
```

    Object `pd.read_xls` not found.



```python
folderpath = "drive/MyDrive/Data/TN Election Results"
filename_results_16 = "2016 Detailed Results.xlsx"
filename_results_21 = "2021 Detailed Results.xlsx"

filename_results_16 = os.path.join(folderpath, filename_results_16)
filename_results_21 = os.path.join(folderpath, filename_results_21)

df_results_16 = pd.read_excel(filename_results_16, skiprows=2, index_col=0, engine='openpyxl')
df_results_21 = pd.read_excel(filename_results_21, skiprows=3, index_col=0, engine='openpyxl')

print(f"There are {df_results_16.shape[0]} rows and {df_results_16.shape[1]} \
columns in the 2016 results file and {df_results_21.shape[0]} rows and \
{df_results_21.shape[1]} columns in the 2021 results file")
```

    There are 4010 rows and 11 columns in the 2016 results file and 4470 rows and 13 columns in the 2021 results file


# 2016


```python
df_results_16.sample(10)
```





  <div id="df-06413f6c-3b04-45e7-97f0-588d19141b8f" class="colab-df-container">
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
      <th>Constituency Name</th>
      <th>Candidate Name</th>
      <th>Candidate Sex</th>
      <th>Candidate Age</th>
      <th>Candidate Category</th>
      <th>Party Name</th>
      <th>VALID VOTES POLLED in General</th>
      <th>VALID VOTES POLLED in Postal</th>
      <th>Total Valid Votes</th>
      <th>Total Electors</th>
      <th>Total Votes</th>
    </tr>
    <tr>
      <th>Constituency No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Thiruvallur</td>
      <td>RADHAKRISHNAN D</td>
      <td>M</td>
      <td>36.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>105.0</td>
      <td>0.0</td>
      <td>105.0</td>
      <td>257558.0</td>
      <td>206244.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Kumarapalayam</td>
      <td>YUVARAJ P</td>
      <td>M</td>
      <td>41.0</td>
      <td>GEN</td>
      <td>DMK</td>
      <td>55408.0</td>
      <td>295.0</td>
      <td>55703.0</td>
      <td>231009.0</td>
      <td>186652.0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Palacode</td>
      <td>ANBALAGAN. E.</td>
      <td>M</td>
      <td>41.0</td>
      <td>SC</td>
      <td>IND</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>173.0</td>
      <td>213136.0</td>
      <td>188767.0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Bhavani</td>
      <td>AMMASAI K</td>
      <td>M</td>
      <td>40.0</td>
      <td>GEN</td>
      <td>JD(U)</td>
      <td>358.0</td>
      <td>3.0</td>
      <td>361.0</td>
      <td>228069.0</td>
      <td>188950.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Uthangarai</td>
      <td>V. THEERTHAGIRI</td>
      <td>M</td>
      <td>43.0</td>
      <td>SC</td>
      <td>KMDK</td>
      <td>1259.0</td>
      <td>7.0</td>
      <td>1266.0</td>
      <td>218671.0</td>
      <td>180616.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Usilampatti</td>
      <td>None of the Above</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NOTA</td>
      <td>1668.0</td>
      <td>4.0</td>
      <td>1672.0</td>
      <td>269244.0</td>
      <td>201126.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Vaniyambadi</td>
      <td>KANNADASAN. L</td>
      <td>M</td>
      <td>46.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>139.0</td>
      <td>0.0</td>
      <td>139.0</td>
      <td>223458.0</td>
      <td>172638.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Tambaram</td>
      <td>DHANACHEZHIYAN K</td>
      <td>M</td>
      <td>40.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>301.0</td>
      <td>0.0</td>
      <td>301.0</td>
      <td>381359.0</td>
      <td>235355.0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>Thiruparankundram</td>
      <td>BALASUBRAMANIAN T</td>
      <td>M</td>
      <td>26.0</td>
      <td>GEN</td>
      <td>AMMK</td>
      <td>182.0</td>
      <td>0.0</td>
      <td>182.0</td>
      <td>279599.0</td>
      <td>197480.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Harur</td>
      <td>S.AGILA</td>
      <td>F</td>
      <td>33.0</td>
      <td>SC</td>
      <td>IND</td>
      <td>558.0</td>
      <td>0.0</td>
      <td>558.0</td>
      <td>225545.0</td>
      <td>190134.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-06413f6c-3b04-45e7-97f0-588d19141b8f')"
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
        document.querySelector('#df-06413f6c-3b04-45e7-97f0-588d19141b8f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-06413f6c-3b04-45e7-97f0-588d19141b8f');
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


<div id="df-81c717e6-60ad-4ab4-b092-506532743eca">
  <button class="colab-df-quickchart" onclick="quickchart('df-81c717e6-60ad-4ab4-b092-506532743eca')"
            title="Suggest charts."
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
        document.querySelector('#df-81c717e6-60ad-4ab4-b092-506532743eca button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df_results_21.head(10)
```





  <div id="df-2d24f2cd-685f-417e-b789-8fd07d9fa646" class="colab-df-container">
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
      <th>AC NAME</th>
      <th>CANDIDATE NAME</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>PARTY</th>
      <th>SYMBOL</th>
      <th>GENERAL</th>
      <th>POSTAL</th>
      <th>TOTAL</th>
      <th>% VOTES POLLED</th>
      <th>TOTAL ELECTORS</th>
    </tr>
    <tr>
      <th>AC NO.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>1 GOVINDARAJAN T.J</td>
      <td>MALE</td>
      <td>60.0</td>
      <td>GENERAL</td>
      <td>DMK</td>
      <td>Rising Sun</td>
      <td>125001.0</td>
      <td>1451.0</td>
      <td>126452.0</td>
      <td>56.94</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>2 PRAKASH M</td>
      <td>MALE</td>
      <td>50.0</td>
      <td>GENERAL</td>
      <td>PMK</td>
      <td>Mango</td>
      <td>75004.0</td>
      <td>510.0</td>
      <td>75514.0</td>
      <td>34.00</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>3 USHA</td>
      <td>FEMALE</td>
      <td>31.0</td>
      <td>SC</td>
      <td>NTK</td>
      <td>Ganna Kisan</td>
      <td>11643.0</td>
      <td>58.0</td>
      <td>11701.0</td>
      <td>5.27</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>4 DILLIY K.M</td>
      <td>MALE</td>
      <td>45.0</td>
      <td>GENERAL</td>
      <td>DMDK</td>
      <td>Nagara</td>
      <td>2553.0</td>
      <td>23.0</td>
      <td>2576.0</td>
      <td>1.16</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>5 NOTA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NOTA</td>
      <td>NOTA</td>
      <td>1775.0</td>
      <td>8.0</td>
      <td>1783.0</td>
      <td>0.80</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>6 NAGARAJ S</td>
      <td>MALE</td>
      <td>28.0</td>
      <td>SC</td>
      <td>BSP</td>
      <td>Elephant</td>
      <td>1031.0</td>
      <td>7.0</td>
      <td>1038.0</td>
      <td>0.47</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>7 SARAVANAN V</td>
      <td>MALE</td>
      <td>33.0</td>
      <td>GENERAL</td>
      <td>IJK</td>
      <td>Auto- Rickshaw</td>
      <td>813.0</td>
      <td>3.0</td>
      <td>816.0</td>
      <td>0.37</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>8 SARAVANAN E</td>
      <td>MALE</td>
      <td>44.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>DIAMOND</td>
      <td>526.0</td>
      <td>6.0</td>
      <td>532.0</td>
      <td>0.24</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>9 DEVANATHAN R</td>
      <td>MALE</td>
      <td>26.0</td>
      <td>SC</td>
      <td>IND</td>
      <td>Whistle</td>
      <td>482.0</td>
      <td>0.0</td>
      <td>482.0</td>
      <td>0.22</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>10 LAKSHMI R</td>
      <td>FEMALE</td>
      <td>52.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>COCONUT FARM</td>
      <td>369.0</td>
      <td>5.0</td>
      <td>374.0</td>
      <td>0.17</td>
      <td>281688.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2d24f2cd-685f-417e-b789-8fd07d9fa646')"
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
        document.querySelector('#df-2d24f2cd-685f-417e-b789-8fd07d9fa646 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2d24f2cd-685f-417e-b789-8fd07d9fa646');
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


<div id="df-525a319b-bd66-492d-848d-77b904cd1215">
  <button class="colab-df-quickchart" onclick="quickchart('df-525a319b-bd66-492d-848d-77b904cd1215')"
            title="Suggest charts."
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
        document.querySelector('#df-525a319b-bd66-492d-848d-77b904cd1215 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df_results_16.columns
```




    Index(['Constituency Name', 'Candidate Name', 'Candidate Sex', 'Candidate Age',
           'Candidate Category', ' Party Name', ' VALID VOTES POLLED in General',
           ' VALID VOTES POLLED in Postal', ' Total Valid Votes', 'Total Electors',
           'Total Votes'],
          dtype='object')




```python
df_results_21.columns
```




    Index(['AC NO.', 'AC NAME', 'CANDIDATE NAME', 'SEX', 'AGE', 'CATEGORY',
           'PARTY', 'SYMBOL', 'GENERAL', 'POSTAL', 'TOTAL', '% VOTES POLLED',
           'TOTAL ELECTORS'],
          dtype='object')




```python
df_results_21 = df_results_21.reset_index(drop = True)
df_results_21 = df_results_21.set_index('AC NO.')

df_results_21.head()
```





  <div id="df-9c75c3fd-afdc-475a-be3d-c6bf9a940274" class="colab-df-container">
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
      <th>AC NAME</th>
      <th>CANDIDATE NAME</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>PARTY</th>
      <th>SYMBOL</th>
      <th>GENERAL</th>
      <th>POSTAL</th>
      <th>TOTAL</th>
      <th>% VOTES POLLED</th>
      <th>TOTAL ELECTORS</th>
    </tr>
    <tr>
      <th>AC NO.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>1 GOVINDARAJAN T.J</td>
      <td>MALE</td>
      <td>60.0</td>
      <td>GENERAL</td>
      <td>DMK</td>
      <td>Rising Sun</td>
      <td>125001.0</td>
      <td>1451.0</td>
      <td>126452.0</td>
      <td>56.94</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>2 PRAKASH M</td>
      <td>MALE</td>
      <td>50.0</td>
      <td>GENERAL</td>
      <td>PMK</td>
      <td>Mango</td>
      <td>75004.0</td>
      <td>510.0</td>
      <td>75514.0</td>
      <td>34.00</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>3 USHA</td>
      <td>FEMALE</td>
      <td>31.0</td>
      <td>SC</td>
      <td>NTK</td>
      <td>Ganna Kisan</td>
      <td>11643.0</td>
      <td>58.0</td>
      <td>11701.0</td>
      <td>5.27</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>4 DILLIY K.M</td>
      <td>MALE</td>
      <td>45.0</td>
      <td>GENERAL</td>
      <td>DMDK</td>
      <td>Nagara</td>
      <td>2553.0</td>
      <td>23.0</td>
      <td>2576.0</td>
      <td>1.16</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>5 NOTA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NOTA</td>
      <td>NOTA</td>
      <td>1775.0</td>
      <td>8.0</td>
      <td>1783.0</td>
      <td>0.80</td>
      <td>281688.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9c75c3fd-afdc-475a-be3d-c6bf9a940274')"
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
        document.querySelector('#df-9c75c3fd-afdc-475a-be3d-c6bf9a940274 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9c75c3fd-afdc-475a-be3d-c6bf9a940274');
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


<div id="df-70c148ec-97ff-48f1-8908-40a9ddb9ac24">
  <button class="colab-df-quickchart" onclick="quickchart('df-70c148ec-97ff-48f1-8908-40a9ddb9ac24')"
            title="Suggest charts."
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
        document.querySelector('#df-70c148ec-97ff-48f1-8908-40a9ddb9ac24 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df_results_combined = df_results_16.merge(df_results_21, left_index = True, right_index = True, suffixes=('_16', '_df21'))

df_results_combined.shape
```




    (76422, 23)




```python
df_results_combined.head()
```





  <div id="df-d72329b0-8675-4b2a-ba5b-0d036d5d0711" class="colab-df-container">
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
      <th>Constituency Name</th>
      <th>Candidate Name</th>
      <th>Candidate Sex</th>
      <th>Candidate Age</th>
      <th>Candidate Category</th>
      <th>Party Name</th>
      <th>VALID VOTES POLLED in General</th>
      <th>VALID VOTES POLLED in Postal</th>
      <th>Total Valid Votes</th>
      <th>Total Electors</th>
      <th>...</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>PARTY</th>
      <th>SYMBOL</th>
      <th>GENERAL</th>
      <th>POSTAL</th>
      <th>TOTAL</th>
      <th>% VOTES POLLED</th>
      <th>TOTAL ELECTORS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>VIJAYAKUMAR  K S</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>89021.0</td>
      <td>311.0</td>
      <td>89332.0</td>
      <td>260912.0</td>
      <td>...</td>
      <td>MALE</td>
      <td>60.0</td>
      <td>GENERAL</td>
      <td>DMK</td>
      <td>Rising Sun</td>
      <td>125001.0</td>
      <td>1451.0</td>
      <td>126452.0</td>
      <td>56.94</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>VIJAYAKUMAR  K S</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>89021.0</td>
      <td>311.0</td>
      <td>89332.0</td>
      <td>260912.0</td>
      <td>...</td>
      <td>MALE</td>
      <td>50.0</td>
      <td>GENERAL</td>
      <td>PMK</td>
      <td>Mango</td>
      <td>75004.0</td>
      <td>510.0</td>
      <td>75514.0</td>
      <td>34.00</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>VIJAYAKUMAR  K S</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>89021.0</td>
      <td>311.0</td>
      <td>89332.0</td>
      <td>260912.0</td>
      <td>...</td>
      <td>FEMALE</td>
      <td>31.0</td>
      <td>SC</td>
      <td>NTK</td>
      <td>Ganna Kisan</td>
      <td>11643.0</td>
      <td>58.0</td>
      <td>11701.0</td>
      <td>5.27</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>VIJAYAKUMAR  K S</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>89021.0</td>
      <td>311.0</td>
      <td>89332.0</td>
      <td>260912.0</td>
      <td>...</td>
      <td>MALE</td>
      <td>45.0</td>
      <td>GENERAL</td>
      <td>DMDK</td>
      <td>Nagara</td>
      <td>2553.0</td>
      <td>23.0</td>
      <td>2576.0</td>
      <td>1.16</td>
      <td>281688.0</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>Gummidipoondi</td>
      <td>VIJAYAKUMAR  K S</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>89021.0</td>
      <td>311.0</td>
      <td>89332.0</td>
      <td>260912.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NOTA</td>
      <td>NOTA</td>
      <td>1775.0</td>
      <td>8.0</td>
      <td>1783.0</td>
      <td>0.80</td>
      <td>281688.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d72329b0-8675-4b2a-ba5b-0d036d5d0711')"
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
        document.querySelector('#df-d72329b0-8675-4b2a-ba5b-0d036d5d0711 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d72329b0-8675-4b2a-ba5b-0d036d5d0711');
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


<div id="df-db75f16c-eb46-43bd-9ecd-33118cc0a8bf">
  <button class="colab-df-quickchart" onclick="quickchart('df-db75f16c-eb46-43bd-9ecd-33118cc0a8bf')"
            title="Suggest charts."
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
        document.querySelector('#df-db75f16c-eb46-43bd-9ecd-33118cc0a8bf button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df_results_16[df_results_16['Constituency Name'].str.lower().str.contains('madurai central')]
```





  <div id="df-5bb645ae-9e87-403e-a043-0b81fd7504eb" class="colab-df-container">
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
      <th>Constituency Name</th>
      <th>Candidate Name</th>
      <th>Candidate Sex</th>
      <th>Candidate Age</th>
      <th>Candidate Category</th>
      <th>Party Name</th>
      <th>VALID VOTES POLLED in General</th>
      <th>VALID VOTES POLLED in Postal</th>
      <th>Total Valid Votes</th>
      <th>Total Electors</th>
      <th>Total Votes</th>
    </tr>
    <tr>
      <th>Constituency No.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>PALANIVEL THIAGARAJAN</td>
      <td>M</td>
      <td>49.0</td>
      <td>GEN</td>
      <td>DMK</td>
      <td>64092.0</td>
      <td>570.0</td>
      <td>64662.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>JEYABAL M</td>
      <td>M</td>
      <td>56.0</td>
      <td>GEN</td>
      <td>ADMK</td>
      <td>58648.0</td>
      <td>252.0</td>
      <td>58900.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SIVAMUTHUKUMAR D</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>DMDK</td>
      <td>11184.0</td>
      <td>51.0</td>
      <td>11235.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>KARTHIK PRABU M</td>
      <td>M</td>
      <td>42.0</td>
      <td>GEN</td>
      <td>BJP</td>
      <td>6904.0</td>
      <td>22.0</td>
      <td>6926.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>VETRIKUMARAN C</td>
      <td>M</td>
      <td>41.0</td>
      <td>GEN</td>
      <td>NTK</td>
      <td>2980.0</td>
      <td>18.0</td>
      <td>2998.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>None of the Above</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NOTA</td>
      <td>2676.0</td>
      <td>7.0</td>
      <td>2683.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>JAFAR SULTHAN IBRAHIM  M</td>
      <td>M</td>
      <td>42.0</td>
      <td>GEN</td>
      <td>SDPI</td>
      <td>1684.0</td>
      <td>2.0</td>
      <td>1686.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SELVAM T</td>
      <td>M</td>
      <td>51.0</td>
      <td>GEN</td>
      <td>PMK</td>
      <td>1003.0</td>
      <td>4.0</td>
      <td>1007.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SUDHAKAR R</td>
      <td>M</td>
      <td>36.0</td>
      <td>GEN</td>
      <td>AIFB</td>
      <td>470.0</td>
      <td>1.0</td>
      <td>471.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>BALASUBRAMANIAN  R</td>
      <td>M</td>
      <td>52.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>253.0</td>
      <td>0.0</td>
      <td>253.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>THAVAMANI A</td>
      <td>M</td>
      <td>51.0</td>
      <td>SC</td>
      <td>BSP</td>
      <td>203.0</td>
      <td>0.0</td>
      <td>203.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>VEL  S</td>
      <td>M</td>
      <td>33.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>161.0</td>
      <td>0.0</td>
      <td>161.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>MUTHURAJA  K.M</td>
      <td>M</td>
      <td>45.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>156.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SHAHUL HAMEED  A</td>
      <td>M</td>
      <td>40.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>135.0</td>
      <td>0.0</td>
      <td>135.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>KARMEGAM M</td>
      <td>M</td>
      <td>44.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>106.0</td>
      <td>0.0</td>
      <td>106.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SATHIYENDRAN  N</td>
      <td>M</td>
      <td>34.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>97.0</td>
      <td>0.0</td>
      <td>97.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>SOUNDARAM G T</td>
      <td>F</td>
      <td>55.0</td>
      <td>SC</td>
      <td>LJP</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>95.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>RAJKUMAR  S</td>
      <td>M</td>
      <td>41.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>88.0</td>
      <td>0.0</td>
      <td>88.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>NAVANEETHAKRISHNAN  R</td>
      <td>M</td>
      <td>35.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>74.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Madurai Central</td>
      <td>AVADAINATHAN  V</td>
      <td>M</td>
      <td>62.0</td>
      <td>GEN</td>
      <td>IND</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>233084.0</td>
      <td>151982.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5bb645ae-9e87-403e-a043-0b81fd7504eb')"
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
        document.querySelector('#df-5bb645ae-9e87-403e-a043-0b81fd7504eb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5bb645ae-9e87-403e-a043-0b81fd7504eb');
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


<div id="df-3d9b6df7-98c8-4923-8efd-859e36909b03">
  <button class="colab-df-quickchart" onclick="quickchart('df-3d9b6df7-98c8-4923-8efd-859e36909b03')"
            title="Suggest charts."
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
        document.querySelector('#df-3d9b6df7-98c8-4923-8efd-859e36909b03 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df_results_21['AC NAME'].isna().value_counts()
```




    False    4232
    True      238
    Name: AC NAME, dtype: int64




```python

df_results_21.dropna(inplace = True)
```


```python
df_results_21[df_results_21['AC NAME'].str.lower().str.contains('madurai central')]
```





  <div id="df-f2c34daf-929e-4fc1-8f12-af92cee753bc" class="colab-df-container">
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
      <th>AC NAME</th>
      <th>CANDIDATE NAME</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>CATEGORY</th>
      <th>PARTY</th>
      <th>SYMBOL</th>
      <th>GENERAL</th>
      <th>POSTAL</th>
      <th>TOTAL</th>
      <th>% VOTES POLLED</th>
      <th>TOTAL ELECTORS</th>
    </tr>
    <tr>
      <th>AC NO.</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>1 PALANIVEL THIAGA RAJAN</td>
      <td>MALE</td>
      <td>55.0</td>
      <td>GENERAL</td>
      <td>DMK</td>
      <td>Rising Sun</td>
      <td>72231.0</td>
      <td>974.0</td>
      <td>73205.0</td>
      <td>48.99</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>2 JOTHI MUTHURAMALINGAM N</td>
      <td>MALE</td>
      <td>54.0</td>
      <td>GENERAL</td>
      <td>ADMK</td>
      <td>Two Leaves</td>
      <td>38834.0</td>
      <td>195.0</td>
      <td>39029.0</td>
      <td>26.12</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>3 MANI B</td>
      <td>MALE</td>
      <td>46.0</td>
      <td>GENERAL</td>
      <td>MNM</td>
      <td>Battery torch</td>
      <td>14437.0</td>
      <td>58.0</td>
      <td>14495.0</td>
      <td>9.70</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>4 PANDIAMMAL J</td>
      <td>FEMALE</td>
      <td>38.0</td>
      <td>GENERAL</td>
      <td>NTK</td>
      <td>Ganna Kisan</td>
      <td>11156.0</td>
      <td>59.0</td>
      <td>11215.0</td>
      <td>7.51</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>5 KREMMER SURESH</td>
      <td>MALE</td>
      <td>49.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>Hat</td>
      <td>4880.0</td>
      <td>27.0</td>
      <td>4907.0</td>
      <td>3.28</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>6 SIKKANDAR BATCHA G S</td>
      <td>MALE</td>
      <td>44.0</td>
      <td>GENERAL</td>
      <td>SDPI</td>
      <td>Pressure Cooker</td>
      <td>3327.0</td>
      <td>20.0</td>
      <td>3347.0</td>
      <td>2.24</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>8 KRISHNAPRABBU M R</td>
      <td>MALE</td>
      <td>39.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>Ring</td>
      <td>437.0</td>
      <td>3.0</td>
      <td>440.0</td>
      <td>0.29</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>9 ESWARI M</td>
      <td>FEMALE</td>
      <td>42.0</td>
      <td>GENERAL</td>
      <td>MIDP</td>
      <td>CCTV Camera</td>
      <td>285.0</td>
      <td>2.0</td>
      <td>287.0</td>
      <td>0.19</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>10 THAVAMANI A</td>
      <td>MALE</td>
      <td>55.0</td>
      <td>SC</td>
      <td>BSP</td>
      <td>Elephant</td>
      <td>283.0</td>
      <td>3.0</td>
      <td>286.0</td>
      <td>0.19</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>11 ELANGOVAN R</td>
      <td>MALE</td>
      <td>49.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>COCONUT FARM</td>
      <td>264.0</td>
      <td>0.0</td>
      <td>264.0</td>
      <td>0.18</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>12 RAJASURIYAA K R</td>
      <td>MALE</td>
      <td>26.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>Baby Walker</td>
      <td>192.0</td>
      <td>1.0</td>
      <td>193.0</td>
      <td>0.13</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>13 RAJAKUMAR NAIDU E V S</td>
      <td>MALE</td>
      <td>51.0</td>
      <td>GENERAL</td>
      <td>TTNP</td>
      <td>Auto- Rickshaw</td>
      <td>175.0</td>
      <td>1.0</td>
      <td>176.0</td>
      <td>0.12</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>14 SIVASANKAR S</td>
      <td>MALE</td>
      <td>35.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>Balloon</td>
      <td>94.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>0.06</td>
      <td>241913.0</td>
    </tr>
    <tr>
      <th>193.0</th>
      <td>Madurai Central</td>
      <td>15 SATHIYENDRAN N</td>
      <td>MALE</td>
      <td>39.0</td>
      <td>GENERAL</td>
      <td>IND</td>
      <td>Petrol Pump</td>
      <td>51.0</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>0.03</td>
      <td>241913.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f2c34daf-929e-4fc1-8f12-af92cee753bc')"
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
        document.querySelector('#df-f2c34daf-929e-4fc1-8f12-af92cee753bc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f2c34daf-929e-4fc1-8f12-af92cee753bc');
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


<div id="df-c3f53dc6-bf0a-4b2c-968b-cfa652d60037">
  <button class="colab-df-quickchart" onclick="quickchart('df-c3f53dc6-bf0a-4b2c-968b-cfa652d60037')"
            title="Suggest charts."
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
        document.querySelector('#df-c3f53dc6-bf0a-4b2c-968b-cfa652d60037 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python

```
