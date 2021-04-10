# `deepfeatx`: Deep Learning Feature Extractor of Images using Transfer Learning Models
> Helper for automatic extraction of features from images (and soon text as well) from transfer learning models like ResNet, VGG16 and EfficientNet.


## Install

```
#hide_output
!pip install deepfeatx
```

## Why this project has been created
- Fill the gap between ML and DL thus allowing estimators beyond only neural networks for computer vision and NLP problems
- Neural network models are too painful to setup and train - data generators, optimizers, learning rates, loss functions, training loops, batch size, etc. 
- State of the art results are possible thanks to pretrained models that allows feature extraction
- With this library we can handle those problems as they were traditional machine learning problems
- Possibility of using low-code APIs like `scikit-learn` for computer vision and NLP problems

## Usage
### Extracting features from an image

```
from deepfeatx.image import ImageFeatureExtractor
fe = ImageFeatureExtractor()
```

```
im_url='https://raw.githubusercontent.com/WittmannF/deepfeatx/master/sample_data/cats_vs_dogs/valid/dog/dog.124.jpg'
fe.read_img_url(im_url)
```




![png](docs/images/output_5_0.png)



```
fe.url_to_vector(im_url)
```




    array([[0.28227222, 1.0504329 , 0.11333513, ..., 0.18499821, 0.02220216,
            0.06158591]], dtype=float32)



### Extracting Features from a Folder with Images

```
!git clone https://github.com/WittmannF/image-scraper.git
```

    Cloning into 'image-scraper'...
    remote: Enumerating objects: 543, done.[K
    remote: Counting objects: 100% (543/543), done.[K
    remote: Compressing objects: 100% (536/536), done.[K
    remote: Total 543 (delta 12), reused 521 (delta 3), pack-reused 0[K
    Receiving objects: 100% (543/543), 23.58 MiB | 43.83 MiB/s, done.
    Resolving deltas: 100% (12/12), done.


```
df=fe.extract_features_from_directory('image-scraper/images/pug',
                                   classes_as_folders=False,
                                   export_vectors_as_df=True)

df.head()
```

    Found 4 validated image filenames.
    1/1 [==============================] - 1s 1s/step





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
      <th>filepaths</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
      <th>2025</th>
      <th>2026</th>
      <th>2027</th>
      <th>2028</th>
      <th>2029</th>
      <th>2030</th>
      <th>2031</th>
      <th>2032</th>
      <th>2033</th>
      <th>2034</th>
      <th>2035</th>
      <th>2036</th>
      <th>2037</th>
      <th>2038</th>
      <th>2039</th>
      <th>2040</th>
      <th>2041</th>
      <th>2042</th>
      <th>2043</th>
      <th>2044</th>
      <th>2045</th>
      <th>2046</th>
      <th>2047</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>image-scraper/images/pug/efd08a2dc5.jpg</td>
      <td>0.030706</td>
      <td>0.042392</td>
      <td>0.422986</td>
      <td>1.316509</td>
      <td>0.020907</td>
      <td>0.000000</td>
      <td>0.081956</td>
      <td>0.404422</td>
      <td>0.489834</td>
      <td>0.004070</td>
      <td>0.046379</td>
      <td>0.438150</td>
      <td>0.181135</td>
      <td>0.013462</td>
      <td>0.177939</td>
      <td>0.000000</td>
      <td>0.007675</td>
      <td>0.346413</td>
      <td>0.774563</td>
      <td>0.242160</td>
      <td>0.146554</td>
      <td>0.000000</td>
      <td>0.374704</td>
      <td>0.546130</td>
      <td>0.914375</td>
      <td>0.034065</td>
      <td>0.018123</td>
      <td>0.018173</td>
      <td>0.137671</td>
      <td>0.062699</td>
      <td>0.054896</td>
      <td>0.461135</td>
      <td>0.121660</td>
      <td>0.041247</td>
      <td>0.389360</td>
      <td>1.212443</td>
      <td>0.021843</td>
      <td>0.000000</td>
      <td>0.583172</td>
      <td>...</td>
      <td>0.048073</td>
      <td>0.858236</td>
      <td>0.054315</td>
      <td>0.176547</td>
      <td>0.009346</td>
      <td>0.220590</td>
      <td>1.808879</td>
      <td>0.165877</td>
      <td>0.446522</td>
      <td>0.181712</td>
      <td>0.076804</td>
      <td>0.651420</td>
      <td>0.812974</td>
      <td>0.710875</td>
      <td>0.331778</td>
      <td>0.112184</td>
      <td>0.294079</td>
      <td>0.075776</td>
      <td>0.000000</td>
      <td>1.752276</td>
      <td>0.279192</td>
      <td>0.541461</td>
      <td>0.226151</td>
      <td>0.000000</td>
      <td>0.556450</td>
      <td>0.101981</td>
      <td>0.666771</td>
      <td>0.006849</td>
      <td>0.085295</td>
      <td>0.020708</td>
      <td>0.013765</td>
      <td>0.642072</td>
      <td>1.818820</td>
      <td>0.299440</td>
      <td>0.000000</td>
      <td>0.419997</td>
      <td>0.200106</td>
      <td>0.179524</td>
      <td>0.026852</td>
      <td>0.079208</td>
    </tr>
    <tr>
      <th>1</th>
      <td>image-scraper/images/pug/6fb189ce56.jpg</td>
      <td>0.373005</td>
      <td>0.102007</td>
      <td>0.097662</td>
      <td>0.362927</td>
      <td>0.549804</td>
      <td>0.118015</td>
      <td>0.000000</td>
      <td>0.104320</td>
      <td>0.102526</td>
      <td>0.013431</td>
      <td>0.358213</td>
      <td>0.404455</td>
      <td>0.124487</td>
      <td>0.050493</td>
      <td>0.120028</td>
      <td>0.346251</td>
      <td>0.070185</td>
      <td>0.262804</td>
      <td>0.808682</td>
      <td>0.055136</td>
      <td>0.006087</td>
      <td>0.757376</td>
      <td>0.097778</td>
      <td>0.420553</td>
      <td>0.000000</td>
      <td>0.141114</td>
      <td>0.010222</td>
      <td>0.000000</td>
      <td>0.233845</td>
      <td>0.908515</td>
      <td>0.023404</td>
      <td>0.637513</td>
      <td>0.022050</td>
      <td>0.084186</td>
      <td>0.084896</td>
      <td>0.299784</td>
      <td>0.000000</td>
      <td>0.015850</td>
      <td>0.100193</td>
      <td>...</td>
      <td>0.288275</td>
      <td>0.037123</td>
      <td>0.045735</td>
      <td>0.374429</td>
      <td>0.196037</td>
      <td>0.734308</td>
      <td>3.006210</td>
      <td>0.114692</td>
      <td>0.981330</td>
      <td>0.853356</td>
      <td>0.000000</td>
      <td>0.103360</td>
      <td>0.988711</td>
      <td>0.621975</td>
      <td>0.000000</td>
      <td>0.136326</td>
      <td>0.352058</td>
      <td>0.138432</td>
      <td>0.000000</td>
      <td>0.149371</td>
      <td>0.004514</td>
      <td>0.011803</td>
      <td>0.257658</td>
      <td>0.208908</td>
      <td>0.029987</td>
      <td>0.341993</td>
      <td>0.078165</td>
      <td>0.177329</td>
      <td>0.025573</td>
      <td>0.002849</td>
      <td>0.210634</td>
      <td>0.213147</td>
      <td>0.013510</td>
      <td>0.574434</td>
      <td>0.017234</td>
      <td>0.628008</td>
      <td>0.000000</td>
      <td>0.184550</td>
      <td>0.000000</td>
      <td>0.248099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>image-scraper/images/pug/ee815ebc87.jpg</td>
      <td>0.263904</td>
      <td>0.430294</td>
      <td>0.391808</td>
      <td>0.033076</td>
      <td>0.200174</td>
      <td>0.019310</td>
      <td>0.002792</td>
      <td>0.129120</td>
      <td>0.050257</td>
      <td>0.212521</td>
      <td>0.253893</td>
      <td>0.226962</td>
      <td>0.343641</td>
      <td>0.071063</td>
      <td>0.003507</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.291578</td>
      <td>1.115863</td>
      <td>0.053286</td>
      <td>0.000000</td>
      <td>0.114654</td>
      <td>0.092095</td>
      <td>0.668449</td>
      <td>0.319849</td>
      <td>0.037851</td>
      <td>0.514418</td>
      <td>0.028072</td>
      <td>0.088368</td>
      <td>2.914616</td>
      <td>0.000000</td>
      <td>0.385466</td>
      <td>0.039530</td>
      <td>0.117206</td>
      <td>1.620699</td>
      <td>2.245774</td>
      <td>0.551754</td>
      <td>0.753468</td>
      <td>0.002836</td>
      <td>...</td>
      <td>0.004382</td>
      <td>0.233663</td>
      <td>0.290555</td>
      <td>0.111746</td>
      <td>0.369157</td>
      <td>0.413205</td>
      <td>1.606386</td>
      <td>0.328596</td>
      <td>0.030728</td>
      <td>0.785285</td>
      <td>0.047967</td>
      <td>0.530228</td>
      <td>0.399490</td>
      <td>0.710600</td>
      <td>0.000000</td>
      <td>0.070264</td>
      <td>0.080831</td>
      <td>0.044579</td>
      <td>0.018119</td>
      <td>0.173507</td>
      <td>0.032501</td>
      <td>0.462589</td>
      <td>0.140939</td>
      <td>0.064460</td>
      <td>0.020042</td>
      <td>1.105275</td>
      <td>0.220959</td>
      <td>0.062867</td>
      <td>0.025815</td>
      <td>0.643085</td>
      <td>0.048243</td>
      <td>0.147806</td>
      <td>1.430153</td>
      <td>0.266686</td>
      <td>0.005126</td>
      <td>0.158225</td>
      <td>0.097526</td>
      <td>0.005045</td>
      <td>0.060016</td>
      <td>1.109626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>image-scraper/images/pug/24d0f1eee3.jpg</td>
      <td>0.068498</td>
      <td>0.319734</td>
      <td>0.081250</td>
      <td>1.248271</td>
      <td>0.035602</td>
      <td>0.003398</td>
      <td>0.000000</td>
      <td>0.131528</td>
      <td>0.099515</td>
      <td>0.153028</td>
      <td>0.124836</td>
      <td>0.462332</td>
      <td>0.027325</td>
      <td>0.139767</td>
      <td>0.060380</td>
      <td>0.000000</td>
      <td>0.122781</td>
      <td>0.838560</td>
      <td>0.000000</td>
      <td>0.112821</td>
      <td>0.624077</td>
      <td>0.000000</td>
      <td>0.053588</td>
      <td>0.492699</td>
      <td>1.002087</td>
      <td>0.000000</td>
      <td>0.622109</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.718141</td>
      <td>0.000000</td>
      <td>0.430249</td>
      <td>0.110824</td>
      <td>0.306448</td>
      <td>1.200562</td>
      <td>0.156815</td>
      <td>0.001156</td>
      <td>0.301857</td>
      <td>0.915826</td>
      <td>...</td>
      <td>0.038859</td>
      <td>0.521360</td>
      <td>0.000000</td>
      <td>0.051310</td>
      <td>0.041922</td>
      <td>0.394816</td>
      <td>0.065257</td>
      <td>0.014697</td>
      <td>0.732560</td>
      <td>0.330840</td>
      <td>0.000000</td>
      <td>0.084516</td>
      <td>0.998141</td>
      <td>0.499400</td>
      <td>0.048280</td>
      <td>0.270612</td>
      <td>0.224201</td>
      <td>0.015745</td>
      <td>0.024890</td>
      <td>0.671853</td>
      <td>0.052918</td>
      <td>0.386016</td>
      <td>0.528613</td>
      <td>0.068114</td>
      <td>0.304690</td>
      <td>0.543489</td>
      <td>0.005538</td>
      <td>0.212559</td>
      <td>0.000000</td>
      <td>0.064679</td>
      <td>0.258502</td>
      <td>1.042543</td>
      <td>0.691716</td>
      <td>0.264938</td>
      <td>0.112621</td>
      <td>0.927996</td>
      <td>0.050389</td>
      <td>0.000000</td>
      <td>0.087217</td>
      <td>0.066992</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 2049 columns</p>
</div>



### Extracting Features from a directory having one sub-folder per class

If the directory structure is the following:
```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```
We can enter `main_directory` as input by changing `classes_as_folders` as True:

```
df=fe.extract_features_from_directory('image-scraper/images/',
                                      classes_as_folders=True,
                                      export_vectors_as_df=True,
                                      export_class_names=True)

df.head()
```

    Found 504 images belonging to 6 classes.
    16/16 [==============================] - 3s 204ms/step





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
      <th>filepaths</th>
      <th>classes</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
      <th>2025</th>
      <th>2026</th>
      <th>2027</th>
      <th>2028</th>
      <th>2029</th>
      <th>2030</th>
      <th>2031</th>
      <th>2032</th>
      <th>2033</th>
      <th>2034</th>
      <th>2035</th>
      <th>2036</th>
      <th>2037</th>
      <th>2038</th>
      <th>2039</th>
      <th>2040</th>
      <th>2041</th>
      <th>2042</th>
      <th>2043</th>
      <th>2044</th>
      <th>2045</th>
      <th>2046</th>
      <th>2047</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>image-scraper/images/chihuahua/00dcf98689.jpg</td>
      <td>chihuahua</td>
      <td>0.640898</td>
      <td>0.887127</td>
      <td>0.017012</td>
      <td>0.723458</td>
      <td>0.164907</td>
      <td>0.010150</td>
      <td>0.042344</td>
      <td>0.987458</td>
      <td>0.000000</td>
      <td>0.014738</td>
      <td>1.079895</td>
      <td>0.402594</td>
      <td>0.813250</td>
      <td>0.898078</td>
      <td>0.283853</td>
      <td>2.501164</td>
      <td>0.010436</td>
      <td>0.805332</td>
      <td>0.355824</td>
      <td>0.256257</td>
      <td>0.000000</td>
      <td>0.040217</td>
      <td>0.037436</td>
      <td>0.693020</td>
      <td>0.079267</td>
      <td>0.314281</td>
      <td>0.078728</td>
      <td>0.721479</td>
      <td>0.109187</td>
      <td>0.159317</td>
      <td>0.231705</td>
      <td>1.474535</td>
      <td>0.184643</td>
      <td>1.271005</td>
      <td>0.379876</td>
      <td>1.453021</td>
      <td>0.380090</td>
      <td>0.907110</td>
      <td>...</td>
      <td>0.020598</td>
      <td>0.286543</td>
      <td>0.002142</td>
      <td>0.030759</td>
      <td>0.660899</td>
      <td>0.103130</td>
      <td>1.164128</td>
      <td>0.371421</td>
      <td>0.350240</td>
      <td>0.330034</td>
      <td>0.053406</td>
      <td>0.000000</td>
      <td>0.037163</td>
      <td>0.444834</td>
      <td>0.212866</td>
      <td>0.492152</td>
      <td>0.874334</td>
      <td>0.237795</td>
      <td>0.115668</td>
      <td>0.348703</td>
      <td>0.458247</td>
      <td>0.558596</td>
      <td>0.192963</td>
      <td>0.161454</td>
      <td>0.085633</td>
      <td>0.450116</td>
      <td>0.200245</td>
      <td>0.000000</td>
      <td>0.250926</td>
      <td>0.265246</td>
      <td>0.289272</td>
      <td>0.182084</td>
      <td>0.638065</td>
      <td>0.092434</td>
      <td>0.212790</td>
      <td>0.077479</td>
      <td>0.255031</td>
      <td>0.006371</td>
      <td>0.489620</td>
      <td>0.028672</td>
    </tr>
    <tr>
      <th>1</th>
      <td>image-scraper/images/chihuahua/01ee02c2fb.jpg</td>
      <td>chihuahua</td>
      <td>0.357992</td>
      <td>0.128554</td>
      <td>0.227736</td>
      <td>0.652591</td>
      <td>0.014283</td>
      <td>0.092680</td>
      <td>0.049545</td>
      <td>0.319636</td>
      <td>0.483190</td>
      <td>0.883061</td>
      <td>0.594676</td>
      <td>1.381473</td>
      <td>0.026144</td>
      <td>0.065988</td>
      <td>0.725521</td>
      <td>0.713691</td>
      <td>0.325972</td>
      <td>0.533582</td>
      <td>0.180176</td>
      <td>0.342377</td>
      <td>0.237265</td>
      <td>0.000000</td>
      <td>0.261737</td>
      <td>0.221033</td>
      <td>0.032584</td>
      <td>0.351651</td>
      <td>0.014421</td>
      <td>0.414543</td>
      <td>0.000000</td>
      <td>0.866484</td>
      <td>0.107451</td>
      <td>0.203948</td>
      <td>0.368253</td>
      <td>0.057964</td>
      <td>0.730031</td>
      <td>1.100734</td>
      <td>0.018189</td>
      <td>0.685484</td>
      <td>...</td>
      <td>0.083613</td>
      <td>0.029979</td>
      <td>0.002172</td>
      <td>0.000702</td>
      <td>0.414535</td>
      <td>0.217415</td>
      <td>2.571724</td>
      <td>0.424689</td>
      <td>1.096397</td>
      <td>0.501114</td>
      <td>0.079659</td>
      <td>0.196180</td>
      <td>0.939940</td>
      <td>0.698140</td>
      <td>0.201000</td>
      <td>0.156991</td>
      <td>0.007649</td>
      <td>0.506530</td>
      <td>0.033568</td>
      <td>0.236135</td>
      <td>0.247923</td>
      <td>0.000000</td>
      <td>0.019695</td>
      <td>0.024427</td>
      <td>1.090950</td>
      <td>0.059133</td>
      <td>0.157476</td>
      <td>0.020930</td>
      <td>0.278073</td>
      <td>0.261410</td>
      <td>0.061090</td>
      <td>0.526585</td>
      <td>2.363333</td>
      <td>0.160860</td>
      <td>0.000000</td>
      <td>0.008739</td>
      <td>0.401080</td>
      <td>1.377396</td>
      <td>0.383463</td>
      <td>0.434211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>image-scraper/images/chihuahua/040df01fb4.jpg</td>
      <td>chihuahua</td>
      <td>0.163307</td>
      <td>0.383920</td>
      <td>0.029491</td>
      <td>0.985439</td>
      <td>0.866045</td>
      <td>0.098337</td>
      <td>0.000000</td>
      <td>0.634066</td>
      <td>0.008103</td>
      <td>0.265017</td>
      <td>0.855049</td>
      <td>0.377369</td>
      <td>0.956488</td>
      <td>0.048767</td>
      <td>0.111992</td>
      <td>0.995532</td>
      <td>0.045995</td>
      <td>0.000000</td>
      <td>0.100024</td>
      <td>0.662924</td>
      <td>0.000000</td>
      <td>0.622694</td>
      <td>1.983342</td>
      <td>0.409406</td>
      <td>0.135824</td>
      <td>1.269810</td>
      <td>0.000000</td>
      <td>0.938176</td>
      <td>0.231427</td>
      <td>0.952571</td>
      <td>0.093252</td>
      <td>0.891095</td>
      <td>0.095773</td>
      <td>0.074930</td>
      <td>2.476267</td>
      <td>0.419716</td>
      <td>0.139721</td>
      <td>0.809897</td>
      <td>...</td>
      <td>1.201412</td>
      <td>0.007480</td>
      <td>0.344691</td>
      <td>0.113256</td>
      <td>0.045065</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010362</td>
      <td>0.000000</td>
      <td>0.051770</td>
      <td>0.089685</td>
      <td>0.875670</td>
      <td>0.746090</td>
      <td>0.968278</td>
      <td>0.339250</td>
      <td>0.000000</td>
      <td>0.088543</td>
      <td>0.181245</td>
      <td>0.727058</td>
      <td>0.932653</td>
      <td>0.055956</td>
      <td>0.388373</td>
      <td>0.000000</td>
      <td>0.027907</td>
      <td>1.606237</td>
      <td>0.018162</td>
      <td>1.293559</td>
      <td>0.025541</td>
      <td>0.134964</td>
      <td>0.907378</td>
      <td>0.188045</td>
      <td>0.000000</td>
      <td>0.056569</td>
      <td>1.115317</td>
      <td>0.000000</td>
      <td>0.005085</td>
      <td>0.072279</td>
      <td>0.555854</td>
      <td>0.333002</td>
      <td>0.413305</td>
    </tr>
    <tr>
      <th>3</th>
      <td>image-scraper/images/chihuahua/04d8487a97.jpg</td>
      <td>chihuahua</td>
      <td>0.206925</td>
      <td>3.128514</td>
      <td>0.147507</td>
      <td>0.104672</td>
      <td>0.554030</td>
      <td>2.415103</td>
      <td>0.009965</td>
      <td>0.171641</td>
      <td>0.023494</td>
      <td>0.093665</td>
      <td>1.055174</td>
      <td>0.416101</td>
      <td>0.000000</td>
      <td>0.035469</td>
      <td>0.795537</td>
      <td>0.347654</td>
      <td>0.110582</td>
      <td>0.966725</td>
      <td>1.661129</td>
      <td>0.609166</td>
      <td>0.174202</td>
      <td>0.021658</td>
      <td>0.000000</td>
      <td>0.413250</td>
      <td>0.059435</td>
      <td>0.672011</td>
      <td>0.194659</td>
      <td>0.093386</td>
      <td>0.496726</td>
      <td>0.614268</td>
      <td>3.678215</td>
      <td>1.522481</td>
      <td>0.444912</td>
      <td>0.170048</td>
      <td>0.081268</td>
      <td>1.027889</td>
      <td>0.120007</td>
      <td>0.060579</td>
      <td>...</td>
      <td>0.800862</td>
      <td>0.063278</td>
      <td>0.014473</td>
      <td>1.348824</td>
      <td>0.430826</td>
      <td>0.588406</td>
      <td>1.584016</td>
      <td>1.293362</td>
      <td>0.641694</td>
      <td>0.218196</td>
      <td>0.000000</td>
      <td>2.098822</td>
      <td>0.102527</td>
      <td>0.850325</td>
      <td>0.020062</td>
      <td>0.407770</td>
      <td>1.071674</td>
      <td>0.093944</td>
      <td>0.374237</td>
      <td>0.304633</td>
      <td>0.035681</td>
      <td>2.885260</td>
      <td>0.085211</td>
      <td>0.517855</td>
      <td>1.634241</td>
      <td>0.173959</td>
      <td>2.049943</td>
      <td>0.297248</td>
      <td>0.594727</td>
      <td>0.031904</td>
      <td>0.000000</td>
      <td>1.297840</td>
      <td>1.165449</td>
      <td>0.562891</td>
      <td>0.000000</td>
      <td>0.395751</td>
      <td>0.250796</td>
      <td>0.295067</td>
      <td>0.534072</td>
      <td>0.051334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>image-scraper/images/chihuahua/0d9fa44dea.jpg</td>
      <td>chihuahua</td>
      <td>0.233232</td>
      <td>0.355026</td>
      <td>0.453335</td>
      <td>0.060354</td>
      <td>0.479403</td>
      <td>0.000000</td>
      <td>0.099391</td>
      <td>0.223717</td>
      <td>0.000000</td>
      <td>0.166825</td>
      <td>1.043408</td>
      <td>0.212968</td>
      <td>0.047919</td>
      <td>0.011581</td>
      <td>0.824220</td>
      <td>1.071727</td>
      <td>0.322423</td>
      <td>2.238094</td>
      <td>0.036986</td>
      <td>0.051276</td>
      <td>0.000000</td>
      <td>0.119234</td>
      <td>0.375851</td>
      <td>0.939742</td>
      <td>0.512171</td>
      <td>0.052703</td>
      <td>0.000000</td>
      <td>0.405724</td>
      <td>0.000000</td>
      <td>0.796944</td>
      <td>0.146392</td>
      <td>0.609674</td>
      <td>0.342722</td>
      <td>0.119698</td>
      <td>0.300473</td>
      <td>1.781652</td>
      <td>0.003325</td>
      <td>0.372860</td>
      <td>...</td>
      <td>0.170190</td>
      <td>0.019021</td>
      <td>0.175870</td>
      <td>0.058903</td>
      <td>0.195530</td>
      <td>0.070747</td>
      <td>1.904165</td>
      <td>0.143940</td>
      <td>1.018590</td>
      <td>0.010964</td>
      <td>0.008130</td>
      <td>1.318745</td>
      <td>1.122194</td>
      <td>1.022019</td>
      <td>0.679122</td>
      <td>0.080073</td>
      <td>0.821368</td>
      <td>0.255947</td>
      <td>0.000000</td>
      <td>0.811980</td>
      <td>0.124174</td>
      <td>0.211784</td>
      <td>0.911710</td>
      <td>0.057748</td>
      <td>2.242502</td>
      <td>0.352429</td>
      <td>1.633789</td>
      <td>0.283629</td>
      <td>0.340727</td>
      <td>0.034955</td>
      <td>0.308504</td>
      <td>0.376597</td>
      <td>1.075248</td>
      <td>0.416980</td>
      <td>0.073678</td>
      <td>0.316827</td>
      <td>0.620356</td>
      <td>0.125714</td>
      <td>0.179846</td>
      <td>0.110404</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2050 columns</p>
</div>



The usage of `export_class_names=True` will add a new column to the dataframe with the classes names.

## Examples
### Cats vs Dogs using Keras vs `deepfeatx`
First let's compare the code of one of the simplest deep learning libraries (Keras) with `deepfeatx`. As example, let's use a subset of Cats vs Dogs:

```
from deepfeatx.image import download_dataset
download_dataset('https://github.com/dl7days/datasets/raw/master/cats-dogs-data.zip', 'cats-dogs-data.zip')
```

    Downloading Dataset...
    Unzipping Dataset
    Removing .zip file


Here's the keras implementation for a great performance result:

```
from keras.models import Sequential
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

TARGET_SHAPE = (224, 224, 3)
TRAIN_PATH = 'cats-dogs-data/train'
VALID_PATH = 'cats-dogs-data/valid'

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_directory(TRAIN_PATH, 
                                        target_size=TARGET_SHAPE[:2], 
                                        class_mode='sparse')
valid_gen = datagen.flow_from_directory(VALID_PATH, 
                                        target_size=TARGET_SHAPE[:2], 
                                        class_mode='sparse',
                                        shuffle=False)

base_model = ResNet50(include_top=False, input_shape=TARGET_SHAPE)

for layer in base_model.layers:
    layer.trainable=False
    
model = Sequential([base_model,
                    GlobalAveragePooling2D(),
                    Dense(1024, activation='relu'),
                    Dense(2, activation='softmax')])

model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=3, validation_data=valid_gen)
```

    Found 2000 images belonging to 2 classes.
    Found 400 images belonging to 2 classes.
    Epoch 1/3
    63/63 [==============================] - 14s 181ms/step - loss: 0.1664 - accuracy: 0.9313 - val_loss: 0.0709 - val_accuracy: 0.9775
    Epoch 2/3
    63/63 [==============================] - 11s 168ms/step - loss: 0.0213 - accuracy: 0.9946 - val_loss: 0.0457 - val_accuracy: 0.9850
    Epoch 3/3
    63/63 [==============================] - 11s 167ms/step - loss: 0.0061 - accuracy: 1.0000 - val_loss: 0.0453 - val_accuracy: 0.9900





    <tensorflow.python.keras.callbacks.History at 0x7fcc440a9150>



By looking at `val_accuracy` we can confirm the results seems great. Let's also plot some other metrics:

```
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
y_pred = model.predict(valid_gen)
y_test = valid_gen.classes
roc = roc_auc_score(y_test, y_pred[:, 1])
print("ROC AUC Score", roc)
```

    ROC AUC Score 0.9989


```
cm=confusion_matrix(y_test, y_pred.argmax(axis=1))
sns.heatmap(cm, annot=True, fmt='g')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcc41c57090>




![png](docs/images/output_19_1.png)


Although we got an almost perfect clssifier, there are multiple details that someone who is coming from sklearn has to be careful when using Keras, for example:
- Correctly setup the Data Generator
- Fine tune the learning rate
- Adjust the batch size

Now let's replicate the same results using `deepfeatx`:

```
from deepfeatx.image import ImageFeatureExtractor
from sklearn.linear_model import LogisticRegression

TRAIN_PATH = 'cats-dogs-data/train'
VALID_PATH = 'cats-dogs-data/valid'

fe = ImageFeatureExtractor()

train=fe.extract_features_from_directory(TRAIN_PATH, 
                                         classes_as_folders=True,
                                         export_class_names=True)
test=fe.extract_features_from_directory(VALID_PATH, 
                                         classes_as_folders=True,
                                         export_class_names=True)

X_train, y_train = train.drop(['filepaths', 'classes'], axis=1), train['classes']
X_test, y_test = test.drop(['filepaths', 'classes'], axis=1), test['classes']
lr = LogisticRegression().fit(X_train, y_train)
```

    Found 2000 images belonging to 2 classes.
    63/63 [==============================] - 10s 141ms/step
    Found 400 images belonging to 2 classes.
    13/13 [==============================] - 2s 140ms/step


    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)


```
roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
```




    0.9996



```
import seaborn as sns
cm=confusion_matrix(y_test, lr.predict(X_test))
sns.heatmap(cm, annot=True, fmt='g')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fcc36689a50>




![png](docs/images/output_23_1.png)


Even though the code is smaller, is still as powerful as the keras code and also very flexible. The most important part is the feature extraction, which `deepfeatx` take care for us, and the rest can be performed as any other ML problem.
