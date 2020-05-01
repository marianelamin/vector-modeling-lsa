## MODEL GENERATION

The generation of a new model is a module totally separate from the Dialogue system. This will generate two files with extensions `.json` and `.pkl` and having the exact same name; they both are needed by the `similarity.py` module in order to use the model and compare the user's and the system's answers.

* Clone the repository 
    ```
   $ git clone https://github.com/emcdona1/dialogue_system.git
   ```
* Move to the main folder
    ```
    $ cd dialogue_system
   ```
### System Setup to Generate a new Model 
1. Install all dependencies/libraries that python needs to generate a model, if you did not do this before.
    See set up on [README.md](https://github.com/emcdona1/dialogue_system/blob/master/README.md) on root folder. 

2. Make sure to have the [all-senate-speech.txt](https://www.dropbox.com/s/rbhpy3qtr5oudr2/all-senate-speeches.txt?dl=0) file in the ```corpus``` folder
   


### Generate a Model
Once you completed all setup steps.  In your CLI, the help command will show you how to use the script to generate the model you need:
 ```
$ python nlu_test/demo_model_generator.py -h 
 ```
```
usage: Demo_model_generator.py [-h] [--action {lsa,docbyterm}]
                               [-min_df MIN_DOCUMENT_FREQUENCY] [-s]
                               [--score {tfidf,zeroone,count}]
                               [-svd_c SVD_COMPONENTS]
                               [-truncate TRUNCATE_MATRIX] [-o OUTPUT]

Generate an LSA model.

optional arguments:
  -h, --help            show this help message and exit
  --action {lsa,docbyterm}
                        Generates a model using the specified action. Latent
                        Semantic Analisys = lsa, or document by term matrix.
                        Default: docbyterm
  -min_df MIN_DOCUMENT_FREQUENCY, --min_document_frequency MIN_DOCUMENT_FREQUENCY
                        When building the vocabulary of words, ignore terms
                        that have a document frequency strictly lower than the
                        given threshold. Default: 1
  -s, --stopwords       Determines whether or not to use stop words when
                        processing the text. Default: False
  --score {tfidf,zeroone,count}
                        Generates a document by term matrix using count
                        (frequency of the word in each document), tfidf
                        (weighted frequency) or zeroone (discrete frequency, 1
                        is term appears, zero if it does not. Default: count
  -svd_c SVD_COMPONENTS, --svd_components SVD_COMPONENTS
                        Number of components that will be used when processing
                        SVD. Default: 100
  -truncate TRUNCATE_MATRIX, --truncate_matrix TRUNCATE_MATRIX
                        Truncates the resulting matrix model. When computing
                        LSA: the (U * Sigma) * VT, only keeping the first
                        -trucate columns of the VT matrix. When computing only
                        term by document matrix: the resulting matrix will
                        only contain the first -truncate columns. En either
                        case, the matrix model will have a size of #totalWords
                        by -truncate. Default: Does not truncate the model
  -o OUTPUT, --output OUTPUT
                        Name of output file, must contain letters, numbers or
                        underscore (_). Default: a descriptive name will be
                        provided containing the options that were selected
                        when the script was executed
```

Once the model is generated, it is stored in a file with extension ```.pkl```.  A second file is created with same name but extension ```.json```.
This last file will contain human readable configuration and other information related to the model generation.

Results will be found : ```dialogue_system/nlu_helper/resources/all_senate_speches/```

### Generate 18 different models with a script
There is an example, which is a script I used to generate all 18 possible models.
Please have a look at ```generate_18_models``` script.
You might need to specify instead of using a conda environment, use the one you desire. So you will have to replace 
``` ~/.conda/envs/dialogue_system/bin/python3.7 ``` by ```python``` or ```python3```.

Once you've edited the script, you can run it by executing:
```
$ ./generate_18_models
```




## LOADING AND USING A MODEL
to be defined . . . 

