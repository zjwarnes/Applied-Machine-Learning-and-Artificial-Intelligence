Filestructure: Make sure the input files are both in the same directory. For testing, make sure to change the filename for the test_output_data data frame. 

CSV Output: Since we used the Scikit learn implementation of NB for the Kaggle competition, we did not use the CSV output to predict a test file since it just was taking too much time to predict on 50000 entries. 

Dataframe consistency: To improve upon speed, we convert to a primative array. Because of this, the index is not preserved, hence if you change the indicies of the dataframes, there is no gurantee it will run without extra modifications. 

