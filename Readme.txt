Sentiment analysis program

Overview
This is an individual coursework of machine learning module in order to do sentiment analysis on giving dataset and check its performance.
This program based on SVM learning model provided by sklearn package.
It is intended to figure out the author's positive or negative intention for a given dataset.

Dataset
Two dataset are used in this program: IMDb sentiment analysis dataset and opinion lexicon dataset.
For IMDb dataset, it contain train, test, and development dataset. For opinion lexicon dataset,
it contains-as its name-opinion lexicon which means words that points to positive intention and
negative intention. This two folder of dataset need to import into project in order to run it
properly.

Package
This project in run on the base of following package:
1. random
2. numpy as np
3. nltk
4. sklearn
5. operator
6. time
which means these package need to be import to your python environment, otherwise it will not work.

How to run:
1. Import the project in this structure:
-Readme.txt
-part2.py
-opinion-lexicon-English
    -negative-words.txt
    -positive-words.txt
-IMDb
    README.txt
    -train
        imdb_train_pos.txt
        imdb_train_neg.txt
    -test
        imdb_test_pos.txt
        imdb_test_neg.txt
    -dev
        imdb_dev_pos.txt
        imdb_dev_neg.txt
2. In python command line or any linux or mac terminal application, use command "python part2.py" in the project path.
Or you can import the file in any ide and run, it's the same anyway.
3. wait for around 10 mins and you will get the result of classification report on test set.

Notice: Some of the code in lab of this module are used and modify in this program. I will mark it on the code.

Thanks for reading.


Alphver											
Dec 26th, 2019