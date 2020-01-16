Exploration to determine the effect of the top 25 news headlines on the daily price of the Dow Jones Industrial Index.
The outcome to be predicted is either a price increase or decrease, and is therefore binary.

I experimented with simple count representations of the word vectors (nrgram), TF-IDF, as well as different types of word embeddings including Word2Vec (pretrained on Google News headlines).

After experimenting further with different types of algorithms, including Logistic Regression, Random Forest, SVM and MLP, overall it seemed that the simple logistic regression model with the nrgram (n = 2) representation with performed the best (Acc: .571).
