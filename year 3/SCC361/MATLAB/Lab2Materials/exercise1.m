%bag of words (BoW)

uniqueWords = ["a" "an" "another" "example" ...
"final" "sentence" "third"];

counts = [ ...
1 2 0 1 0 1 0;
0 0 3 1 0 4 6;
1 0 0 5 0 3 1;
1 0 9 1 7 0 0];

sum(counts);

bag = bagOfWords(uniqueWords,counts);

newBag = removeWords(bag,stopWords);

topkwords(bag,7);

topkwords(newBag);

M1 = tfidf(bag);
full(M1);
M2 = tfidf(newBag);
full(M2);

figure(1)
subplot(1,2,1)
wordcloud(bag);
title('Wordcloud')
subplot(1,2,2)
wordcloud(newBag);
title('Refined Wordcloud')



%TF-IDF matrix (Term Frequency–Inverse Document Frequency)
