clear
strs = extractFileText("stephenfry.txt");
textData = split(strs,newline);

documents = tokenizedDocument(textData); 

length(documents.Vocabulary)

bag = bagOfWords(documents);

newBag = removeWords(bag,stopWords);

topkwords(bag, 10);
topkwords(newBag, 10);

M1 = tfidf(newBag);
full(M1(1:10, 1:10))

figure(1);
subplot(1,1,1);
wordcloud(newBag);
