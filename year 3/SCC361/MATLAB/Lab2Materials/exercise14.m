clear
unzip Shakespeare.zip Shakespeare
readFcn = @extractFileText;
fds = fileDatastore('Shakespeare/*.txt','ReadFcn',readFcn);

bag = bagOfWords;
while hasdata(fds)
 str = read(fds);
 document = tokenizedDocument(str);
 bag = addDocument(bag,document);
end

newBag = removeWords(bag, stopWords);
topkwords(newBag, 10)

M1 = tfidf(newBag);
full(M1(1:10, 1:10))

figure(1);
subplot(1,1,1);
wordcloud(newBag);