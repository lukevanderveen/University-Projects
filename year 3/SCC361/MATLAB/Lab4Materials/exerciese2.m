load fisheriris
X = meas;
Y = species;

Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
Mdl.Prior = [0.5 0.2 0.3];