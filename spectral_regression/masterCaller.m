function masterCaller()

%Edit these values to change how many patients to evaluate and how many
%points per patient (within each region)
numPatients = 92;
desiredPoints =30;

%Retrieve all X and Y values for 92 patients
%Randomly grab values from 30 points within the mask and 30 points outside
%of the mask

[X, Y] = callCreateXYMatrix(numPatients, desiredPoints);

%Standardize X using 0-to-1 standardization
X = standardizeX(X);

%Initialize all necessary matrices to store test and train data
nbSamplePerPatient = 2 * desiredPoints;
Xtest = [];
Xtrain = [];
Ytrain = [];
Ytruth = [];
totalYhat = [];

%Randomize patients
randIndex = randperm(numPatients);
    
%leave-one-out cross-validation
for j = 1 : numPatients,
    listofTestRows = ((randIndex(j) - 1) * nbSamplePerPatient + 1):(randIndex(j) * nbSamplePerPatient);
    Xtest = X(listofTestRows,:);
    Ytruth = [Ytruth; Y(listofTestRows,:)];
    Xtrain = X;
    Xtrain(listofTestRows,:) = [];
    Ytrain = Y;
    Ytrain(listofTestRows,:) = [];

    %Produce Yhat vector for ROC evaluation
    Yhat = callKSR(Xtest, Xtrain, Ytrain);
    totalYhat = [totalYhat; Yhat];
end

[Xeval, Yeval] = perfcurve(Ytruth, totalYhat, 1);
plot(Xeval,Yeval);
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC for classification by spectral regression');

