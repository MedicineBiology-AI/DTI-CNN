interaction = load('../data/mat_drug_protein.txt');
drug_feat = load('../DAE/drug_dae_d100.txt');
prot_feat = load('../DAE/protein_dae_d400.txt');

nFold = 10;
shape_train=[];
shape_test=[];
seed = 2;
rng(seed);
Pint = find(interaction); 
Nint = length(Pint); 
Pnoint = find(~interaction);
Pnoint = Pnoint(randperm(length(Pnoint), Nint * 1));
Nnoint = length(Pnoint); 

posFilt = crossvalind('Kfold', Nint, nFold);
negFilt = crossvalind('Kfold', Nnoint, nFold);

	for foldID = 1 : nFold
        i=foldID;
        abc=i-1;
		train_posIdx = Pint(posFilt ~= foldID);
		train_negIdx = Pnoint(negFilt ~= foldID);
		train_idx = [train_posIdx; train_negIdx];
		Ytrain = [ones(length(train_posIdx), 1); zeros(length(train_negIdx), 1)];
        
        shape_train=[shape_train,size(Ytrain,1)];

		test_posIdx =Pint(posFilt == foldID);
		test_negIdx = Pnoint(negFilt == foldID);
		test_idx = [test_posIdx; test_negIdx]; 
		Ytest = [ones(length(test_posIdx), 1); zeros(length(test_negIdx), 1)]; 

		[I, J] = ind2sub(size(interaction), train_idx);
        
        B=drug_feat(I,:);
        C=prot_feat(J,:);
        trainint=[B,C];
        [k,z] = ind2sub(size(interaction), test_idx);
        D = drug_feat(k,:);
        F = prot_feat(z,:);
        testint=[D,F];
        testindex=[k,z];
        shape_test=[shape_test,size(testint,1)];

        dlmwrite(['../cnn_input/label/testlabel/testlabel', num2str(abc), '.txt'],Ytest, '\t');
        dlmwrite(['../cnn_input/train/train', num2str(abc), '.txt'],trainint, '\t');
        dlmwrite(['../cnn_input/test/test', num2str(abc), '.txt'],testint, '\t');
        dlmwrite(['../cnn_input/label/trainlabel/trainlabel', num2str(abc), '.txt'],Ytrain, '\t');

    end
dlmwrite(['../cnn_input/trainrow.txt'],shape_train, '\t');
dlmwrite(['../cnn_input/testrow.txt'],shape_test, '\t');
       

