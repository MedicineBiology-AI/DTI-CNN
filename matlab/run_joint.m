
maxiter = 20;
restartProb = 0.50;

drugNets = {'Sim_mat_drug_drug', 'Sim_mat_drug_disease', 'Sim_mat_drug_se', 'Sim_mat_Drugs'};
proteinNets = {'Sim_mat_protein_protein', 'Sim_mat_protein_disease', 'Sim_mat_Proteins'};

tic
X = joint(drugNets, restartProb, maxiter);
toc
tic
Y = joint(proteinNets, restartProb, maxiter);
toc

dlmwrite(['../feature/drug_vector.txt'], X, '\t');
dlmwrite(['../feature/protein_vector.txt'], Y, '\t');
