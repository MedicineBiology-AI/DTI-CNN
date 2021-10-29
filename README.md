## DTI-CNN:
A learning-based method for drug-targetinteraction prediction based on feature representation learning and deep neural network

### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./python/run_model.py`: predict drug-target interactions, and evaluate the results with ten cross-validation. 

### All process
1. -Run `compute_similarity.m`

2. -Run `run_joint.m`

3. -Run `run_DAE.py`

4. -Run `run_model.py`



### Code and data
#### `matlab/` directory
- `compute_similarity.m`: compute Jaccard similarity based on interaction/association network
- `joint.m`: splicing the network of drugs and proteins
- `diffusionRWR.m`: network diffusion algorithm (random walk with restart)
- `run_joint.m`: implement the joint and RWR above.

#### `python/` directory
- `au_class.py`: implement the Auto-encoder
- `DAE.py`: implement the DAE
- `run_DAE`: use the dataset to run DAE
- `run_model.py`: predict drug-target interactions, and evaluate the results with ten cross-validation. 


#### `data/` directory
- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of disease names
- `se.txt`: list of side effect names
- `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
- `mat_drug_se.txt` 		: Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_protein_drug.txt` 	: Protein-Drug interaction matrix
- `mat_drug_protein.txt` 	: Drug_Protein interaction matrix (transpose of the above matrix)
- `mat_drug_protein_remove_homo.txt`: Drug_Protein interaction matrix, in which homologous proteins with identity score >40% were excluded (see the paper).
- `mat_drug_drug.txt` 		: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` 	: Drug-Disease association matrix
- `Similarity_Matrix_Drugs.txt` 	: Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` 	: Protein similarity scores based on primary sequences of proteins

**Note**: drugs, proteins, diseases and side-effects are organized in the same order across all files, including name lists, ID mappings and interaction/association matrices.
