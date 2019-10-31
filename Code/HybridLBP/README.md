

# INSTRUCTIONS
	
## Folder Formation
1. input	
   - Class Prediction (Protein PDB)
   - Binding Prediction (Protein PDB + Ligand PDB)
2. output	
   - Class Prediction
   - Binding Prediction
   
## Supported File Extensions				
1. Protein class prediction 
   - ".ent"  
   - ".pdb"
   - ".txt"
2. Binding Prediction
   - "_pro_cg.pdb" (for protein) 
   - "_lig_cg.pdb" (for ligand)
		
***Note: use same file names along with the above mentioned extensions for a single protein-ligand pair***
			(Ex. asd_pro_cg.pdb & asd_lig_cg.pdb => one pair with same name)

### Run "src/main/Execute.java" OR "Execute.jar" from command line
### "Execute.java" has main method. To generate random/clustered undersampled dataset please check the main method 
	

	
## Informations

1. "output\Class Prediction"	
    - contains one "Hybrid_LBP_protein_dataset.csv" file including all feature vectors generated from the protein PDB files from "input\Class Prediction (Protein PDB)" folder

2. "output\Binding Prediction"	
   - contains one "ProteinFeature.csv" file including all feature vectors generated from the protein PDB files from "input\Binding Prediction (Protein PDB + Ligand PDB)" folder
   - contains one "LigandFeature.csv" file including all feature vectors generated from the protein PDB files from "input\Binding Prediction (Protein PDB + Ligand PDB)" folder
   - contains one "HybridLBP_(merged).csv" file including merged feature vectors of Protein Features from "ProteinFeature.csv" file & Ligand Features from  "LigandFeature.csv" file on the same folder
   - contains one "HybridLBP_(random).csv" file including randomly generated negative data with the positive data from "HybridLBP_(merged).csv" file on the same folder
   - contains one "HybridLBP_(cluster).csv" file including clustering based generated negative data with the positive data from "HybridLBP_(merged).csv" file on the same folder						
