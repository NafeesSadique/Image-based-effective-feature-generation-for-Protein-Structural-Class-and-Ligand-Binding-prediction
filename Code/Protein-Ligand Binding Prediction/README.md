# Instructions
1. "test.csv" 
    - contains protein-ligand feature instance(s) along with headers for Protein-Ligand Binding Prediction
				(you can generate instance(s) of a protein-ligand using Hybrid LBP)

	***run "Protein-Ligand Binding (prediction).py"***


# Information
prediction is show based on majority voting of "Euclidean distance + distance mean" of both distance1 & distance2
			(prediction is possible for other category too by slightly updating the code)

1. training dataset 
   - "resources/HybridLBP_(positive).csv" (Actual Positive Dataset)
					
2. ouput		 
   - "prediction.txt" (predicted binding list)
