# Generating SNP Data for Section 4.1

The data is synthetic SNP data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/COXHAP.

We use data for European ancestries, merging the HapMap3 SNPs across chromosomes and then clumping to reduce dimensions.

## 1. Download Data
### a.
```
wget https://dataverse.harvard.edu/api/access/datafile/6154072 -O EUR_chr10.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154073 -O EUR_chr10.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154079 -O EUR_chr11.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154078 -O EUR_chr11.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154087 -O EUR_chr12.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154088 -O EUR_chr12.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154254 -O EUR_chr13.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154255 -O EUR_chr13.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154273 -O EUR_chr14.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154270 -O EUR_chr14.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154272 -O EUR_chr15.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154271 -O EUR_chr15.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154280 -O EUR_chr16.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154290 -O EUR_chr16.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154282 -O EUR_chr17.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154289 -O EUR_chr17.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154291 -O EUR_chr18.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154281 -O EUR_chr18.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154292 -O EUR_chr19.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154285 -O EUR_chr19.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154043 -O EUR_chr1.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154044 -O EUR_chr1.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154045 -O EUR_chr1.fam
wget https://dataverse.harvard.edu/api/access/datafile/6154284 -O EUR_chr20.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154293 -O EUR_chr20.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154283 -O EUR_chr21.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154286 -O EUR_chr21.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154287 -O EUR_chr22.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154288 -O EUR_chr22.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154046 -O EUR_chr2.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154047 -O EUR_chr2.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154049 -O EUR_chr3.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154048 -O EUR_chr3.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154053 -O EUR_chr4.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154054 -O EUR_chr4.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154061 -O EUR_chr5.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154062 -O EUR_chr5.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154063 -O EUR_chr6.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154064 -O EUR_chr6.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154066 -O EUR_chr7.bed.zip
wget https://dataverse.harvard.edu/api/access/datafile/6154067 -O EUR_chr7.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154068 -O EUR_chr8.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154069 -O EUR_chr8.bim
wget https://dataverse.harvard.edu/api/access/datafile/6154070 -O EUR_chr9.bed
wget https://dataverse.harvard.edu/api/access/datafile/6154071 -O EUR_chr9.bim
```

### b. Unzip
```
unzip '*.zip'
```

### c. Rename unzipped files using python correctly since some don't have the EUR prefix.
```
import glob
import os

for f in glob.glob("*.bed"):        
	if not f.startswith("EUR"):     
		os.system(f"mv {f} EUR_{f}")  
```

## 2.

### a. Replace RSIDs to align with the hapmap file https://zenodo.org/records/7773502:

Align ID formatting
```
import glob
import os

for f in glob.glob("*.bim"):  
	os.system("awk '{split($2, id, \":\"); $2 = id[1]; print}' " + f + " > " + f.replace(".bim", "_id.bim"))  # hacky but works
```

### b. Extract hapmap from all files
```
for f in glob.glob("*.bed"):
	prefix = f.split(".bed")[0]
	os.system(f"/users/achin/snp/plink/plink --bfile {prefix} --bim {prefix}_id.bim --fam EUR_chr1.fam --extract /users/achin/snp/w_hm3.snplist --make-bed --out {prefix}_hapmap3")
```

## 3. Merge files
Create `file_list.txt` as

```
EUR_chr1_hapmap3   
EUR_chr2_hapmap3  
EUR_chr3_hapmap3  
EUR_chr4_hapmap3  
EUR_chr5_hapmap3  
EUR_chr6_hapmap3  
EUR_chr7_hapmap3  
EUR_chr8_hapmap3  
EUR_chr9_hapmap3  
EUR_chr10_hapmap3  
EUR_chr11_hapmap3  
EUR_chr12_hapmap3  
EUR_chr13_hapmap3  
EUR_chr14_hapmap3  
EUR_chr15_hapmap3  
EUR_chr16_hapmap3  
EUR_chr17_hapmap3  
EUR_chr18_hapmap3  
EUR_chr19_hapmap3  
EUR_chr20_hapmap3  
EUR_chr21_hapmap3  
EUR_chr22_hapmap3
```
and run

```
plink  --make-bed  --merge-list file_list.txt --out all_merged
```
## 4. Make simulated coefficients file
```
import pandas as pd
import numpy as np
import glob

np.random.seed(1)

bims = pd.read_table("all_merged.bim", header=None)

with open("EUR_chr1.fam", "r") as f:
    n = len(f.readlines())
p = len(bims)

num_causal_snp = int(p*0.001)
print(num_causal_snp)
coefs = np.zeros(p)  
nonzero_coef_idxs = np.random.choice(p, size=num_causal_snp, replace=False)
coefs[nonzero_coef_idxs] = np.random.normal(size=num_causal_snp)
bims["coef"] = coefs
bims.to_csv("true_coefficients.txt", index=False, header=False, sep=" ")
```

## 5. Simulating phenotype
### a. Run PRS
```
plink --bfile all_merged --score true_coefficients.txt 2 5 7 sum --out simulated_phenotypes
```
### b. Add noise
```
import pandas as pd
import numpy as np

df = pd.read_table("simulated_phenotypes.profile", sep ="\s+")
np.random.seed(2)
df["OBSERVED"] = (df.SCORESUM + np.random.normal(0, 40, size=120000) > 60).astype(int) # binary version with 9.3% prevalance
with open("observed_phenotype.txt", "w") as f: 
	f.write("FID\tIID\tPHENOTYPE\n") # Header 
	for i, p in enumerate(df["OBSERVED"], start=1): 
		f.write(f"{i}\t{i}\t{p}\n")
```

## 6. Clump
```
plink --bfile all_merged --clump reg_results.assoc.logistic  --clump-p1 0.001 --out clumped 
plink --bfile all_merged --extract clumped.clumped --make-bed --out clumped_SNP
```

## 7. Convert into numpy matrix
```
import pandas as pd
import numpy as np
import pickle

from pysnptools.snpreader import Bed  

snp_on_disk = Bed("clumped_SNP", count_A1=True)
snpdata = snp_on_disk.read(dtype='int8',_require_float32_64=False)
X = snpdata.val
y = np.array(pd.read_table("observed_phenotype.txt").PHENOTYPE)

with open("snp_Xy_binary_1379.p", "wb") as f:  
	pickle.dump((X, y),f)
```

