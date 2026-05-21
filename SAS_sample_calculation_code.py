import numpy as np 
import pandas as pd
'''
sn = #sensitivity
sp = #specificity
p = #prevalence of disease
w = #width of 95% CI
a_c = ##diseased subjects TPtF”
nl = #sample size for SN
n2 = #sample size for SP
n = #total #subjects N;
'''

def calculate_sample_size(sn, sp, p, w):
    #step 2 calculate TP+FN;
    z = 1.96  # for 95% confidence interval
    a_c = (z**2)*sn*(1 - sn)/(w**2)

    #step 3 calculate Nl;
    n1 = a_c/p
    #round up to the next whole integer;
    n1_rounded = int(np.ceil(n1))

    #step 4 calculate FP+TN
    b_c = (z**2)*sp*(1-sp)/(w**2)

    #step 5 calculate N2;
    n2 = b_c/(1 - p)
    #round up to the next whole integer;
    n2_rounded = int(np.ceil(n2))

    #step 6 get final sample size;
    n = max(n1_rounded, n2_rounded)

    #formatted print, SAS labels
    '''
    print(f"""
    width of      sensitivity   specificity   prevalence of disease
    95% CI
    {w:<14} {sn:<13} {sp:<13} {p}

    sample size for SN     sample size for SP
    {n1_rounded:<22} {n2_rounded}

    total #subjects N
    {n}
    """)
    '''
    return n

#step 1
sn, sp = 0.95, 0.95 #substitute your values for sensitivity, specificity
w = [0.05, 0.10] #widths
diseases = {"Candidiasis": 0.09, "Bacterial Vaginosis": 0.26, "Trichomoniasis": 0.06, "Syphilis": 0.03, "HIV": 0.007, "HCV": 0.008, "Hepatitis B": 0.04}
#candidiasis:  11% (COLOMBIA) http://www.scielo.org.co/scielo.php?pid=S0034-74342012000100002&script=sci_abstract 10% (USA, PREGNANT OWMEN) https://pmc.ncbi.nlm.nih.gov/articles/PMC9329029/#sec1 75% once in a lifetime https://www.sciencedirect.com/science/article/pii/S0882401025000841#sec2 5-9% (India, large study: 2528 women) https://pmc.ncbi.nlm.nih.gov/articles/PMC3478712/
#BV: 26 Worldwide https://www.who.int/news-room/fact-sheets/detail/bacterial-vaginosis 48.6% https://pmc.ncbi.nlm.nih.gov/articles/PMC5558670/ 20-60% https://www.ncbi.nlm.nih.gov/books/NBK459216/ 39.6%(COLOMBIA) http://www.scielo.org.co/scielo.php?pid=S0034-74342012000100002&script=sci_abstract
#Tricho: 6% https://pmc.ncbi.nlm.nih.gov/articles/PMC10713349/ 2% (COLOMBIA)
#Syphillis: 3% https://pmc.ncbi.nlm.nih.gov/articles/PMC2563915/
#HIV: 0.7% https://www.unaids.org/en/resources/fact-sheet
#HCV: 0.8% # https://www.who.int/news-room/fact-sheets/detail/hepatitis-c
#Hepatits B: 4% https://www.who.int/data/gho/indicator-metadata-registry/imr-details/4985

results_d = np.zeros((len(w), len(diseases)))

for width in range(len(w)):
    key_list = list(diseases.keys())
    for disease, prevalence in diseases.items():
        pos = key_list.index(disease)
        #print(f"{disease} Sample Size Calculation:")
        calculate_sample_size(sn, sp, prevalence, w[width])
        results_d[width, pos] = calculate_sample_size(sn, sp, prevalence, w[width])

#Labeled table
df = pd.DataFrame(
    results_d,
    index=[f"w={val}" for val in w],           # rows = widths
    columns=list(diseases.keys())             # columns = disease names
)

print("Summary of Sample Size Calculations:")
print("Specificity:", sp, "Sensitivity:", sn)
print(df)