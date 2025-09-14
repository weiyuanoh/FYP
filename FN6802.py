import pandas as pd 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
from scipy import stats



def gen_normal_random_variables(mu, sigma, size): 
    sample = np.random.normal(mu, sigma, size)
    return sample 


def confidence_interval(sample, mu, size) : 
    sample_mean = np.mean(sample) 
    sample_std = np.std(sample)
    alpha = 0.05

    # first pivot 
    s2 = np.mean((sample - mu)**2)
    chi_upper_1 = stats.chi2.ppf(1-(alpha/2), size)
    chi_lower_1 = stats.chi2.ppf(alpha/2, size)
    ci_1 = ((size*s2)/chi_upper_1, (size*s2)/chi_lower_1)

    # second pivot 
    sample_var = np.var(sample, ddof = 1)
    chi_upper_2 = stats.chi2.ppf(1-(alpha/2), size-1)
    chi_lower_2 = stats.chi2.ppf(alpha/2, size-1)
    lower_2 = (size-1) * sample_var / chi_upper_2
    upper_2 = (size-1) * sample_var / chi_lower_2 
    ci_2 = (lower_2, upper_2)

    # pivot 3 
    xbar = np.mean(sample)
    stat = size * (xbar - mu)**2
    chi_upper_3 = stats.chi2.ppf(1-(alpha/2), 1)
    chi_lower_3 = stats.chi2.ppf(alpha/2, 1)
    lower_3 = stat / chi_upper_3
    upper_3 = stat / chi_lower_3 
    ci_3 = (lower_3, upper_3)

    return ci_1, ci_2, ci_3

def storing(rep=50, size=100, mu=0.0, sigma=2.0, alpha=0.05): 
    rows = []
    for i in range(1,rep+1): 
        sample = np.random.normal(mu, sigma, size)
        ci1, ci2, ci3 = confidence_interval(sample, mu, size)
        
        for pivot, (lo,hi) in [
            ("pivot 1", ci1), 
            ("pivot 2", ci2), 
            ("pivot 3", ci3)]: 
            rows.append({ 
                "method" : pivot, 
                "low": lo,
                "high": hi, 
                "length" : hi - lo 

            })

    df = pd.DataFrame(rows)
    return df
df =storing(rep=50, size=100, mu=0.0, sigma=2.0, alpha=0.05)
print(df)

print(df.groupby("method")["length"].describe())
order = ["pivot 1",
         "pivot 2",
         "pivot 3"]

groups = [df.loc[df.method==m, "length"].to_numpy() for m in order]
fig, ax = plt.subplots()
ax.boxplot(groups, labels=["pivot 1","pivot 2","pivot 3"])
ax.set_ylabel("Interval length")
ax.set_title("CI length distribution by pivot")
plt.tight_layout()
plt.show()