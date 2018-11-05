from sampling import *
import pandas as pd
import csv



atta_rows = sample_xml('/scratch/gussteen/final_project/attasidor.xml.bz2', sample_percent=1.0)
atta_df = pd.DataFrame(atta_rows)

atta_sent_count = len(atta_df['sent_id'].unique())
print("Sentences in 8 Sidor:", atta_sent_count)

with open('/scratch/gussteen/final_project/attasidor.csv', 'w+') as f:
    atta_df.to_csv(f, index=False)  

gp2013_rows = sample_xml('/scratch/gussteen/final_project/gp2013.xml.bz2', sample_percent=50937 / 1251479)

gp2013_df = pd.DataFrame(gp2013_rows)

print("Sentences in GP2013:", len(gp2013_rows))

with open('/scratch/gussteen/final_project/gp2013_sample.csv', 'w+') as f:
    gp2013_df.to_csv(f, index=False)