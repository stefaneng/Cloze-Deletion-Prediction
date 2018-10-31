from sampling import *
import pandas as pd
import csv



#atta_rows = sample_xml('/scratch/gussteen/final_project/attasidor.xml.bz2', sample_percent=0.20)
#atta_df = pd.DataFrame(atta_rows)

#atta_sent_count = len(atta_df['sent_id'].unique())
#print("Sentences in 8 Sidor:", atta_sent_count)

#with open('/scratch/gussteen/final_project/attasidor_sample.csv', 'w+') as f:
#    atta_df.to_csv(f, index=False)  

#gp2013_total, gp2013_rows = reservoir_sample('/scratch/gussteen/final_project/gp2013.xml.bz2', 50937)


gp2013_rows = sample_xml('/scratch/gussteen/final_project/gp2013.xml.bz2', sample_percent=50937 / 1251479)
# gp2013_total, gp2013_rows = reservoir_sample('/scratch/gussteen/final_project/gp2013.xml.bz2', 254711)

gp2013_df = pd.DataFrame(gp2013_rows)

with open('/scratch/gussteen/final_project/gp2013_sample.csv', 'w+') as f:
#    csvwriter = csv.DictWriter(f, ["lemma","msd","pos","sent_id","word"])
#    csvwriter.writeheader()
#    csvwriter.writerows(gp2013_rows)
    gp2013_df.to_csv(f, index=False)
    
gp2013_sent_count = len(gp2013_df['sent_id'].unique())
print("Sentences in GP2013:", gp2013_sent_count)
#print("Percentage sampled:", round(gp2013_sent_count / gp2013_total, 2))