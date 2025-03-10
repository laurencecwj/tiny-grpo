import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import json
import sys

with open(sys.argv[1], 'r') as rfl:
    data = json.load(rfl)

df = pd.DataFrame(data)
profile = ProfileReport(df, title="Profiling Report", minimal=True)
profile.to_file(f"{sys.argv[1]}_report.html")