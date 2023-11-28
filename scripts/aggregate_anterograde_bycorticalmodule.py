# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:50:20 2023

@author: ashwin.bhandiwad
"""
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path

metadata_path = Path('../data')

metadata = pd.read_excel(metadata_path / 'anterograde_annotated_Quanxin.xlsx',sheet_name='cortical input to CP',
                         index_col=0)

l5 = metadata.loc[metadata['injected layer'].isin(['L5','L5 IT','L5 ET', 'L5 IT ET'])]
