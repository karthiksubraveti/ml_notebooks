#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from fastai.tabular import *


# In[ ]:


PATH = 'data/msft/'


# In[ ]:


cols = ['SmartScreen',
 'AVProductStatesIdentifier',
 'AVProductsInstalled',
 'EngineVersion',
 'AvSigVersion',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
 'Census_PrimaryDiskTotalCapacity',
 'AppVersion',
 'Census_TotalPhysicalRAM',
 'Census_OSArchitecture',
 'Processor',
 'Census_IsAlwaysOnAlwaysConnectedCapable',
 'Census_OSVersion',
 'OsBuildLab',
 'Census_SystemVolumeTotalCapacity',
 'Census_OSBuildRevision',
 'Census_IsVirtualDevice',
 'Census_ProcessorCoreCount',
 'Census_OSInstallTypeName',
 'Census_ProcessorModelIdentifier',
 'Census_InternalBatteryType',
 'Census_OEMNameIdentifier',
 'Census_InternalPrimaryDisplayResolutionHorizontal',
 'Census_OEMModelIdentifier',
 'Wdft_IsGamer',
 'Census_OSSkuName',
 'Census_PowerPlatformRoleName',
 'Census_OSBranch',
 'Census_ActivationChannel',
 'IsProtected',
 'CityIdentifier',
 'GeoNameIdentifier',
 'AVProductsEnabled',
 'IeVerIdentifier',
 'Census_OSBuildNumber',
 'CountryIdentifier',
 'Census_InternalPrimaryDisplayResolutionVertical',
 'Census_FirmwareVersionIdentifier',
 'LocaleEnglishNameIdentifier',
 'HasDetections']


# In[ ]:


df = pd.read_csv(f'{PATH}train.csv', usecols= cols, low_memory=False)


# In[ ]:


def is_string_dtype(arr_or_dtype):
    if arr_or_dtype is None:
        return False
    try:
        dtype = arr_or_dtype.dtype
        return dtype.kind in ('O', 'S', 'U') and not pd.api.types.is_period_dtype(dtype)
    except TypeError:
        return False


# In[ ]:


cat_flds = []
for n,c in df.items():
    if is_string_dtype(c): 
        df[n] = c.astype('category').cat.as_ordered()
        cat_flds.append(n)
        
cont_flds = set(cols) - set(cat_flds) - set(['HasDetections'])


# In[ ]:


fillMissing = FillMissing(cat_flds, cont_flds)
fillMissing.apply_train(df)


# In[ ]:


categorify = Categorify(cat_flds, cont_flds)
categorify.apply_train(df)


# In[ ]:


for c in cat_flds:df[c] = df[c].cat.codes


# In[ ]:


y_trn = df["HasDetections"].values
df.drop("HasDetections", axis=1, inplace=True)


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_trn = int(0.9 *len(df))
raw_train, raw_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)


# In[ ]:


m_rf = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=3, max_features=0.6,bootstrap=True, n_jobs=-1, oob_score=True)
m_rf.fit(X_train, y_train)
m_rf.score(X_train,y_train)
m_rf.score(X_valid,y_valid)


# In[ ]:


##Test


# In[ ]:


cols_test = list(set(cols) - set(["HasDetections"]))
cols_test.append("MachineIdentifier")
df_test = pd.read_csv(f'{PATH}test.csv', usecols= cols_test,  low_memory=False)
mach_id = df_test.MachineIdentifier.values
df_test.drop(columns=['MachineIdentifier'], inplace=True)


# In[ ]:


fillMissing.apply_test(df_test)


# In[ ]:


categorify.apply_test(df_test)


# In[ ]:


for c in cat_flds:df_test[c] = df_test[c].cat.codes


# In[ ]:


preds = m_rf.predict(df_test)


# In[ ]:


df_output = pd.DataFrame( {'MachineIdentifier' : mach_id, 'HasDetections' : preds})


# In[ ]:


df_output.to_csv("output_rf", index=False)

