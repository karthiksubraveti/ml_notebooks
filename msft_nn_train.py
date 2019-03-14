#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.tabular import *


# In[2]:


PATH = 'data/msft/'


# In[ ]:


cols = ['SmartScreen',
 'MachineIdentifier',        
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





# In[3]:


cat_flds = ['ProductName',
 'EngineVersion',
 'AVProductStatesIdentifier',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches',  
 'Census_PrimaryDiskTotalCapacity',
 'Census_TotalPhysicalRAM',            
 'AppVersion',
 'AvSigVersion',
 'RtpStateBitfield',
 'AVProductsInstalled',
 'OrganizationIdentifier',
 'Platform',
 'Processor',
 'OsVer',
 'OsBuild',
 'OsSuite',
 'OsPlatformSubRelease',
 'OsBuildLab',
 'SkuEdition',
 'AutoSampleOptIn',
 'PuaMode',
 'SMode',
 'SmartScreen',
 'Firewall',
 'UacLuaenable',
 'Census_MDC2FormFactor',
 'Census_DeviceFamily',
 'Census_ProcessorCoreCount',
 'Census_ProcessorManufacturerIdentifier',
 'Census_ProcessorClass',
 'Census_PrimaryDiskTypeName',
 'Census_HasOpticalDiskDrive',
 'Census_ChassisTypeName',
 'Census_PowerPlatformRoleName',
 'Census_InternalBatteryType',
 'Census_OSVersion',
 'Census_OSArchitecture',
 'Census_OSBranch',
 'Census_OSBuildNumber',
 'Census_OSEdition',
 'Census_OSSkuName',
 'Census_OSInstallTypeName',
 'Census_OSInstallLanguageIdentifier',
 'Census_OSWUAutoUpdateOptionsName',
 'Census_IsPortableOperatingSystem',
 'Census_GenuineStateName',
 'Census_ActivationChannel',
 'Census_FlightRing',
 'Census_ThresholdOptIn',
 'Wdft_RegionIdentifier']

cont_flds = ['Census_SystemVolumeTotalCapacity',
 'LocaleEnglishNameIdentifier',
 'IsBeta',             
 'Census_IsSecureBootEnabled',
 'Census_IsWIMBootEnabled',
 'Census_IsVirtualDevice',
 'Census_IsTouchEnabled',
 'Census_IsPenCapable',             
 'IsProtected',
 'HasTpm',
 'AVProductsEnabled',        
 'IsSxsPassiveMode',             
 'Wdft_IsGamer',
 'Census_IsAlwaysOnAlwaysConnectedCapable',
 'Census_IsFlightingInternal',
 'Census_IsFlightsDisabled',             
 'Census_InternalPrimaryDisplayResolutionVertical',
 'CityIdentifier',
 'Census_OSUILocaleIdentifier',
 'GeoNameIdentifier',
 'HasDetections',
 'Census_FirmwareManufacturerIdentifier',
 'CountryIdentifier',
 'Census_ProcessorModelIdentifier',
 'Census_OEMModelIdentifier',
 'Census_InternalBatteryNumberOfCharges',
 'DefaultBrowsersIdentifier',
 'Census_FirmwareVersionIdentifier',
 'IeVerIdentifier',
 'Census_InternalPrimaryDisplayResolutionHorizontal',
 'Census_OSBuildRevision',
 'Census_OEMNameIdentifier']


# In[4]:


df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
for c in cat_flds:
    df_raw[c] = df_raw[c].astype('str')    
df_raw.drop(columns=['MachineIdentifier'], inplace=True)


# In[5]:


cat_flds = list(set(cat_flds).intersection(cols))
cont_flds = list(set(cont_flds).intersection(cols))


# In[6]:


n_trn = int(0.9 * len(df_raw))
val_idx = list(range(n_trn, len(df_raw)))
procs = [FillMissing, Categorify, Normalize]


# In[7]:


dep_var = 'HasDetections'


# In[8]:


def databunch_from_df(path, df:DataFrame, dep_var:str, valid_idx:Collection[int], procs=None,
            cat_names:OptStrList=None, cont_names:OptStrList=None, classes:Collection=None, 
            test_df=None, bs:int=64, val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, 
            device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False)->DataBunch:
    "Create a `DataBunch` from `df` and `valid_idx` with `dep_var`. `kwargs` are passed to `DataBunch.create`."
    cat_names = ifnone(cat_names, []).copy()
    cont_names = ifnone(cont_names, list(set(df)-set(cat_names)-{dep_var}))
    procs = listify(procs)
    src = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                       .random_split_by_pct(0.1, seed=42))
                       #.label_from_df(cols=dep_var, label_cls=FloatList))
    src = src.label_from_df(cols=dep_var) if classes is None else src.label_from_df(cols=dep_var, classes=classes)
    if test_df is not None: src.add_test(TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names,
                                                             processor = src.train.x.processor))
    return src.databunch(path=path, bs=bs, val_bs=val_bs, num_workers=num_workers, device=device, 
                         collate_fn=collate_fn, no_check=no_check)


# In[9]:


data = databunch_from_df(df=df_raw, path=PATH, dep_var=dep_var, cat_names=cat_flds, procs=procs, 
                                 valid_idx=val_idx, bs=128)


# In[10]:


df_raw = None


# In[ ]:


#data.batch_size = 128


# In[ ]:


dir(data)


# In[ ]:


sum([ y for x, y in data.get_emb_szs() ])


# In[ ]:


get_ipython().run_line_magic('pinfo', 'tabular_learner')


# In[ ]:


learner.summary()


# In[11]:


learner = tabular_learner(data, layers=[1000, 10])


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


from fastai.callbacks import *
cbs = [EarlyStoppingCallback(learner), SaveModelCallback(learner)]


# In[ ]:


learner.fit_one_cycle(3, slice(1e-2))


# In[ ]:


plt.plot(learner.recorder.losses)


# In[ ]:


learner.save("msft_model_one_cycle_1150_10_0.04_2_y_ps_final")


# In[ ]:


learner.show_results()


# In[ ]:


data = None


# In[ ]:


learner.purge()


# In[ ]:


learner = None


# # Inference

# In[ ]:


df_test = pd.read_csv(f'{PATH}test.csv', low_memory=False)
mach_id = df_test.MachineIdentifier.values
df_test.drop(columns=['MachineIdentifier'], inplace=True)


# In[ ]:


for c in cat_flds:
    df_test[c] = df_test[c].astype('str')


# In[ ]:


test = TabularList.from_df(df_test, path=PATH, cat_names=cat_flds, cont_names=cont_flds)


# In[ ]:


learner = load_learner(PATH, test=test)


# In[ ]:


pred_val = learner.get_preds(ds_type=DatasetType.Test)


# In[ ]:


df_output = pd.DataFrame( {'MachineIdentifier' : mach_id, 'HasDetections' : [v[1].item() for v in pred_val[0]]})


# In[ ]:


df_output.to_csv("output.csv", index=False)


# In[ ]:
