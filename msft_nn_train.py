#!/usr/bin/env python
# coding: utf-8
# one epoch score - 63.2
# In[1]:


from fastai.tabular import *


# In[2]:


PATH = 'data/msft/'


# In[3]:


cat_flds = ['ProductName',
 'EngineVersion',
 'AppVersion',
 'AvSigVersion',
 'IsBeta',
 'RtpStateBitfield',
 'IsSxsPassiveMode',
 'AVProductsInstalled',
 'AVProductsEnabled',
 'HasTpm',
 'OrganizationIdentifier',
 'Platform',
 'Processor',
 'OsVer',
 'OsBuild',
 'OsSuite',
 'OsPlatformSubRelease',
 'OsBuildLab',
 'SkuEdition',
 'IsProtected',
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
 'Census_IsFlightingInternal',
 'Census_IsFlightsDisabled',
 'Census_FlightRing',
 'Census_ThresholdOptIn',
 'Census_IsSecureBootEnabled',
 'Census_IsWIMBootEnabled',
 'Census_IsVirtualDevice',
 'Census_IsTouchEnabled',
 'Census_IsPenCapable',
 'Census_IsAlwaysOnAlwaysConnectedCapable',
 'Wdft_IsGamer',
 'Wdft_RegionIdentifier']

cont_flds = ['Census_SystemVolumeTotalCapacity',
 'LocaleEnglishNameIdentifier',
 'Census_InternalPrimaryDisplayResolutionVertical',
 'Census_PrimaryDiskTotalCapacity',
 'Census_TotalPhysicalRAM',
 'CityIdentifier',
 'Census_OSUILocaleIdentifier',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
 'GeoNameIdentifier',
 'HasDetections',
 'Census_FirmwareManufacturerIdentifier',
 'CountryIdentifier',
 'Census_ProcessorModelIdentifier',
 'Census_OEMModelIdentifier',
 'Census_InternalBatteryNumberOfCharges',
 'DefaultBrowsersIdentifier',
 'AVProductStatesIdentifier',
 'Census_FirmwareVersionIdentifier',
 'IeVerIdentifier',
 'Census_InternalPrimaryDisplayResolutionHorizontal',
 'Census_OSBuildRevision',
 'Census_OEMNameIdentifier']


# In[ ]:


df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
for c in cat_flds:
    df_raw[c] = df_raw[c].astype('str')    
df_raw.drop(columns=['MachineIdentifier'], inplace=True)


# In[ ]:


n_trn = int(0.9 * len(df_raw))
val_idx = list(range(n_trn, len(df_raw)))
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


dep_var = 'HasDetections'


# In[ ]:


data = TabularDataBunch.from_df(df=df_raw, path=PATH, dep_var=dep_var, cat_names=cat_flds, procs=procs, 
                                 valid_idx=val_idx)


# In[ ]:


df_raw = None


# In[ ]:


# running for first time
learner = tabular_learner(data, layers=[200,100])


# In[ ]:


learner.fit_one_cycle(1, 1e-2)


# In[ ]:


learner.save("msft_model_one_cycle_3epochs")


# In[ ]:


learner.export()


# # inference

# In[ ]:


df_test = pd.read_csv(f'{PATH}test.csv', low_memory=False)
mach_id = df_test.MachineIdentifier.values
df_test.drop(columns=['MachineIdentifier'], inplace=True)


# In[ ]:


test = TabularList.from_df(df_test, path=PATH, cat_names=cat_flds, cont_names=cont_flds)


# In[ ]:


learner = load_learner(PATH, test=test)


# In[ ]:


pred_val = learner.get_preds(ds_type=DatasetType.Test)


# In[ ]:


df_output = pd.DataFrame( {'MachineIdentifier' : mach_id, 'HasDetections' : [v[1].item() for v in pred_val[0]]})


# In[ ]:


df_output.to_csv("output_march3.csv", index=False)

