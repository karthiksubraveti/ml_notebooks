#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.tabular import *


# In[ ]:


PATH = 'data/msft/'


# In[ ]:


cat_flds = ['ProductName',
 'EngineVersion',
 'AVProductStatesIdentifier',
 'Census_InternalPrimaryDiagonalDisplaySizeInInches',  
 'Census_PrimaryDiskTotalCapacity',
 'Census_TotalPhysicalRAM',            
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


learner = tabular_learner(data, layers=[1000, 10], emb_drop=0.04)


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


from fastai.callbacks import *
cbs = [EarlyStoppingCallback(learner), SaveModelCallback(learner)]


# In[ ]:


learner.fit_one_cycle(1, 1e-2, callbacks=cbs)


# In[ ]:


learner.save("msft_model_one_cycle_1000_1")


# In[ ]:


learner.export()


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


df_output.to_csv("output_march4_1000_drop04.csv", index=False)
