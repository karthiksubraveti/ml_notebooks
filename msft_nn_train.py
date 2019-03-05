#!/usr/bin/env python
# coding: utf-8

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


# In[4]:


df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
for c in cat_flds:
    df_raw[c] = df_raw[c].astype('str')    
df_raw.drop(columns=['MachineIdentifier'], inplace=True)


# In[5]:


n_trn = int(0.9 * len(df_raw))
val_idx = list(range(n_trn, len(df_raw)))
procs = [FillMissing, Categorify, Normalize]


# In[6]:


dep_var = 'HasDetections'


# In[7]:


data = TabularDataBunch.from_df(df=df_raw, path=PATH, dep_var=dep_var, cat_names=cat_flds, procs=procs, 
                                 valid_idx=val_idx)


# In[8]:


df_raw = None


# In[28]:


learner = tabular_learner(data, layers=[1000, 10])


# In[29]:


learner = learner.load("msft_model_one_cycle_1000")


# In[30]:


learner.lr_find()


# In[31]:


learner.recorder.plot()


# In[ ]:


# 1 epoch - msft_model_one_cycle (200, 100)
# 2 epochs - msft_model_one_cycle_2epochs (200, 100)
# 3 epochs - msft_model_hidden_100 (100,10)
# 3 epochs - msft_model_hidden_100 (1000,10)
# only works when hidden layer size is same
#learner = learner.load("msft_model_one_cycle_2epochs")


# In[33]:


from fastai.callbacks import *
cbs = [EarlyStoppingCallback(learner), SaveModelCallback(learner)]


# In[36]:


get_ipython().run_line_magic('pinfo', 'learner.fit_one_cycle')


# In[42]:


learner.fit_one_cycle(1, 1e-2, callbacks=cbs)


# In[43]:


learner.save("msft_model_one_cycle_1000_1")


# In[44]:


learner.export()


# # Inference

# In[45]:


df_test = pd.read_csv(f'{PATH}test.csv', low_memory=False)
mach_id = df_test.MachineIdentifier.values
df_test.drop(columns=['MachineIdentifier'], inplace=True)


# In[46]:


for c in cat_flds:
    df_test[c] = df_test[c].astype('str')


# In[47]:


test = TabularList.from_df(df_test, path=PATH, cat_names=cat_flds, cont_names=cont_flds)


# In[48]:


learner = load_learner(PATH, test=test)


# In[49]:


pred_val = learner.get_preds(ds_type=DatasetType.Test)


# In[50]:


df_output = pd.DataFrame( {'MachineIdentifier' : mach_id, 'HasDetections' : [v[1].item() for v in pred_val[0]]})


# In[51]:


df_output.to_csv("output_march4_1000_1.csv", index=False)

