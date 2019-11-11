from eolearn.core.eoworkflow import LinearWorkflow #Dependency
from eolearn.core.eodata import FeatureType
import numpy as np
import datetime
from eolearn.core import SaveToDisk
from eolearn.io import S2L1CWCSInput
from sentinelhub import  BBox, CRS, CustomUrlParam
from pandas import DataFrame
from eolearn.core import EOTask, EOPatch, FeatureType, OverwritePermission
from eolearn.io import  ExportToTiff
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
from plotnine import ggplot, aes, geom_line, geom_point, theme_linedraw, theme, scales,geom_area, element_text
import os

class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):        
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool), 
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))
    

class NormalizedDifferenceIndex(EOTask):   
    """
    The tasks calculates user defined Normalised Difference Index (NDI) between two bands A and B as:
    NDI = (A-B)/(A+B).
    """
    def __init__(self, feature_name, band_a, band_b):
        self.feature_name = feature_name
        self.band_a_fetaure_name = band_a.split('/')[0]
        self.band_b_fetaure_name = band_b.split('/')[0]
        self.band_a_fetaure_idx = int(band_a.split('/')[-1])
        self.band_b_fetaure_idx = int(band_b.split('/')[-1])
        
    def execute(self, eopatch):
        band_a = eopatch.data[self.band_a_fetaure_name][..., self.band_a_fetaure_idx]
        band_b = eopatch.data[self.band_b_fetaure_name][..., self.band_b_fetaure_idx]
        
        ndi = (band_a - band_b) / (band_a  + band_b)
        
        eopatch.add_feature(FeatureType.DATA, self.feature_name, ndi[..., np.newaxis])
        
        return eopatch

    
class EuclideanNorm(EOTask):   
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """
    def __init__(self, feature_name, in_feature_name):
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name
    
    def execute(self, eopatch):
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))
        
        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch



def gather_data():
    
    
    cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
    add_clm = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='80m', cm_size_x='80m', 
                           cmask_feature='CLM', # cloud mask name
                           cprobs_feature='CLP' # cloud prob. map name
                          )
    ndvi = NormalizedDifferenceIndex('NDVI', 'BANDS/3', 'BANDS/2')
    add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(), 
                                      'IS_VALID')
    layer = 'BANDS-S2-L1C'
    custom_script = 'return [B02, B03];'
    input_task = S2L1CWCSInput(layer=layer,
                           feature=(FeatureType.DATA, 'BANDS'), 
    custom_url_params={CustomUrlParam.EVALSCRIPT: custom_script},
                           resx='10m', resy='10m',
                           maxcc=.8)
    add_ndvi = S2L1CWCSInput(layer='NDVI')
    save = SaveToDisk('io_example', overwrite_permission=2)#compress_level=1
    workflow = LinearWorkflow(
       input_task,
        add_clm,
        add_ndvi,
        add_sh_valmask,
    
        )
    time_interval = ('2017-01-01', '2017-12-31')
    result = workflow.execute({input_task: {'bbox': roi_bbox, 'time_interval': time_interval},
                           save: {'eopatch_folder': 'eopatch'}})
    return list(result.values())[0].data['NDVI'], list(result.values())[-1].mask['IS_VALID'],  np.array(list(result.values())[0].timestamp)


def ndvi_clean_data(ndvi, mask, time):
    '''returns a cloud free dataframe'''
    
    t, w, h, _ = ndvi.shape 
    ndvi_clean = ndvi.copy()
    ndvi_clean[~mask] = np.nan
    # Calculate means, remove NaN's from means
    ndvi_mean = np.nanmean(ndvi.reshape(t, w * h).squeeze(), axis=1) 
    ndvi_mean_clean = np.nanmean(ndvi_clean.reshape(t, w * h).squeeze(), axis=1)
    time_clean = time[~np.isnan(ndvi_mean_clean)]
    ndvi_mean_clean = ndvi_mean_clean[~np.isnan(ndvi_mean_clean)]
    df  = DataFrame(  {'Time': time_clean, 'NDVI': ndvi_mean_clean})
    
    return df

def graph(df):
    
    graph = (ggplot(data=df,
           mapping=aes(x='Time', y='NDVI'))
         + geom_line(size =2, color = 'green')
         +geom_point()
         +theme_linedraw()
         + theme(axis_text_x= element_text(rotation=45, hjust=1))
         +scales.ylim(0,1)
         + geom_area(fill = "green", alpha = .4)
    )
    return graph



def plot_ndvi(bounding_box, time_interval):
    
    ndvi, mask, time  = gather_data()
    data_frame =  ndvi_clean_data(ndvi,mask,time)
    
    return graph(data_frame)
    
roi_bbox =BBox(bbox=[ 124.8729916 ,  1.3286215,  124.876657, 1.326319], crs=CRS.WGS84)
time_interval = ['2017-01-01', '2017-12-31']  

plot_ndvi(roi_bbox, time_interval)
