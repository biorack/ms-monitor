import sys
sys.path.insert(0,'/global/project/projectdirs/metatlas/anaconda/lib/python2.7/site-packages' )
#sys.path.append('/project/projectdirs/metatlas/projects/ms_monitor_tools')
import metatlas_get_data_helper_fun as ma_data

from metatlas import metatlas_objects as metob
from metatlas import h5_query as h5q
from metatlas import gui as mgui
import numpy as np
import time
import os

from IPython.display import display

try:
    import ipywidgets as widgets
except ImportError:
    from IPython.html import widgets

try:
    import qgrid
    qgrid.nbinstall(overwrite=True)
    qgrid.set_grid_option('defaultColumnWidth', 200)
except Exception:
    print('Could not import QGrid')

from datetime import datetime
import pandas as pd
import json
import gspread
from oauth2client.client import SignedJwtAssertionCredentials

from matplotlib import pyplot as plt
import re

def clean_string(oldstr):
    newstr = re.sub('[\[\]]','',oldstr)
    newstr = re.sub('[^A-Za-z0-9+-]+', '_', newstr)
    newstr = re.sub('i_[A-Za-z]+_i_', '', newstr)
    return newstr

def get_rt_mz_tolerance_from_user():
    mz_tolerance = float(raw_input('Enter mz tolerance in ppm (ex "20"): ').replace('ppm',''))
    rt_tolerance = float(raw_input('Enter the retention time uncertainty in minutes (ex "0.3"): '))
    return mz_tolerance, rt_tolerance

def get_blank_qc_pos_neg_string():
    qc_widget = widgets.Text(description="QC ID: ",value='QC')
    blank_widget = widgets.Text(description="Blank ID:",value = 'BLANK')
    pos_widget = widgets.Text(description="Neg ID: ",value='NEG')
    neg_widget = widgets.Text(description="Pos ID:",value = 'POS')
    container = widgets.VBox([widgets.HBox([qc_widget, blank_widget]),widgets.HBox([pos_widget, neg_widget])])

    display(container)
    return qc_widget, blank_widget, pos_widget, neg_widget

def get_files_for_experiment(experiment_name):
    files = metob.retrieve('LcmsRun',username='*',experiment=experiment_name)
    flist = []
    for f in files:
        flist.append(f.hdf5_file)    
    flist = np.unique(flist)
    df = pd.DataFrame()
    for counter,f in enumerate(flist):
        df.loc[counter,'file'] = os.path.basename(f)
#    del df['index']   
    df.set_index('file', drop=True, append=False, inplace=True)
    #df.reset_index(drop=True,inplace=True)
    
    options = qgrid.grid.defaults.grid_options
    options['defaultColumnWidth'] = 600
    #mygrid = qgrid.show_grid(df, remote_js=True,grid_options = options)
    grid = qgrid.grid.QGridWidget(df=df,
                                  precision=6,
                                  grid_options=options,
                                  remote_js=True)

    def handle_msg(widget, content, buffers=None):
        if content['type'] == 'cell_change':
            obj = objects[content['row']]
            try:
                setattr(obj, content['column'], content['value'])
            except Exception:
                pass

    grid.on_msg(handle_msg)
    gui = widgets.Box([grid])
    display(gui)  
    return files

def get_recent_experiments(num_days):
    if not num_days:
        num_days = 5
    query = 'SELECT DISTINCT experiment,creation_time FROM lcmsruns where creation_time >= UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL %d DAY))'%num_days
    entries = [e for e in metob.database.query(query)]
    df = pd.DataFrame() 
    counter = 0
    experiments = []
    for entry in entries:
        if entry['experiment']:
            experiments.append( entry['experiment'] )
    experiments = np.unique(experiments)
    experiment_widget = widgets.Dropdown(
        options=list(experiments),
        value=experiments[0],
        description='Experiments: '
    )
    display(experiment_widget)
    #files = get_files_for_experiment(experiment_widget.value)
    #def experiment_change(trait,value):
    #    files = get_files_for_experiment(value)
    #    return files
    #experiment_widget.on_trait_change(experiment_change,'value')

    return experiment_widget

def get_files_from_recent_experiment(num_days):
    if not num_days:
        num_days = 5
    query = 'SELECT DISTINCT experiment,creation_time,username FROM lcmsruns where creation_time >= UNIX_TIMESTAMP(DATE_SUB(CURDATE(), INTERVAL %d DAY))'%num_days
    entries = [e for e in metob.database.query(query)]
    df = pd.DataFrame() 
    counter = 0
    for entry in entries:
        if entry['experiment']:
            df.loc[counter,'experiment'] = entry['experiment']
            df.loc[counter,'username'] = entry['username']
            df.loc[counter, 'utc time'] = datetime.utcfromtimestamp(entry['creation_time'])
            counter = counter + 1
    #TODO: filter by unique experiment name
    #df.drop_duplicates(cols = 'experiment', inplace = True)
    df.groupby('experiment', group_keys=False)
    options = qgrid.grid.defaults.grid_options
    grid = qgrid.grid.QGridWidget(df=df,
                                  precision=6,
                                  grid_options=options,
                                  remote_js=True)

    def handle_msg(widget, content, buffers=None):
        if content['type'] == 'cell_change':
            obj = objects[content['row']]
            try:
                setattr(obj, content['column'], content['value'])
            except Exception:
                pass

    grid.on_msg(handle_msg)
    return grid    
    #mygrid = qgrid.show_grid(df, remote_js=True,)
    #print "Enter the experiment name here"
    #my_experiment = raw_input()
    #files =  get_files_for_experiment(my_experiment)
    #files = qgrid.get_selected_rows(mygrid)    
    #return files

def get_method_dropdown():
    methods = ['Not Set',
            '6550_HILIC_0.5min_25ppm_500counts',
            '6520_HILIC_0.5min_25ppm_500counts',
            'QE_HILIC_0.5min_25ppm_1000counts',
            '6550_RP_0.5min_25ppm_500counts',
            '6520_RP','QE_RP']
    method_widget = widgets.Dropdown(
        options=methods,
        value=methods[0],
        description='LC-MS Method:'
    )
    display(method_widget)

#    methods_widget.on_trait_change(filter_istd_qc_by_method,'value')

    return method_widget

def get_ms_monitor_reference_data(notebook_name = "20160203 ms-monitor reference data", token='/project/projectdirs/metatlas/projects/google_sheets_auth/ipython to sheets demo-9140f8697062.json', sheet_name = 'QC and ISTD'):
    json_key = json.load(open(token))
    scope = ['https://spreadsheets.google.com/feeds']

    credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

    gc = gspread.authorize(credentials)

    wks = gc.open(notebook_name)
    istd_qc_data = wks.worksheet(sheet_name).get_all_values()
#     blank_data = wks.worksheet('BLANK').get_all_values()
    return istd_qc_data#, blank_data

def filter_istd_qc_by_method(method):
    istd_qc_data = get_ms_monitor_reference_data()
    lc_method = method.split('_')[1]
    ms_method = method.split('_')[0]
    rt_minutes_tolerance = float(method.split('_')[2].replace('min',''))
    mz_ppm_tolerance = float(method.split('_')[3].replace('ppm',''))
    peak_height_minimum = float(method.split('_')[4].replace('counts',''))

    column_names = istd_qc_data[0]
    use_index = []
    permanent_charge_index = []
    for i,cn in enumerate(column_names):
        if (lc_method in cn) and (ms_method in cn):
            use_index.append(i)

    for i,cn in enumerate(column_names):
        if 'Permanent Charge' in cn:
            permanent_charge_index.append(i)
            
    for i,cn in enumerate(column_names):
        if 'COMMON-HILIC' in cn:
            common_hilic_index= i

    for i,cn in enumerate(column_names):
        if 'ISTD-HILIC' in cn:
            istd_hilic_index=i
    
    for i,cn in enumerate(column_names):
        if 'QC-HILIC' in cn:
            qc_hilic_index=i

    def make_dataframe_from_list(present_absent_index,data,rt_mz_intensity_index,permanent_charge_index):
        df = pd.DataFrame()
        counter = 0            
        for compound in data[1:]:
            if compound[present_absent_index] == '1':
                df.loc[counter,'compound'] = compound[0]
                #df.loc[counter,'permanent_charge'] = compound[permanent_charge_index]
                for idx in rt_mz_intensity_index:
                    temp_col = column_names[idx].replace(lc_method,'').replace(ms_method,'').replace('_','')
                    df.loc[counter,temp_col] = compound[idx]
                counter = counter + 1
        df.set_index('compound', drop=True, inplace=True)
        return df

    qc_hilic = make_dataframe_from_list(qc_hilic_index,istd_qc_data,use_index,permanent_charge_index)
    istd_hilic = make_dataframe_from_list(istd_hilic_index,istd_qc_data,use_index,permanent_charge_index)
    common_hilic = make_dataframe_from_list(common_hilic_index,istd_qc_data,use_index,permanent_charge_index)

    istd_hilic = setup_atlas_values(istd_hilic,rt_minutes_tolerance,mz_ppm_tolerance)
    qc_hilic = setup_atlas_values(qc_hilic,rt_minutes_tolerance,mz_ppm_tolerance)
    common_hilic = setup_atlas_values(common_hilic,rt_minutes_tolerance,mz_ppm_tolerance)
    reference_data = {}
    reference_data['qc'] = qc_hilic
    reference_data['common'] = common_hilic
    reference_data['istd'] = istd_hilic
    reference_data['blank'] = istd_hilic
    reference_data['parameters'] = {}
    reference_data['parameters']['rt_minutes_tolerance'] = rt_minutes_tolerance
    reference_data['parameters']['mz_ppm_tolerance'] = mz_ppm_tolerance
    reference_data['parameters']['peak_height_minimum'] = peak_height_minimum

    return reference_data

def setup_atlas_values(df,rt_minutes_tolerance,mz_ppm_tolerance):
    temp_dict = df.to_dict()['RT']
    for compound_name in temp_dict.keys():
        if temp_dict[compound_name]:
            compound = metob.retrieve('Compound',name=compound_name,username='*')[-1]
            pos_mz = compound.mono_isotopic_molecular_weight + 1.007276
            neg_mz = compound.mono_isotopic_molecular_weight - 1.007276
            df.loc[compound_name,'pos_mz'] = pos_mz
            df.loc[compound_name,'neg_mz'] = neg_mz

            #df.loc[compound_name,'pos_mz_min'] = pos_mz - pos_mz * mz_ppm_tolerance / 1e6
            #df.loc[compound_name,'pos_mz_max'] = pos_mz + pos_mz * mz_ppm_tolerance / 1e6
            #df.loc[compound_name,'neg_mz_min'] = neg_mz - neg_mz * mz_ppm_tolerance / 1e6
            #df.loc[compound_name,'neg_mz_max'] = neg_mz + neg_mz * mz_ppm_tolerance / 1e6

            df.loc[compound_name,'rt_min'] = float(temp_dict[compound_name]) - rt_minutes_tolerance
            df.loc[compound_name,'rt_max'] = float(temp_dict[compound_name]) + rt_minutes_tolerance
    return df

def construct_result_table_for_files(files,qc_str,blank_str,neg_str,pos_str,method,reference_data,experiment):
    df = pd.DataFrame()
    counter = 0
    for my_file in files:
    #     if my_file.name == '20160119_pHILIC___POS_MSMS_KZ__KBL_QCMix_V2_______Run80_2.mzML':
        #     finfo = h5q.get_info(my_file.hdf5_file)
        #     num_pos_data = finfo['ms1_pos']['nrows'] + finfo['ms2_pos']['nrows']
        #     num_neg_data = finfo['ms1_neg']['nrows'] + finfo['ms2_neg']['nrows']
        #     do_polarity = []
        #     if num_pos_data > 0:
        #         do_polarity.append('positive')
        #     if num_neg_data > 0:
        #         do_polarity.append('negative')
            #determine if the file is a blank, qc or istd:
            filetype = 'istd'
            if blank_str.value.upper() in my_file.name.upper():
                filetype = 'blank'
            elif (qc_str.value.upper() in my_file.name.upper()):
                filetype = 'qc'

            is_pos = pos_str.value.upper() in my_file.name.upper()
            delta_rt = []
            delta_mz = []
            delta_intensity = []
            detected = []
            atlas = reference_data[filetype]
        #     df.loc[counter, 'num positive data'] = num_pos_data
        #     df.loc[counter, 'num negative data'] = num_neg_data
            for cidx,compound in atlas.iterrows():
                if compound['RT']:
                    mz_ref = metob.MzReference()
                    if is_pos:
                        mz_ref.mz = compound['pos_mz']
                        mz_ref.detected_polarity = 'positive'
                        ref_intensity = 0#float(compound['Peak-HeightPos'])
                    else:
                        mz_ref.mz = compound['neg_mz']
                        mz_ref.detected_polarity = 'negative'
                        ref_intensity = 0#float(compound['Peak-HeightNeg'])
                    mz_ref.mz_tolerance = reference_data['parameters']['mz_ppm_tolerance']
                    mz_ref.mz_tolerance_units = 'ppm'
                    
                    if is_pos:
                        ref_intensity = float(compound['Peak-HeightPos'])
                    else:
                        ref_intensity = float(compound['Peak-HeightNeg'])
                    if ref_intensity>=0:
                        rt_ref = metob.RtReference()
                        rt_ref.rt_peak = compound['RT']
                        rt_ref.rt_min = compound['rt_min']
                        rt_ref.rt_max = compound['rt_max']

                        result = ma_data.get_data_for_a_compound(mz_ref,
                                                rt_ref,[ 'ms1_summary' ],
                                                my_file.hdf5_file,0.3) #extra_time is not used by ms1_summary
                        if result['ms1_summary']['rt_peak']:
                            if result['ms1_summary']['peak_height'] > reference_data['parameters']['peak_height_minimum']:
                                detected.append(1)
                                delta_rt.append( result['ms1_summary']['rt_peak'] - rt_ref.rt_peak )
                                delta_mz.append( (result['ms1_summary']['mz_centroid'] - mz_ref.mz)/mz_ref.mz*1e6 )
                                delta_intensity.append( (result['ms1_summary']['peak_height'] - ref_intensity) / ref_intensity )
                                df.loc[counter,'expected_rt'] = compound['RT']
                                df.loc[counter,'expected_mz'] = mz_ref.mz
                                df.loc[counter,'expected_intensity'] = ref_intensity
                                df.loc[counter,'delta_rt'] = delta_rt[-1]
                                df.loc[counter,'delta_mz'] = delta_mz[-1]
                                df.loc[counter,'delta_intensity'] = delta_intensity[-1]
                                df.loc[counter,'measured_rt'] = result['ms1_summary']['rt_peak']
                                df.loc[counter,'measured_mz'] = result['ms1_summary']['mz_centroid']
                                df.loc[counter,'measured_intensity'] = result['ms1_summary']['peak_height']
                                df.loc[counter, 'name has blank'] = blank_str.value.upper() in my_file.name.upper()
                                df.loc[counter, 'name has QC'] = qc_str.value.upper() in my_file.name.upper()
                                df.loc[counter, 'name has pos'] = pos_str.value.upper() in my_file.name.upper()
                                df.loc[counter, 'name has neg'] = neg_str.value.upper() in my_file.name.upper()
                                df.loc[counter, 'experiment'] = my_file.experiment
                                df.loc[counter, 'filename'] = my_file.name
                                df.loc[counter, 'datestamp'] = my_file.creation_time
                                df.loc[counter, 'utc time'] = datetime.utcfromtimestamp(my_file.creation_time)
                                df.loc[counter, 'lcms method'] = my_file.method #TODO: get instrument and lcms from the method object
                                df.loc[counter, 'sample'] = my_file.sample
                                df.loc[counter, 'username'] = my_file.username
                                df.loc[counter, 'method'] = method.value
                                df.loc[counter,'Compound name'] = cidx
                                if delta_rt:
                                    df.loc[counter,'fraction_rt_pass'] = sum(np.asarray(delta_rt)<0.5) / float(len(atlas))
                                    df.loc[counter,'fraction_mz_pass'] = sum(np.asarray(delta_mz)<5) / float(len(atlas))
                                    df.loc[counter,'fraction_intensity_pass'] = sum(np.asarray(delta_intensity)<0.2) / float(len(atlas))
                                    df.loc[counter,'num_detected'] = sum(detected)

                                else:
                                    df.loc[counter,'fraction_rt_pass'] = 0
                                    df.loc[counter,'fraction_mz_pass'] = 0
                                    df.loc[counter,'num_detected'] = 0
                                    df.loc[counter,'fraction_intensity_pass'] = 0
                                counter = counter + 1
    timestr = time.strftime("%Y%m%d-%H%M%S")
    df.to_excel('%s_%s_%s.xls'%( timestr, clean_string(experiment.value), clean_string(method.value) ) )
    df.to_excel('%s/%s_%s_%s.xls'%( '/project/projectdirs/metatlas/projects/ms_monitor_tools/ms_monitor_logs', timestr,clean_string(experiment.value), clean_string(method.value) ) )

    f = make_compound_plots(df,'QC',pos_str.value,experiment,method)

    f = make_compound_plots(df,'QC',neg_str.value,experiment,method)

    f = make_compound_plots(df,'ISTD',pos_str.value,experiment,method)

    f = make_compound_plots(df,'ISTD',neg_str.value,experiment,method)

    return df


def make_compound_plots(df,plot_type,polarity,experiment,method):
    if plot_type == 'QC':
        compound_names = df[df['name has QC'] == 1]['Compound name'].unique()
    else:
        compound_names = df[df['name has QC'] == 0]['Compound name'].unique()
    counter = 1
    nRows = len(compound_names)
    nCols = 3
    if nRows > 0:
        f, ax = plt.subplots(nRows, nCols, figsize=(6*nCols,nRows * 2)) #original 8 x 6
        print ax.shape
        for i,cname in enumerate(compound_names):
        #     sdf = df[df['Compound name'].str.contains(cname, regex=False) & df['filename'].str.contains("POS") & (df['name has blank'] == 0)  & (df['name has QC'] == 1)]
            if plot_type == 'QC':        
                sdf = df[(df['Compound name'] == cname) & df['filename'].str.contains(polarity, case=False, regex=False) & (df['name has blank'] == 0)  & (df['name has QC'] == 1)]
            else:
                sdf = df[(df['Compound name'] == cname) & df['filename'].str.contains(polarity, case=False, regex=False) & (df['name has blank'] == 0)  & (df['name has QC'] == 0)]

            #     ax[i,0].scatter(sdf['measured_rt'].tolist(), sdf['measured_mz'].tolist())#,c=sdf['delta_intensity'])
            ax[i,0].plot(sdf['measured_rt'].tolist(),'.-')#,c=sdf['delta_intensity'])
            ev = sdf['expected_rt'].tolist()
            if not ev:
                ev = [0,0,0]
            l = ax[i,0].axhline(float(ev[0]),color='black',linewidth=2)
            p = ax[i,0].axhspan(float(ev[0]) - 0.5, float(ev[0]) + 0.5, facecolor='0.5', alpha=0.5)
            
            ax[i,1].plot(sdf['measured_mz'].tolist(),'.-')#,c=sdf['delta_intensity'])
            ev = sdf['expected_mz'].tolist()
            if not ev:
                ev = [0,0,0]
            l = ax[i,1].axhline(float(ev[0]),color='black',linewidth=2)
            p = ax[i,1].axhspan(float(ev[0]) - float(ev[0])*5/1e6, float(ev[0]) + float(ev[0])*5/1e6, facecolor='0.5', alpha=0.5)

            
            ax[i,2].plot(sdf['measured_intensity'].tolist(),'.-')#,c=sdf['delta_intensity'])
            ev = sdf['expected_intensity'].tolist()
            if not ev:
                ev = [0,0,0]
            l = ax[i,2].axhline(float(ev[0]),color='black',linewidth=2)
            p = ax[i,2].axhspan(float(ev[0]) - float(ev[0])*0.25, float(ev[0]) + float(ev[0])*0.25, facecolor='0.5', alpha=0.5)

            # plot measured m/z and rt.  Show a line with shading to indicate expected values.  Mouseover to print name of run.
            # run order plots with trendline.  for now use upload time as a proxy for run order.
            ax[i,0].set_title(cname)
            ax[i,1].set_title(cname)
            ax[i,2].set_title(cname)
            
            ax[i,0].get_yaxis().get_major_formatter().set_useOffset(False)
            ax[i,1].get_yaxis().get_major_formatter().set_useOffset(False)
            ax[i,2].get_yaxis().get_major_formatter().set_useOffset(False)
            
            ax[i,0].set_ylabel('RT (min)')
            ax[i,1].set_ylabel('mz')
            ax[i,2].set_ylabel('peak height')
            
        #     ax[i,0].set_ylabel('mz (ppm)')
        #     ax[i,0].set_title(cname)
        #     ax[i,0].get_xaxis().get_major_formatter().set_useOffset(False)
        #     ax[i,0].get_yaxis().get_major_formatter().set_useOffset(False)
        # http://matplotlib.org/examples/pylab_examples/scatter_hist.html
        plt.tight_layout()
        plt.show()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        f.savefig('/project/projectdirs/metatlas/projects/ms_monitor_tools/ms_monitor_logs/plot_summary_%s_%s_%s_%s_%s.png'%( timestr, plot_type, polarity, clean_string(experiment.value), clean_string(method.value) ) )
        return f

# it takes too long to write to a sheet this way.  need to redo it with getting all cells, updating their values and then sending the data over as a large transfer
#import json
#import gspread
#from oauth2client.client import SignedJwtAssertionCredentials
## def append_result_to_google_sheet(df):
#json_key = json.load(open('/project/projectdirs/metatlas/projects/google_sheets_auth/ipython to sheets demo-9140f8697062.json'))
#scope = ['https://spreadsheets.google.com/feeds']
#credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)
#gc = gspread.authorize(credentials)
#wks = gc.open("lcms_run_log")
#sheet_data = wks.worksheet('active').get_all_values()
##     blank_data = wks.worksheet('BLANK').get_all_values()
#print sheet_data
#keys = df.keys()
#for index,row in df.iterrows():
#    vals = []
#    for i,k in enumerate(keys):
#        vals.append(row[k])
#    wks.worksheet('active').insert_row(vals, index=1)
