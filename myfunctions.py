import io
import os.path
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import re
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from itertools import compress
from sklearn.feature_selection import RFE, r_regression, SelectKBest
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import moviepy.video.io.ImageSequenceClip
from natsort import natsorted
from PIL import Image, ImageFilter





#--------------------------------------------#
#--------------------------------------------#
#---------------- USAGE  --------------------# 

#import sys
#sys.path.insert(0, '/home/adeyemi/Documents/mypythonlibrary')
#from myfunctions import odd_even, add

#--------------------------------------------#
#--------------------------------------------#


def read_RF_data(path,S_parameter):

    #print(path)

    data_string = open(path).read()

    ignore_lines = 8
    New_string = data_string.split('\n', ignore_lines)[ignore_lines]
    data_tmp = io.StringIO(New_string)
    df = pd.read_csv(data_tmp, names = ['FREQ.GHZ','S11DB','S11A','S21DB','S21A','S12DB','S12A','S22DB','S22A'], delim_whitespace=True)
    #new_df = df[['FREQ.GHZ',S_parameter]]

    return df


# Absorbance or Transmitance




def read_NIR_data(path, is_transmittance = True):

    if is_transmittance == True:
        file_name = "T0"
        output_path = "/Transmittance.csv"
    else:
        file_name = "A0"
        output_path = "/Absorbance.csv"

    data_string = open(path).read().replace(';', ',')
    temp = data_string.split("\n")[0]


    # ignore the first 7 lines since they dont contain useful information
    ignore_lines = 7
    New_string = data_string.split("\n", ignore_lines)[ignore_lines]
    data_tmp = io.StringIO(New_string)
    df = pd.read_csv(data_tmp, names=['Wavelength', 'Sample', 'Dark', 'Reference', 'Transmittance'])
    new_df = df[['Wavelength', 'Transmittance']]
    return new_df, temp






def subtract_baseline_glucose(df_path_1, df_path_2 , drop_feature_names = ['Round','Temp', 'measurement_type']):

    #check if input is a path or a dataframe
    if type(df_path_1) == str:
        #read path
        df_1_tmp = pd.read_csv(df_path_1)
        df_2_tmp = pd.read_csv(df_path_2)
    else:
        #read dataframe
        df_1_tmp = df_path_1
        df_2_tmp = df_path_2


    feature_name =  list(df_1_tmp.columns)
    #get unique numbers in list
    Rounds = df_1_tmp['Round'].unique()

    output = []
    for round in Rounds:
        df_1 = df_1_tmp[df_1_tmp['Round']== round]
        df_2 = df_2_tmp[df_2_tmp['Round'] == round]
        # assign dataframe to dictionary
        df_dict = {}
        df_dict[1] = df_1
        df_dict[2] = df_2

        #set template column names
        column_names = list(df_1.columns)
        column_names = column_names[:-4]

        # find which dataframe has zero glucose
        if df_dict[1]['glucose_level'].sum() == 0:
            baseline_df = df_1
            glucose_df = df_2

        elif df_dict[2]['glucose_level'].sum() == 0:
            baseline_df = df_2
            glucose_df = df_1

        #drop_feature_names = ['Round','Temp', 'measurement_type']
        tmp_features = glucose_df[drop_feature_names]
        df_1.drop(drop_feature_names, inplace = True, axis = 1)
        df_2.drop(drop_feature_names, inplace = True, axis = 1)

        # find mean of the baseline
        basline_df_mean = baseline_df.iloc[:].mean(axis=0).to_frame().T

        #set temperature to zero
        #basline_df_mean['Temp'] = 0
        num_of_samples = len(glucose_df)

        #create template baseline df the same size as df
        baseline = pd.concat([basline_df_mean] * num_of_samples, ignore_index=True)

        glucose_df.reset_index(drop = True, inplace = True)
        baseline.reset_index(drop=True, inplace=True)
        tmp_features.reset_index(drop=True, inplace=True)

        output_df = pd.DataFrame(glucose_df.values - baseline.values)
        glucose_level = output_df.iloc[: , -1:]
        output_df = output_df.iloc[: , :-1]
        output_df.set_axis(column_names,axis = 1, inplace = True)

        output_df[drop_feature_names] = tmp_features
        output_df['glucose_level'] = glucose_level


        #rename glucose level
        #output_df.columns = [*output_df.columns[:-1], 'glucose_level']
        #output_df['Round'] = tmp_features


        #output_df.iloc[:,0:-2].set_axis(feature_name.pop(2), axis=1, inplace=False)
        output.append(output_df)

    output = pd.concat(output)


    return output



def remove_temp_and_glucose_and_transpose(df, transpose = False):

    '''
    removes the temperature and glucose column, has the option to perform transpose of the dataframe.
    This function is useful when plotting

    :param df:
    :return new_df:
    '''
    new_df = df.drop(['Temp', 'glucose_level'], axis=1)

    if transpose == True:

        new_df = new_df.T

        return new_df
    else:
        return new_df



def Absorbance_2_Transmittance(input, output_type):

    '''

    :param input: Input dataframe either transmission or absorbance
    :param output_type: either "T" or "A"
    :return: return the transformed dataframe
    '''


    if output_type == 'T':
       output = 10**(2-input)


    elif output_type == 'A':
        output = np.log10(input)



    return output


#####################################################################
#####################################################################
#         function to plot glucose concentration
#####################################################################
#####################################################################

def plot_glucose_concentration(all_data, title,save_path= False,ignore_features = None,
                               save_fig = False , bounds = None, plot_type = 'line', ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    color_palets_options = ['black','red','blue','green', 'orange', 'magenta']
    glucose_levels = all_data['glucose_level'].unique()


    num_concentration = len(glucose_levels)
    color_palets = color_palets_options[:num_concentration]



    glucose_level_list = []
    patches = []
    for indx, x in enumerate(glucose_levels):

        tmp = all_data[all_data['glucose_level'] == x ]
        glucose_level_list.append(tmp)


        #ignore the usless features
        if ignore_features != None:
            for ind in ignore_features:
                if ind in tmp.columns:
                    tmp.drop(ind, inplace = True, axis =1)

        tmp.drop('glucose_level', inplace = True, axis = 1)
        colour = color_palets[indx]
        #ax.plot(tmp.columns, tmp.T, color=colour, kind=plot_type)

        new_columns_array = [round(float(x)) for x in tmp.columns.to_numpy()]
        if plot_type == 'line':
            #ax.plot(tmp.columns.to_numpy(), tmp.T.to_numpy(), color=colour)
            ax.plot(new_columns_array, tmp.T.to_numpy(), color=colour)

        elif plot_type == 'scatter':
            ax.plot(tmp.columns.to_numpy(), tmp.T.to_numpy(), color=colour, marker= 'o',markerfacecolor='None', linestyle = 'None')

        patch = mpatches.Patch(color=colour, label='Glucose Level ' + str(x))
        patches.append(patch)


    num_x_tick = round(len(tmp.columns)/25)
    if num_x_tick == 0:
        num_x_tick = 4

    columns_array = tmp.columns.values.tolist()
    new_columns_array = columns_array[0::30]
    new_columns_array = [round(float(x)) for x in new_columns_array]
    font_size = 20
    if ax == None:
        ax.xaxis.set_major_locator(plt.FixedLocator(new_columns_array))
        plt.legend(handles=patches,fontsize=20)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission (dB)', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid()
        plt.title(title)

    else:
        #locator = MaxNLocator
        #ax.xticks(np.arange(min(columns_array), max(columns_array)+1, 1.0))
        #ax.set_xticks(ax.get_xticks()[::2])
        ax.xaxis.set_major_locator(MaxNLocator(num_x_tick,integer=True))
        #ax.xaxis.set_major_locator(ticker.FixedLocator(new_columns_array))
        #ax.set_xticks(np.arange(988.14, 2501.64+1, 1.0), minor=False)
        #ax.set_xticks(np.arange(min(columns_array), max(columns_array)+1, 1.0), minor=False)
        ax.legend(handles=patches,fontsize=20)
        ax.set_xlabel('Wavelength (nm)',fontsize=font_size)
        plt.ylabel('Transmittance (%)', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        ax.grid()
        ax.title.set_text(title)





    if save_fig == True:
        plt.savefig(save_path)

    if bounds is not None:
        x_1 = bounds[0]
        x_2 = bounds[1]
        ax.axvspan(x_1, x_2, alpha=0.3, color='red')

    return ax

    #plt.show()




def find_nearest(array, values, K=1):

    '''
    Returns the index of the nearest k element to the value as list
    :param array:
    :param values:
    :return: indices
    '''
    #indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    #return indices

    #K = 2
    array = [float(value) for value in array]
    array = np.asarray(array)
    X = abs(array - values)
    indexes = sorted(range(len(X)), key=lambda sub: X[sub])[:K]

    if K==1:
        indexes = indexes[0]

    return indexes

def find_nearest_2(array, value):
    '''

    :param array:  np array
    :param value: int/np array

    :return:  Index and value at index
    '''
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]





def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)





def generate_figures_per_round(df,fig_dir,title = 'Round'):
    '''
    :param df:  input dataframe
    :param fig_dir:     directory of output figures
    :return:
    '''

    #Check if directory exist
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)


    #get numb of rounds
    num_rounds = df['Round'].unique()

    # plot all rounds
    #for round in range(0, num_rounds + 1):
    for round in num_rounds:
        #round += 1
        df_new = df[df['Round'] == round]
        save_fig_path = fig_dir + 'Round_' + str(round)
        plot_glucose_concentration(df_new, title=title + str(round), plot_type='line',
                                   ignore_features=['Round', 'measurement_type', 'Temp'],
                                   save_fig=True, save_path=save_fig_path)


def algorithm_pipeline(X_train_data, y_train_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error'
                       ):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)

    '''
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    '''
    return fitted_model




#-------------------------------------------#
#Sort out the time stamps for the glucose data and return the timestamps in min and the midpoints
#-------------------------------------------#
def get_midpoints_in_min(start_time, end_time):

    """
    This function creates a midpoint of the timestamps in minutes

    :param start_time: A dataframe containing 3 columns (hours, min, and sec)
    :param end_time: A dataframe containing 3 columns (hours, min, and sec)

    :return: A dataframe of 3 columns contains (start time, endtime and midpoints)
    """

    start_time_min = []
    end_time_min = []
    midpoint = []
    for i in range(0,len(start_time)):

        if end_time.iloc[i].isnull().all() == True:
            end_time.iloc[i] = start_time.iloc[i]

        #if row is nan, insert nan directly as placeholder for midpoint
        if start_time.iloc[i,0] == 'n' and start_time.iloc[i,1] == 'a' and start_time.iloc[i,2] == 'n':
            start_time_min.append('nan')
            end_time_min.append('nan')
            midpoint.append('nan')
            continue
        #convert time to minutes

        start_time_tmp = float(start_time.iloc[i,0])*60+ float(start_time.iloc[i,1]) + float(start_time.iloc[i,2])/60
        start_time_min.append(start_time_tmp)
        end_time_tmp = float(end_time.iloc[i, 0]) * 60 + float(end_time.iloc[i, 1]) + float(end_time.iloc[i, 2]) / 60
        end_time_min.append(end_time_tmp)

        #calculate midpoint
        midpoint.append(start_time_tmp + (end_time_tmp- start_time_tmp)/2)


    start_time = pd.to_datetime(start_time['Hour'].astype(str) +':'+
                                        start_time['min'].astype(str) +':'+
                                        start_time['sec'].astype(str),format='%H:%M:%S',errors='coerce').dt.time
    end_time = pd.to_datetime(end_time['Hour'].astype(str) +':'+
                                        end_time['min'].astype(str) +':'+
                                        end_time['sec'].astype(str),format='%H:%M:%S',errors='coerce').dt.time


    time_stamps = pd.DataFrame()
    time_stamps['start time'] = start_time
    time_stamps['end time'] = end_time
    time_stamps['midpoint time'] = midpoint

    return time_stamps




def get_indexes(glucose_time, RF_Time_ ,measurement_time, index):
    '''
    Find the two indexes in which define the range between the two time stamps.


    :param glucose_time: time of the glucose/Temp measurements - 1D
    :param RF_Time_:  time of the RF/IR measurements - 1D
    :param measurement_time: the time stamp were interested in - scalar
    :param index: the current index - scalar
    :return:
    '''

    # check if the index is the first element
    if  index == 0: #RF_Time_[index] == RF_Time_[0]:
        index_1 = index
        index_2 = index + 1
    # check if the index is the last element
    elif index == len(RF_Time_):#RF_Time_.shape:#RF_Time_[index] == RF_Time_[-1]:
        index_1 = index - 1
        index_2 = index

    elif measurement_time <= glucose_time[index]:
        index_1 = index - 1
        index_2 = index

    #use adjacent index +1 if the adjecent index is not the last element in the glucose time stamp
    elif measurement_time >= glucose_time[index] and len(glucose_time) > index+1:
        index_1 = index
        index_2 = index + 1

    elif measurement_time >= glucose_time[index] and len(glucose_time) <= index+1:
        index_1 = index -1
        index_2 = index



    return index_1,index_2



def interp_measurement(RF_time, glucose_data):


    '''
    :param RF_data: 1D dataframe
    :param glucose_data: 2D dataframe
    :return:
    '''


    RF_time = RF_time.squeeze().values.tolist()
    glucose_time = glucose_data.iloc[:, 0].values.tolist()
    glucose_data_ = glucose_data.iloc[:, 1].values.tolist()

    new_glucose = []
    for i in range(0, len(RF_time)):
        RF_time_tmp = RF_time[i]
        #print(i)
        # find the interp for glucose
        closest_index = find_nearest(glucose_time, RF_time_tmp, 1)
        index_1, index_2 = get_indexes(glucose_time, RF_time, RF_time_tmp, closest_index)

        new_time = RF_time_tmp
        x_1 = glucose_time[index_1]
        x_2 = glucose_time[index_2]

        y_1 = float(glucose_data_[index_1])
        y_2 = float(glucose_data_[index_2])

        x_points_time = [x_1, x_2]
        y_points_glucose = [y_1, y_2]

        # interpolate to find new temperature value
        new_temp = float(np.interp(new_time, x_points_time, y_points_glucose))
        new_glucose.append(new_temp)

        new_glucose_df = pd.DataFrame(new_glucose, columns=['new glucose'])
    return new_glucose_df




def read_raw_IR_data(path):
    data_string = open(path).read()

    #remove last line of string
    data_string = data_string[:data_string.rfind('\n')]
    New_string_ = data_string.split('\n')

    #Get index and name of LED firing
    positions = [ [i,x ]for i,x in enumerate(New_string_) if 'Firing LED' in x ]


    #iterate over the 3 LEDs
    LED_value = []
    for i in positions:
        index_LED = i[0]+2
        name = i[1]
        New_string = New_string_[index_LED]
        data_tmp = io.StringIO(New_string)
        data_list = list(New_string.split(','))
        data_list = data_list[:-1]

        data_ = [int(x) for x in data_list]

        LED_value.append(data_)
    LED_df = pd.DataFrame(data=LED_value)
    LED_df = LED_df.T
    LED_df.columns = ['LED_1','LED_2','LED_3']
    return LED_df


def remove_outlier(array, std_multiplier =2):

    elements = np.array(array)

    mean = np.median(elements, axis=0)
    sd = np.std(elements, axis=0)

    final_list = [x for x in array if (x > mean -std_multiplier*sd)]
    final_list = [x for x in final_list if (x < mean + std_multiplier * sd)]

    return final_list




class VotingSelector():
   def __init__(self):
       self.selectors = {
           "pearson": self._select_pearson,
           "vif": self._select_vif,
           "rfe": self._select_rfe,
       }
       self.votes = None

   @staticmethod
   def _select_pearson(X, y, **kwargs):
       selector = SelectKBest(r_regression, k=kwargs.get("n_features_to_select", 5)).fit(X, y)
       return selector.get_feature_names_out()

   @staticmethod
   def _select_vif(X, y, **kwargs):
       return [
           X.columns[feature_index]
           for feature_index in range(len(X.columns))
           if variance_inflation_factor(X.values, feature_index) <= kwargs.get("vif_threshold", 10)
       ]

   @staticmethod
   def _select_rfe(X, y, **kwargs):
       svr = SVR(kernel="linear")
       rfe = RFE(svr, n_features_to_select=kwargs.get("n_features_to_select", 5))
       rfe.fit(X, y)
       return rfe.get_feature_names_out()

   def select(self, X, y, voting_threshold=0.5, **kwargs):
       votes = []
       for selector_name, selector_method in self.selectors.items():
           features_to_keep = selector_method(X, y, **kwargs)
           votes.append(
               pd.DataFrame([int(feature in features_to_keep) for feature in X.columns]).T
           )
       self.votes = pd.concat(votes)
       self.votes.columns = X.columns
       self.votes.index = self.selectors.keys()
       features_to_keep = list(compress(X.columns, self.votes.mean(axis=0) > voting_threshold))
       return X[features_to_keep]



#Defining MAPE function
def MARD(Y_actual,Y_Predicted):
    mard = np.mean(np.abs(Y_actual - Y_Predicted)/Y_actual)*100
    return mard

def my_MSE(Y_actual,Y_Predicted):

    Y_actual = Y_actual.to_numpy()
    Y_Predicted = Y_Predicted.to_numpy()
    error = Y_actual-Y_Predicted
    mse = np.square(error).mean()
    return mse


def my_RMSE(Y_actual,Y_Predicted):

    #mse = my_MSE(Y_actual,Y_Predicted)
    RMSE  = np.sqrt(np.mean((Y_actual - Y_Predicted) ** 2))

    return RMSE



def my_linear_plot_regression(y_test,y_pred,metrics=None):

    figure, ax = plt.subplots(figsize=(10,10))

    ax.scatter(y_test,y_pred)



    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))

    ax.plot([p1,p2], [p1,p2], 'b-')
    ax.set_xlabel('True Glucose (mg/dl)',fontsize = 15)
    ax.set_ylabel('Predicted Glucose (mg/dl)',fontsize = 15)
    ax.grid()
    ax.axis('equal')
    ax.set_title('Measurments vs predictions')

    if metrics:
        labels_ = []
        MSE = metrics['MSE']
        MAPE = metrics['MAPE']

        labels_.append("RMSE = {0:.2g}".format(MSE))
        labels_.append("MARD = {0:.2g}%".format(MAPE))


        at = AnchoredText(
            '\n'.join(labels_), prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
    #plt.show()

    return figure



def images_to_video(path, output_name, output_path = 0):

    """
    Function that convert jpeg images to an mp4 video
    THis function takes in 3 parameters, the path of the file and the outputname of the video

    :param path:
    :param output_name:
    :return:
    """
    image_folder = path


    if output_path != 0:
        path_ = output_path
        oputput_video_file = os.path.join(path_ , output_name +'.mp4')
    else:
        oputput_video_file = os.path.join(path , output_name +'.mp4')


    fps = 1

    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files = natsorted(image_files)

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(oputput_video_file)
    
    
    







