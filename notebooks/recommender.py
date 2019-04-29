import pandas as pd
import numpy as np
import re
import operator
from datetime import datetime
import time

from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score


def get_weekday(strtimestamp, strformat='%Y-%m-%d %H:%M:%S'):

    """
    Returns the weekday from a timestamp
    0: Monday
    6: Sunday

    Input:
    strtimestamp: String with date that will be used to extract the weekday
    strformat: String defining the format os paremeter strtimestmap with default value defined

    Output:
    int weekday

    """
    return datetime.strptime(strtimestamp, strformat).weekday()

def get_day_type(weekday):

    """
    Returns the day type from a weekday
    0,2,4: Type 1 - Even days
    1,3: Type 2 - Odd days
    > 4 - Type 3 - Weekends

    Input:
    weekday: Number with the weekday representation (0= Monday...6=Sunday)
    
    Output:
    int day type

    """

    if weekday in (0,2,4):
        return 0
    elif weekday in (1,3):
        return 1
    else:
        return 2



class SendTimeRecommender:
    
    """
    Class SendTimeRecommender:
    
    Student: Leandro Correa Goncalves

    This is the Send Time Recommender class that is proposed to recommend 
    optimal e-mail send times according to customer's history data

    """
    @staticmethod
    def get_hour_range(self, hra):

        """
        Function to group data into hour-ranges

        Input:
        hra: String with hour which needs to be converted to hour-range

        Output:
        String with hour range

        """

        hour_range = {'00': '00-01', '01': '00-01', '02': '02-03', '03': '02-03', '04': '04-05', '05': '04-05',
                    '06': '06-07', '07': '06-07', '08': '08-09', '09': '08-09', '10': '10-11', '11': '10-11',
                    '12': '12-13', '13': '12-13', '14': '14-15', '15': '14-15', '16': '16-17', '17': '16-17',
                    '18': '18-19', '19': '18-19', '20': '20-21', '21': '20-21', '22': '22-23', '23': '22-23'}

        return hour_range.get(hra)

    @staticmethod
    def fill_empty_hour_range(self, customer_data):

        """
        Function to fill empty hour range within customer data

        Input:
        customer_data: Pandas DataFrame with customer events data

        Output:
        Pandas Series with empty data filled

        """

        cs_data = customer_data

        hour_ranges = ['00-01', '02-03', '04-05', '06-07', '08-09', '10-11','12-13', 
                      '14-15','16-17','18-19', '20-21','22-23']

        for hr in hour_ranges:
            if hr not in cs_data.index:
                cs_data[hr] = 0.0

        return cs_data.sort_index()

    def set_additional_columns(self, df_events_data):

        """
        Set additional information into events datasets that are used in others methods.

        Input:
        df_events_data: Pandas DataFrame with events data

        Output:
        Pandas DataFrame with additional columns

        """

        df_aux = df_events_data

        df_aux['flg_open'] = df_aux['action'].apply(lambda x: 1 if x in ('open', 'open-notification') else 0)

        df_aux['flg_received'] = df_aux['action'].apply(lambda x: 1 if x in ('received', 'receive-notification') else 0)

        df_aux['event_time'] = df_aux['timestamp'].apply(lambda x: re.findall('[0-9]+:[0-9]+:[0-9]+', x)[0])

        df_aux['event_date'] = df_aux['timestamp'].apply(lambda x: x[:10])

        df_aux['event_hour'] = df_aux['event_time'].apply(lambda x: x[0:2])

        df_aux['event_month'] = df_aux['timestamp'].apply(lambda x: x[5:7])

        df_aux['hour_range'] = df_aux['event_hour'].apply(lambda x: self.get_hour_range(self,x))

        df_aux['weekday'] = df_aux['timestamp'].apply(lambda x: get_weekday(x[:19]))

        df_aux['flg_weekend'] = df_aux['weekday'].apply(lambda x: 0 if x < 5 else 1)

        df_aux['day_type'] = df_aux['weekday'].apply(lambda x: get_day_type(x))

        return df_aux

    @staticmethod
    def get_customer_data(df_events):

        """
        Transforms customer data in proportions of open e-mails for each hour-range
        to use on training phase

        Input:
        df_events: Pandas DataFrame with events data

        Output:
        df_customers_wknd: Pandas DataFrame with customers weekend data
        df_customers_reg: Pandas DataFrame with customers non weekdend data (Regular days)

        """

        df_events_aux = df_events

        # weekends
        df_open_events_wknd = df_events_aux[(df_events_aux['flg_open'] == 1) & (df_events_aux['flg_weekend'] == 1)]

        df_open_agg_wknd = df_open_events_wknd[['id',
                                                'hour_range',
                                                'flg_open']].groupby(by=['id', 'hour_range']).sum()

        # regular day

        #Even days
        df_open_events_even = df_events_aux[(df_events_aux['flg_open'] == 1) & (df_events_aux['day_type'] == 0)]
        df_open_agg_even = df_open_events_even[['id',
                                              'hour_range',
                                              'flg_open']].groupby(by=['id', 'hour_range']).sum()

        #Odd days
        df_open_events_odd = df_events_aux[(df_events_aux['flg_open'] == 1) & (df_events_aux['day_type'] == 1)]
        df_open_agg_odd = df_open_events_odd[['id',
                                              'hour_range',
                                              'flg_open']].groupby(by=['id', 'hour_range']).sum()


        total_customer_wknd = df_open_agg_wknd.pivot_table(index='id', values='flg_open', aggfunc=sum)
        total_customer_even = df_open_agg_even.pivot_table(index='id', values='flg_open', aggfunc=sum)
        total_customer_odd = df_open_agg_odd.pivot_table(index='id', values='flg_open', aggfunc=sum)

        # Connecting customer events and total for the proportion calculation
        df_customers_wknd = df_open_agg_wknd.join(total_customer_wknd, rsuffix='_total')
        df_customers_even = df_open_agg_even.join(total_customer_even, rsuffix='_total')
        df_customers_odd = df_open_agg_odd.join(total_customer_odd, rsuffix='_total')

        # Calculating events proportion as customer's open events for each hour range / total of customer's open events

        if (len(df_customers_wknd) > 0) and (len(total_customer_wknd) > 0):
            df_customers_wknd['open_prop'] = df_customers_wknd['flg_open'] / df_customers_wknd['flg_open_total']
        else:
            df_customers_wknd['open_prop'] = 0.0

        if (len(df_customers_even) > 0) and (len(total_customer_even) > 0):
            df_customers_even['open_prop'] = df_customers_even['flg_open'] / df_customers_even['flg_open_total']
        else:
            df_customers_even['open_prop'] = 0.0

        if (len(df_customers_odd) > 0) and (len(total_customer_even) > 0):
            df_customers_odd['open_prop'] = df_customers_odd['flg_open'] / df_customers_odd['flg_open_total']
        else:
            df_customers_odd['open_prop'] = 0.0


        # Setting data format to further clustering algorithm
        df_customers_wknd = df_customers_wknd.pivot_table(index='id', columns='hour_range', values='open_prop',
                                                          fill_value=0.0)
        
        df_customers_even = df_customers_even.pivot_table(index='id', columns='hour_range', values='open_prop',
                                                        fill_value=0.0)

        df_customers_odd = df_customers_odd.pivot_table(index='id', columns='hour_range', values='open_prop',
                                                        fill_value=0.0)


        return df_customers_wknd,df_customers_even,df_customers_odd

    @staticmethod
    def format_customer_data(self, df_events, customer_id, weekday):

        """
        Formats customer data for a given customer_id in proportions of open e-mails for
        each hour-range to use on predicting stage

        Input:
        df_events: Pandas DataFrame with events data
        customer_id: Customer identifier used to return data
        weekday: Int representing weekday (0=Monday...6=Sunday)

        Output:
        df_return: Pandas DataFrame with customer data for the input customer_id and weekday.

        """

        df_events_aux = df_events

        if weekday in (0,2,4):
            df_open_events_weekday = df_events_aux[df_events_aux['day_type'] == 0]
        elif weekday in (1,3):
            df_open_events_weekday = df_events_aux[df_events_aux['day_type'] == 1]
        else:
            df_open_events_weekday = df_events_aux[df_events_aux['day_type'] == 2]

        if len(df_open_events_weekday) > 0:
            df_open_events_all = df_open_events_weekday
        else:
            df_open_events_all = df_events_aux[(df_events_aux['flg_open'] == 1)]

        df_open_agg_all = df_open_events_all[['id', 'hour_range', 'flg_open']].groupby(by=['id', 'hour_range']).sum()

        total_customer_all = df_open_agg_all.pivot_table(index='id', values='flg_open', aggfunc=sum)

        df_customers_all = df_open_agg_all.join(total_customer_all, rsuffix='_total')

        if len(df_customers_all) > 0 and len(total_customer_all) > 0:
            df_customers_all['open_prop'] = df_customers_all['flg_open'] / df_customers_all['flg_open_total']
        else:
            df_customers_all['open_prop'] = 0.0

        df_customers_all = df_customers_all.pivot_table(index='id', columns='hour_range', values='open_prop',
                                                        fill_value=0.0)

        df_return = df_customers_all.loc[customer_id]

        return self.fill_empty_hour_range(self,df_return)

    @staticmethod
    def get_optimal_cluster_number(self, df_events_data):

        """
        Get the optimal cluster number according to Silhouette Method
        to use on the classifier creation.

        Input:
        df_events_data: Pandas DataFrame with events data

        Output:
        Int with optimal cluster number

        """

        # Silhouette score

        scores_model = {}

        if df_events_data.empty:
            return 1
        else:
            for i in np.arange(2, 30):
                cluster = GMM(n_components=i, random_state=0, covariance_type=self.__covariance_type).fit(df_events_data)
                pred = cluster.predict(df_events_data)
                scores_model[i] = round(silhouette_score(df_events_data, pred), 2)

            return max(scores_model.items(), key=operator.itemgetter(1))[0]

        # AIC / # BIC
        """
        n_components = np.arange(2,25)

        models = [gmm(n, covariance_type=self.__covariance_type, random_state=0).fit(df_events_data)
                 for n in n_components]

        score_bic = np.array([m.bic(df_events_data) for m in models]).argmin()
        score_aic = np.array([m.aic(df_events_data) for m in models]).argmin()

        if score_bic < score_aic:
            return score_bic
        else:
            return score_aic 
        """

    @staticmethod
    def get_cluster_classifier(self, df_train_data, n_cluster):

        """
        Returns a trained GMM cluster classifier for given data and number of components(cluster)

        Input:
        df_train_data: Pandas DataFrame with events training data that will be used to fit the classifier
        n_cluster: Number of clusters used in the classifier.

        Output:
        GaussianMixture Classifier from skelearn.mixture

        """

        return GMM(n_components=n_cluster, random_state=0, covariance_type= self.__covariance_type).fit(df_train_data)

    @staticmethod
    def get_cluster_data(self, cluster_classifier, df_events_data):

        """
        Returns clustered (predicted) data for given classifier and data
        to assigning cluster number to the customers

        Input:
        cluster_classifier: GaussianMixture trained classifier
        df_events_data: Pandas DataFrame with events data that needs to be clustered

        Output:
        Array of predicted clusters

        """

        return cluster_classifier.predict(df_events_data)

    def get_cluster_table(self, df_events):

        """
        Returns cluster data tables with hour-range in columns
        and cluster number in rows with average proportions of open
        e-mails in the cells

        Input:
        df_events: Pandas DataFrame with clustered data

        Output:
        Array of predicted clusters

        """
        return df_events.pivot_table(index='cluster')

    def train(self, df_train_data, covariance_type='spherical'):

        """
        Method to train the internal classifiers that uses data from _getCustomerData method.
        This method uses most of private methods to follow the solution flow and
        tests whether data contains weekends and regular days.

        Input:
        df_train_data: Pandas DataFrame with dataset to be trained
        covariance_type: String with the covariance type parameter for clustering algorithm.

        Output:
        None: It fills internal properties of this class.

        """

        start = time.time()

        self.__covariance_type = covariance_type

        # Getting training data
        self.__df_events_wknd, self.__df_events_even, self.__df_events_odd = self.get_customer_data(df_train_data)

        if len(self.__df_events_wknd) > 0:
            self.__n_cluster_wknd = self.get_optimal_cluster_number(self, self.__df_events_wknd)
            self.__model_cluster_wknd = self.get_cluster_classifier(self, self.__df_events_wknd, self.__n_cluster_wknd)
            self.__df_events_wknd['cluster'] = self.get_cluster_data(self, self.__model_cluster_wknd, self.__df_events_wknd)
            self.__cluster_table_wknd = self.get_cluster_table(self.__df_events_wknd)

        if len(self.__df_events_even) > 0:
            self.__n_cluster_even = self.get_optimal_cluster_number(self, self.__df_events_even)
            self.__model_cluster_even = self.get_cluster_classifier(self, self.__df_events_even, self.__n_cluster_even)
            self.__df_events_even['cluster'] = self.get_cluster_data(self, self.__model_cluster_even, self.__df_events_even)
            self.__cluster_table_even = self.get_cluster_table(self.__df_events_even)

        if len(self.__df_events_odd) > 0:
            self.__n_cluster_odd = self.get_optimal_cluster_number(self, self.__df_events_odd)
            self.__model_cluster_odd = self.get_cluster_classifier(self, self.__df_events_odd, self.__n_cluster_odd)
            self.__df_events_odd['cluster'] = self.get_cluster_data(self, self.__model_cluster_odd, self.__df_events_odd)
            self.__cluster_table_odd = self.get_cluster_table(self.__df_events_odd)


        end = time.time()

        print('Time to train: {0}'.format(end - start))

    def predict_customer_cluster(self, customer_id, weekday):

        """
        Predict Customer Cluster for a given customer_id and weekday
        using the trained internal classifier.

        Input:
        customer_id: String with customer id that needs to be predicted
        weekday: Int representing weekday (0=Monday...6=Sunday)

        Output:
        Array of probabilyties of a given customer belongs to each cluster (soft clustering approach).

        """

        customer_data = self.format_customer_data(self, self.__df_events, customer_id, weekday)

        if weekday in (0,2,4):
            customer_probs = self.__model_cluster_even.predict_proba(customer_data.values.reshape(1, -1))
        elif weekday in (1,3):
            customer_probs = self.__model_cluster_odd.predict_proba(customer_data.values.reshape(1, -1))
        else:
            customer_probs = self.__model_cluster_wknd.predict_proba(customer_data.values.reshape(1, -1))


        return customer_probs

    def predict_customer_cluster_from_data(self, customer_data, weekday):

        """
        Predict Customer Cluster for a given customer_id and weekday
        using the trained internal classifier.

        Input:
        customer_data: Pandas DataFrame with customer open events data
        weekday: Int representing weekday (0=Monday...6=Sunday)

        Output:
        Array of probabilyties of a given customer belongs to each cluster (soft clustering approach).

        """
        customer_id = customer_data['id'].iloc[0]
        
        cs_data = self.format_customer_data(self, customer_data, customer_id, weekday)

        if weekday in (0,2,4):
            customer_probs = self.__model_cluster_even.predict_proba(cs_data.values.reshape(1, -1))
        elif weekday in (1,3):
            customer_probs = self.__model_cluster_odd.predict_proba(cs_data.values.reshape(1, -1))
        else:
            customer_probs = self.__model_cluster_wknd.predict_proba(cs_data.values.reshape(1, -1))


        return customer_probs


    def get_customer_table_probs(self, customer_id, weekday, learning_rate):

        """
        Returns customer probabilities tables of opening e-mails in each hour-range.

        Input:
        customer_id: String with customer id that needs to be predicted
        weekday: Int representing weekday (0=Monday...6=Sunday)
        learning_rate: Float value used as weight to exploration x exploitation. This
        parameter controls how much is desired to model learn (explore new time slots)
        or use the known data (exploit).

        Output:
        final_probs: Customer's array of ajusted probabilities of open e-mails for each hour_range
        final_table_prob: final probs in Pandas DataFrame format.
        cluster_table: The cluster table used to ajust probabilities of the given customer.

        """

        customer_probs = self.predict_customer_cluster(customer_id, weekday)

        prob_cluster = np.argsort(-customer_probs)[0][0]
        next_cluster = np.argsort(-customer_probs)[0][1]

        if weekday in (0,2,4):
            filter_values = self.__df_events_even[self.__df_events_even['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_even.predict_proba(filter_values)
            cluster_table = self.__cluster_table_even
        elif weekday in (1,3):
            filter_values = self.__df_events_odd[self.__df_events_odd['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_odd.predict_proba(filter_values)
            cluster_table = self.__cluster_table_odd
        else:
            filter_values = self.__df_events_wknd[self.__df_events_wknd['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_wknd.predict_proba(filter_values)
            cluster_table = self.__cluster_table_wknd

        customer_freq_table = pd.DataFrame(np.argsort(-customer_pred)[:, 1], columns=['freq'])[
            'freq'].value_counts().to_frame()
        next_similar_cluster = customer_freq_table.iloc[0].name

        recommended_clusters = [prob_cluster, next_cluster, next_similar_cluster]

        column_array = cluster_table.loc[recommended_clusters].values.argmax(axis=1)

        # Creating the final probabilities table
        probs_recommendation = []

        for i in range(len(cluster_table.iloc[recommended_clusters[0]])):
            probs_recommendation.append((i, cluster_table.iloc[recommended_clusters[0]][i]))

        final_table_prob = pd.DataFrame(probs_recommendation, columns=['hour', 'prob']).set_index('hour')

        # Updating the probabilities of recommended hour ranges
        # with additional probabilities of next cluster and next similar cluster

        # This is the max hour range of the prob_cluster
        prob_cluster_index = final_table_prob['prob'].idxmax()

        for i in range(3):

            # Getting the index of most probable cluster hour rante to use on learning rate update
            if recommended_clusters[i] == prob_cluster:
                prob_cluster_index = column_array[i]

            actual_value = final_table_prob.iloc[column_array[i]][0]
            new_value = cluster_table.iloc[recommended_clusters[i]][column_array[i]]
            
            #It only updates the values when the new_value (value from another cluster)
            #is greater than the actual value for customer's cluster.
            if new_value > actual_value:
                final_table_prob.iloc[column_array[i]] = cluster_table.iloc[recommended_clusters[i]][column_array[i]]

        # Learning Rate
        for i in range(len(final_table_prob)):
            # If this is the most probable cluster
            if i == prob_cluster_index:
                final_table_prob['prob'].iloc[i] *= (1 - learning_rate)
            else:
                final_table_prob['prob'].iloc[i] *= learning_rate

        final_probs = list(final_table_prob['prob'] / final_table_prob['prob'].sum())

        return final_probs, final_table_prob, cluster_table


    def get_customer_table_probs_from_data(self, customer_data, weekday, learning_rate):

        """
        Returns customer probabilities tables of opening e-mails in each hour-range.

        Input:
        customer_data: Pandas Dataframe with customer open events data
        weekday: Int representing weekday (0=Monday...6=Sunday)
        learning_rate: Float value used as weight to exploration x exploitation. This
        parameter controls how much is desired to model learn (explore new time slots)
        or use the known data (exploit).

        Output:
        final_probs: Customer's array of ajusted probabilities of open e-mails for each hour_range
        final_table_prob: final probs in Pandas DataFrame format.
        cluster_table: The cluster table used to ajust probabilities of the given customer.

        """
        
        customer_probs = self.predict_customer_cluster_from_data(customer_data, weekday)

        prob_cluster = np.argsort(-customer_probs)[0][0]
        next_cluster = np.argsort(-customer_probs)[0][1]

        if weekday in (0,2,4):
            filter_values = self.__df_events_even[self.__df_events_even['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_even.predict_proba(filter_values)
            cluster_table = self.__cluster_table_even
        elif weekday in (1,3):
            filter_values = self.__df_events_odd[self.__df_events_odd['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_odd.predict_proba(filter_values)
            cluster_table = self.__cluster_table_odd
        else:
            filter_values = self.__df_events_wknd[self.__df_events_wknd['cluster'] == prob_cluster].values[:, :12]
            customer_pred = self.__model_cluster_wknd.predict_proba(filter_values)
            cluster_table = self.__cluster_table_wknd

        customer_freq_table = pd.DataFrame(np.argsort(-customer_pred)[:, 1], columns=['freq'])[
            'freq'].value_counts().to_frame()
        next_similar_cluster = customer_freq_table.iloc[0].name

        recommended_clusters = [prob_cluster, next_cluster, next_similar_cluster]

        column_array = cluster_table.loc[recommended_clusters].values.argmax(axis=1)

        # Creating the final probabilities table
        probs_recommendation = []

        for i in range(len(cluster_table.iloc[recommended_clusters[0]])):
            probs_recommendation.append((i, cluster_table.iloc[recommended_clusters[0]][i]))

        final_table_prob = pd.DataFrame(probs_recommendation, columns=['hour', 'prob']).set_index('hour')

        # Updating the probabilities of recommended hour ranges
        # with additional probabilities of next cluster and next similar cluster

        # This is the max hour range of the prob_cluster
        prob_cluster_index = final_table_prob['prob'].idxmax()

        for i in range(3):

            # Getting the index of most probable cluster hour rante to use on learning rate update
            if recommended_clusters[i] == prob_cluster:
                prob_cluster_index = column_array[i]

            actual_value = final_table_prob.iloc[column_array[i]][0]
            new_value = cluster_table.iloc[recommended_clusters[i]][column_array[i]]
            
            #It only updates the values when the new_value (value from another cluster)
            #is greater than the actual value for customer's cluster.
            if new_value > actual_value:
                final_table_prob.iloc[column_array[i]] = cluster_table.iloc[recommended_clusters[i]][column_array[i]]

        # Learning Rate
        for i in range(len(final_table_prob)):
            # If this is the most probable cluster
            if i == prob_cluster_index:
                final_table_prob['prob'].iloc[i] *= (1 - learning_rate)
            else:
                final_table_prob['prob'].iloc[i] *= learning_rate

        final_probs = list(final_table_prob['prob'] / final_table_prob['prob'].sum())

        return final_probs, final_table_prob, cluster_table

    def recommend_send_time(self, customer_id, target_date, learning_rate):

        """
        Recommends a hour-range for a given customer and target date,
        using a specific learning_rate.

        Input:
        customer_id: String with customer id that needs to be predicted

        target_date: The desired date for hour-range recommendation. This parameter
                     is used to map the recommendation classifier (weekend and regular day)

        learning_rate: Float value used as weight to exploration x exploitation. This
                       parameter controls how much is desired to model learn
                       (explore new time slots) or use the known data (exploit).
        

        Output:
        Recommended hour-range index and description.

        """

        weekday = get_weekday(target_date, strformat='%Y-%m-%d')

        final_probs, final_table_prob, cluster_table = self.get_customer_table_probs(customer_id, weekday, learning_rate)

        hour_range_recommended = np.random.choice(final_table_prob.index, p=final_probs)

        return hour_range_recommended, cluster_table.columns[hour_range_recommended]

    
    def recommend_send_time_customer(self, customer_data, target_date, learning_rate):

        """
        Recommends a hour-range for a given customer and target date,
        using a specific learning_rate.

        Input:
        customer_data: Pandas Data Frame with customer open events data

        target_date: The desired date for hour-range recommendation. This parameter
                     is used to map the recommendation classifier (weekend and regular day)

        learning_rate: Float value used as weight to exploration x exploitation. This
                       parameter controls how much is desired to model learn
                       (explore new time slots) or use the known data (exploit).
        

        Output:
        Recommended hour-range index and description.

        """

        weekday = get_weekday(target_date, strformat='%Y-%m-%d')

        final_probs, final_table_prob, cluster_table = self.get_customer_table_probs_from_data(customer_data, weekday, learning_rate)

        hour_range_recommended = np.random.choice(final_table_prob.index, p=final_probs)

        return hour_range_recommended, cluster_table.columns[hour_range_recommended]

    @property
    def formatted_data(self):

        """
        Returns the data formatted to cluster algorithms.

        Input:
        None

        Output:
        Pandas DataFrame with regular events formatted.
        Pandas DataFrame with weekend events formatted.
        """

        return self.__df_events_even, self.__df_events_odd, self.__df_events_wknd

    @property
    def events_data(self):

        """
        Returns data events used on the current instance
        of the class

        Input:
        None

        Output:
        Pandas DataFrame with events data.

        """

        return self.__df_events

    def get_model_classifier(self, weekday):

        """
        Returns the internal weekend or regular day classifier
        according to isWeekend parameter

        Input:
        weekday: Int represeting the weekday (0=Monday... 6=Sunday).

        Output:
        GaussianMixture classifer from sklearn.mixture

        """

        if weekday in (0,2,4):
            return self.__model_cluster_even
        elif weekday in (1,3):
            return self.__model_cluster_odd
        else:
            return self.__model_cluster_wknd

    def get_cluster_matrix(self, weekday):

        """
        Returns the internal cluster Matrix 
        according to isWeekend parameter

        Input:
        weekday: Int number representing the weekday (0=Monday...6=Sunday).

        Output:
        Pandas DataFrame with cluster information

        """

        if weekday in (0,2,4):
            return self.__cluster_table_even
        elif weekday in (1,3):
            return self.__cluster_table_odd
        else:
            return self.__cluster_table_wknd

    def get_cluster_number(self, hour_range, weekday):

        """
        Returns the cluster matrix according to a given hour-range

        Input:
        hour-range: String with the desired hour-range cluster data
        weekday: Int that represents the weekday 0=Monday 6=Sunday

        Output:
        Pandas DataFrame with cluster information for a given hour-range

        """

        return list(self.get_cluster_matrix(weekday).columns).index(hour_range)

    # Dummy Model
    def get_most_frequent(self, customer_id, weekday, df_train):

        """
        Predicts the most frequent time for a given customer_id and day type
        This is the baseline model to recommend a optimal send time.

        Input:
        customer_id: String with desired customer id
        weekday: Int number representing the weekday (0=Monday...6=Sunday)
        df_train: Pandas DataFrame with trained data

        Output:
        String with the most frequent hour-range.

        """
        day_type = get_day_type(weekday)

        open_aux = df_train[(df_train['id'] == customer_id) & (df_train['day_type'] == day_type)]

        if len(open_aux) > 0:
            return open_aux.sort_values('flg_open', ascending=False)[:1]['hour_range'].values[0]
        else:  # most frequent hour range
            return self.__df_events['hour_range'][self.__df_events['day_type'] == day_type].value_counts().index[
                0]

    def multiclass_roc_auc_score(self, y_test, y_pred, average="micro"):

        """
        Returns AUC for multiclass classification

        Input:
        y_test = Array with hour_ranges of test set
        y_pred = Array with predicted hour ranges
        average: micro or macro definiton of average type to evaluation

        Output:
        Float with AUC score
        """
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)

        return roc_auc_score(y_test, y_pred, average=average)

    def recommend_simple_send_time(self, customer_id, weekday):

        """
        Returns the most probable hour range according to a given customer_id and weekday
        It gets the probability table of the cluster of the parameter customer_id
        and suggest a hour range

        Input:
        customer_id: String with the customer_id
        weekday: Int representing the weekday that needs to be use to predict

        Output:
        String with recommended hour range

        """
        cluster = self.predict_customer_cluster(customer_id, weekday).argmax()
        cluster_matrix = self.get_cluster_matrix(weekday).iloc[cluster]
        probs = cluster_matrix.values
        pred_number = np.random.choice(np.arange(len(probs)), p=probs)
        return cluster_matrix.index[pred_number]

    def __init__(self, path_file):

        # Reading data file
        self.__df_events = pd.read_csv(path_file)

        # Creating useful columns
        self.__df_events = self.set_additional_columns(self.__df_events)

        self.__df_events_wknd = None
        self.__df_events_even = None
        self.__df_events_odd = None

        self.__n_cluster_wknd = 0
        self.__n_cluster_even = 0
        self.__n_cluster_odd = 0

        self.__model_cluster_even = None
        self.__model_cluster_odd = None
        self.__model_cluster_wknd = None

        self.__cluster_table_wknd = None
        self.__cluster_table_even = None
        self.__cluster_table_odd = None

        #Clustering covariance type
        self.__covariance_type = 'spherical'
