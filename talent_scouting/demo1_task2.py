import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import copy
import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import pandasgui as pg
from pandasgui import show

entry2 = None
entry3 = None
entry4 = None

def standardize(dfc):
    assembler = VectorAssembler(inputCols=list(set(dfc.columns) - set(["player_id", "preferred_foot", "work_rate", "position", "sub_position", "age_copy"])), 
                                outputCol="features")

    dfc = assembler.transform(dfc)

    scaler = StandardScaler(inputCol="features", 
                            outputCol="scaled_features",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dfc)

    # Normalize each feature to have unit standard deviation
    dfc = scalerModel.transform(dfc)
    return dfc

def apply_one_hot_encoding(df, column_names, df_id):

    def generate_encoded_columns(df, name_column, column_values):
        return [
            when(df[name_column].isin(value), 1).otherwise(0).alias(f"{name_column}_{value}")
            for value in column_values
        ]
    diz = {}
    for col in column_names:
        diz[col]=[row[col] for row in df.select(col).distinct().collect()]
    encoded_columns = []
    for name_column, column_values in diz.items():
        encoded_columns.extend(generate_encoded_columns(df, name_column, column_values))
    encoded_df = df.select(df_id, *encoded_columns)
    #join the encoded df with the original df on the key df_id
    df = df.join(encoded_df, df_id)
    #drop the columns in column_names
    df = df.drop(*column_names)
    return df

def dataset_assembler(df): 
   # Crea un'istanza di VectorAssembler specificando le colonne di input e la colonna di output
    assembler = VectorAssembler(inputCols=list(set(df.columns) - set(["player_id", "features"])), outputCol="pcaFeatures")

    # Applica il VectorAssembler al DataFrame
    df_pca = assembler.transform(df)
    return df_pca

def get_optimal_k(df, desired_variance_percentage, k):
    
    assembler = VectorAssembler(inputCols=list(set(df.columns) - set(["player_id", "features", "age_copy"])), outputCol="features_")
    df_assembled = assembler.transform(df)

    pca = PCA(k=k, inputCol="features_", outputCol="pcaFeatures")
    model = pca.fit(df_assembled)

    explained_variance = model.explainedVariance.toArray()
    total_variance = np.sum(explained_variance)

    cumulative_variance = 0.0
    num_selected_features = 0

    for variance in explained_variance:
        cumulative_variance += variance
        num_selected_features += 1
        if cumulative_variance >= desired_variance_percentage * total_variance:
            break

    return num_selected_features


def apply_PCA(df, features_name_list, k):

    assembler = VectorAssembler(inputCols=features_name_list, outputCol="features_")
    df_assembled = assembler.transform(df)
    pca = PCA(k=k, inputCol="features_", outputCol="pcaFeatures")
    model = pca.fit(df_assembled)
    df_pca = model.transform(df_assembled).select("player_id", "pcaFeatures")
    
    return df_pca


def flag_PCA(df,flag,dfc): # if flag is True, apply PCA, otherwise apply assembler
    if flag is True:
        desired_variance_percentage = 0.95
        k = len(list(set(df.columns) - set(["player_id", "features", "age_copy"]))) + len(dfc.first()["scaled_features"])
        optimal_k = get_optimal_k(df, desired_variance_percentage, k-1)
        print("I'm applying pca with an optimal k= ", optimal_k)
        df_pca = apply_PCA(df, list(set(df.columns)-{"player_id", "features", "age_copy"}), optimal_k)
    else:
        df_pca = dataset_assembler(df)
        print("I'm not applying pca, but i'm assembling the dataset for k-means")
    return df_pca

def k_means(dataset, n_clusters, distance_measure="euclidean", max_iter=80, features_col="pcaFeatures", prediction_col="cluster", random_seed=42):
  
    print("""k-means parameters: - number of clusters = {:d} - max iterations = {:d} - distance measure = {:s} - random seed = {:d}""".format(n_clusters, max_iter, distance_measure, random_seed))

    # Train a K-means model
    kmeans = KMeans(featuresCol=features_col, 
                    predictionCol=prediction_col, 
                    k=n_clusters, 
                    initMode="k-means||", # or random
                    initSteps=10, 
                    tol=0.0000001, 
                    maxIter=max_iter, 
                    seed=random_seed, 
                    distanceMeasure=distance_measure)
    model = kmeans.fit(dataset)

    # Make clusters
    clusters_df = model.transform(dataset)

    return model, clusters_df

def applying_elbow(df, features_col="pcaFeatures", prediction_col="cluster", min_k=60, max_k=100, random_seed=42):        

    clustering_results = {}
    k_values = range(min_k, max_k)

    silhouette_values = []

    for k in k_values:
        # Creazione dell'istanza del modello K-means
        model, clusters_df = k_means(df, k, features_col="pcaFeatures", prediction_col=prediction_col, random_seed=42)
        clusters_df = clusters_df.withColumn("features", clusters_df["pcaFeatures"])

        # Calcolo del SSE (Sum of Squared Errors)
        silhouette = evaluate_k_means(clusters_df)
        silhouette_values.append(silhouette)
        clustering_results[k] = silhouette

    # Plot del grafico SSE vs. Valori k
    #plt.plot(k_values, silhouette_values, 'bx-')
    #plt.xlabel('Number of Clusters (k)')
    #plt.ylabel('Silhouette')
    #plt.title('Elbow Method for Optimal k')
    #plt.show()

    return clustering_results

def evaluate_k_means(clusters, metric_name="silhouette", distance_measure="squaredEuclidean",prediction_col="cluster"):
  
  # Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator(metricName=metric_name,
                                  distanceMeasure=distance_measure, 
                                  predictionCol=prediction_col
                                  )

  return evaluator.evaluate(clusters)


# split dfc in two dataframes: one composed of player with age >26, the other composed of player with age <=26
def split_dataframe(dfc):
    dfc1 = dfc.filter(dfc.age_copy > 25)
    dfc2 = dfc.filter(dfc.age_copy <= 25)
    return dfc1, dfc2

# def funtion to remove columns in df_k_means1 and df_k_means2
def remove_columns(df_k_means):
    # drop pcaFeatures, features
    df_k_means = df_k_means.drop("pcaFeatures")
    df_k_means = df_k_means.drop("features")
    df_k_means = df_k_means.drop("nationality_name")
    df_k_means = df_k_means.drop("work_rate")
    df_k_means = df_k_means.drop("weight_kg")
    return df_k_means

'''
def search_cluster(df1, df2, player_id): #player_id is related to df1

    # insert in a dictionaty all the values of player_id of df1 except player_id, name, age, cluster
    player_id_dict = {}
    for column in df1.columns:
        if column not in ["player_id", "name", "age", "cluster"]:
            values = df1.filter(col("player_id") == player_id).select(col(column)).collect()
            player_id_dict[column] = values[0][0] if len(values) > 0 else None
    
    clusters_value = []
    no_cluster = df2.select(col("cluster")).distinct().count()
    for i in range(no_cluster):
        # filter the rows where the column "cluster" is equal to i
        df_filtered = df2.filter(col("cluster") == i)
        col_to_drop = ["player_id", "name", "age", "cluster"]
        df_filtered = df_filtered.drop(*col_to_drop)
        clus_values = {}

        # Itera su ogni colonna nel DataFrame
        for column in df_filtered.columns:
            if column not in ["player_id", "name", "age", "cluster"]:
                # check if the column is numeric or not
                if df_filtered.schema[column].dataType in [IntegerType(), FloatType()]:
                    # compute the mean of the column
                    avg = df_filtered.select(mean(col(column))).collect()[0][0]
                    clus_values[column] = avg
                else:
                    # Calcola il valore che appare maggiormente nella colonna non numerica
                    max_value = df_filtered.groupBy(column).count().orderBy(desc("count")).first()[column]
                    clus_values[column] = max_value

        clusters_value.append(clus_values)

    return player_id_dict, clusters_value
'''
def search_cluster(df1, df2, player_id):
    # Converti i DataFrames di PySpark in DataFrames di Pandas per un'elaborazione più veloce
    pdf1 = df1.filter(col("player_id") == player_id).toPandas()
    pdf2 = df2.toPandas()

    # Estrai i dati per player_id da pdf1
    player_id_dict = pdf1.drop(columns=["player_id", "name", "age", "cluster"]).iloc[0].to_dict()

    # Trova i cluster unici in pdf2
    unique_clusters = pdf2["cluster"].unique()
    clusters_value = []

    for cluster in unique_clusters:
        # Filtra le righe in cui la colonna "cluster" è uguale al valore corrente del cluster
        pdf_filtered = pdf2[pdf2["cluster"] == cluster]
        pdf_filtered = pdf_filtered.drop(columns=["player_id", "name", "age", "cluster"])

        # Calcola i valori medi per le colonne numeriche e la moda per le colonne non numeriche
        clus_values = pdf_filtered.mean(numeric_only=True).to_dict()
        non_numeric_columns = pdf_filtered.select_dtypes(exclude=["number"]).columns

        for column in non_numeric_columns:
            clus_values[column] = pdf_filtered[column].mode()[0]

        clusters_value.append(clus_values)

    return player_id_dict, clusters_value

def find_most_similar_index(d, dlist):
    categorical_keys = [key for key, value in d.items() if isinstance(value, str)]
    numeric_keys = [key for key in d if key not in categorical_keys]

    # Converti gli attributi categorici in codifica one-hot
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    categorical_values = encoder.fit_transform([[d[key]] for key in categorical_keys])

    # Costruisci i vettori includendo gli attributi numerici e la codifica one-hot
    vectors = []
    for d_dict in dlist:
        numeric_values = [d_dict.get(key, 0.0) for key in numeric_keys]
        cat_values = categorical_values.flatten().tolist()
        vector = numeric_values + cat_values
        vectors.append(vector)

    vector = [d.get(key, 0.0) for key in numeric_keys] + categorical_values.flatten().tolist()

    # Calcola la similarità del coseno tra i vettori
    similarities = cosine_similarity(vectors, [vector])
    most_similar_index = np.argmax(similarities)
    return most_similar_index

def get_clus(player_dict, clusters_list, min_value, max_value): # return the list of the clusters for the player of player_dict
    player_dict_c = copy.deepcopy(player_dict)
    clusters_list_c = copy.deepcopy(clusters_list)

    clus_list_first = [] # dictionary that have passed the first filter

    # first filter
    for i in range(len(clusters_list_c)):
        if clusters_list_c[i]["position"] == player_dict_c["position"] and clusters_list_c[i]["sub_position"] == player_dict_c["sub_position"] and clusters_list_c[i]["preferred_foot"] == player_dict_c["preferred_foot"]:
            height_diff = int(clusters_list_c[i]["height_cm"]) - int(player_dict_c["height_cm"])
            skills_diff = int(clusters_list_c[i]["skill_moves"]) - int(player_dict_c["skill_moves"])
            if math.sqrt(height_diff**2) < 20 and math.sqrt(skills_diff**2) <= 2 and int(clusters_list_c[i]["last_valuation"]) <= max_value and int(clusters_list_c[i]["last_valuation"]) >= min_value:
                clus_list_first.append((i,clusters_list_c[i]))

    # delete from clus_list_first and player_dict_c keys height_cm, skill_moves, last_valuation, position, sub_position, preferred_foot

    keys_todelete = ["height_cm", "skill_moves", "last_valuation", "position", "sub_position", "preferred_foot"]

    for clus in clus_list_first:
        for key in keys_todelete:
            del clus[1][key]
    #print(clus_list_first)

    # delete from player_dict_c keys height_cm, skill_moves, last_valuation, position, sub_position, preferred_foot
    for key in keys_todelete:
        del player_dict_c[key]

    # sort the player_dict_c keys based on the values
    sorted_keys_player_id = sorted(player_dict_c, key=player_dict_c.get, reverse=True)
    #print(sorted_keys_player_id)

    # sort the clusters based on the values
    sorted_keys_clusters = []
    for clus in clus_list_first:
        sorted_keys_clusters.append((clus[0], sorted(clus[1], key=clus[1].get, reverse=True)))
    #print(sorted_keys_clusters)

    # filter the filtered clusters based on the remaining stats value
    list_to_return = []
    for clus in sorted_keys_clusters:
        if len(list((set(sorted_keys_player_id[:10])) & (set(clus[1][:10])))) > 4:
            list_to_return.append(clus[0])

    return list_to_return

def run(df, player_id, min_value, max_value):

    #df pandas prova
    # carica csv file come pandas dataframe
    #df_prova = pd.read_csv(df)
    # show df_prova with show of pandasgui
    #pg.show(df_prova)

    existing_spark_session = SparkSession.getActiveSession()
    if existing_spark_session is not None:
        existing_spark_session.stop()

    # Create the session
    conf = SparkConf(). \
        set('spark.ui.port', "4050"). \
        set('spark.executor.memory', '15G'). \
        set('spark.driver.memory', '50G'). \
        set('spark.driver.maxResultSize', '40G'). \
        setAppName("PySparkProject"). \
        set('spark.executor.cores', "10"). \
        setMaster("local[*]")

    sc = pyspark.SparkContext.getOrCreate(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    sc._conf.getAll()


    df = spark.read.csv(df, header=True, inferSchema=True)

    # convert from double to int the features goalkeeping_speed, pace, shooting, passing, dribbling, defending, physic
    df = df.withColumn("goalkeeping_speed", df.goalkeeping_speed.cast("int"))
    df = df.withColumn("pace", df.pace.cast("int"))
    df = df.withColumn("shooting", df.shooting.cast("int"))
    df = df.withColumn("passing", df.passing.cast("int"))
    df = df.withColumn("dribbling", df.dribbling.cast("int"))
    df = df.withColumn("defending", df.defending.cast("int"))
    df = df.withColumn("physic", df.physic.cast("int"))
    df = df.withColumn("mentality_composure", df.mentality_composure.cast("int"))

    # drop column player_id
    df = df.drop("games_played_club", "games_won_club", "games_draw_club", "games_lost_club")

    # drop the features nationality_name and name
    dfc = df.drop("nationality_name", "name")

    # crea una copia della feature age in age_copy
    dfc = dfc.withColumn("age_copy", dfc.age)

    dfc = standardize(dfc)

    dfc = dfc.drop(*list(set(df.columns) - set(["player_id", "preferred_foot", "work_rate", "position", "sub_position", "age_copy"])))

    column_names = ["preferred_foot", "work_rate", "position", "sub_position"] #columns to apply one hot encoding (categorical features)
    dfc = apply_one_hot_encoding(dfc, column_names, "player_id") # apply one hot encoding to the specified columns on the df id player_id

    dfc1, dfc2 = split_dataframe(dfc)

    df1_pca = flag_PCA(dfc1,True,dfc)
    df2_pca = flag_PCA(dfc2,True,dfc)

    # def applying_elbow(df, features_col="pcaFeatures", prediction_col="cluster", min_k=60, max_k=100, random_seed=RANDOM_SEED):
    clustering_results1 = applying_elbow(df1_pca, features_col="pcaFeatures", prediction_col="cluster", min_k=50, max_k=51, random_seed=42)
    clustering_results2 = applying_elbow(df2_pca, features_col="pcaFeatures", prediction_col="cluster", min_k=50, max_k=51, random_seed=42)

    # obtain the key of the max items of clustering_results
    k_clustering1 = list(dict(sorted(clustering_results1.items(), key=lambda item: item[1], reverse=True)))[0]
    k_clustering2 = list(dict(sorted(clustering_results2.items(), key=lambda item: item[1], reverse=True)))[0]

    model1, df_clusters1 = k_means(df1_pca, k_clustering1, features_col="pcaFeatures", prediction_col="cluster", random_seed=42) # k_clustering or we can choose the number of clusters
    model2, df_clusters2 = k_means(df2_pca, k_clustering2, features_col="pcaFeatures", prediction_col="cluster", random_seed=42) # k_clustering or we can choose the number of clusters

    df_clusters1 = df_clusters1.withColumn("features", df_clusters1["pcaFeatures"]) # we need to do this because the function evaluate_k_means needs a column named "features"
    df_clusters2 = df_clusters2.withColumn("features", df_clusters2["pcaFeatures"]) # we need to do this because the function evaluate_k_means needs a column named "features"
    silhouette1 = evaluate_k_means(df_clusters1, metric_name="silhouette", distance_measure="squaredEuclidean", prediction_col="cluster")
    silhouette2 = evaluate_k_means(df_clusters2, metric_name="silhouette", distance_measure="squaredEuclidean", prediction_col="cluster")

    #join the clusters_df with the original df on the key player_id
    df_k_means1 = df.join(df_clusters1, "player_id")
    df_k_means2 = df.join(df_clusters2, "player_id")

    #oridinare df_k_means per cluster
    df_k_means1 = df_k_means1.orderBy("cluster")
    df_k_means2 = df_k_means2.orderBy("cluster")

    df_k_means1 = remove_columns(df_k_means1)
    df_k_means2 = remove_columns(df_k_means2)

###################### SCOUTING ######################

    player_id_dict, clusters_value = search_cluster(df_k_means1, df_k_means2, player_id)

###################### SCOUTING 1 ######################

    player_id_dict_copy = copy.deepcopy(player_id_dict)
    clusters_value_copy = copy.deepcopy(clusters_value)
    keys_to_remove = ["overall", "weak_foot", "pace", "shooting", "dribbling", "defending", "physic", "last_valuation"]

    for key in keys_to_remove:
        del player_id_dict_copy[key]

    for i in range(len(clusters_value_copy)):
        for key in keys_to_remove:
            del clusters_value_copy[i][key]

    if df_k_means2.filter(col("cluster") == 0).count() > 0:
        cluster_index = find_most_similar_index(player_id_dict_copy, clusters_value_copy)
    else:
        cluster_index = find_most_similar_index(player_id_dict_copy, clusters_value_copy) + 1

    result1 = df_k_means2.filter(col("cluster") == int(cluster_index))

###################### SCOUTING 2 ######################

    key_list = []
    for k,v in player_id_dict.items():
        key_list.append(k)

    # remove: overall, height_cm, skill_moves, pace, shooting, dribbling, defending, physic from key_list
    key_list = list(set(key_list) - set(["overall", "weak_foot", "pace", "shooting", "dribbling", "defending", "physic"]))

    player_id_dict_c = player_id_dict.copy()
    clusters_value_c = copy.deepcopy(clusters_value)
    keys_to_remove = ["overall", "weak_foot", "pace", "shooting", "dribbling", "defending", "physic"]

    for key in keys_to_remove:
        del player_id_dict_c[key]

    for i in range(len(clusters_value_c)):
        for key in keys_to_remove:
            del clusters_value_c[i][key]

    l = get_clus(player_id_dict_c, clusters_value_c, min_value, max_value)

    result2 = df_k_means2.filter(col("cluster") == l[0])

    #####################################################################
    # convert result1 and result2 to pandas dataframe
    result1 = result1.toPandas()
    result2 = result2.toPandas()
    pg.show(result1)
    pg.show(result2)
    



def run_button_click():
    # obtain the values inserted by the user
    param2 = int(entry2.get())
    param3 = int(entry3.get())
    param4 = int(entry4.get())

    # run the function run with the inserted parameters
    run("task2dataset.csv", param2, param3, param4)
    messagebox.showinfo("Done!", "Execution completed!")


def main():
    # Gui creation
    root = tk.Tk()
    root.title("Insert parameters")
    root.geometry("400x200")

    global entry2, entry3, entry4

    entry_font = ("Arial", 12) 

    # labels
    label2 = tk.Label(root, text="Player_id:", font=("Arial", 14))
    label2.pack()
    label3 = tk.Label(root, text="min price:", font=("Arial", 14))
    label3.pack()
    label4 = tk.Label(root, text="max price:", font=("Arial", 14))
    label4.pack()

    # text cell for inserting the parameters
    entry2 = tk.Entry(root, width=30, font=entry_font)
    entry2.pack()
    entry3 = tk.Entry(root, width=30, font=entry_font)
    entry3.pack()
    entry4 = tk.Entry(root, width=30, font=entry_font)
    entry4.pack()

    # button to run the function run
    button_font = ("Arial", 12)
    run_button = tk.Button(root, text="Search", command=run_button_click, width=20, height=2, font=button_font)
    run_button.pack()

    # GUI start
    root.mainloop()

if __name__ == '__main__':
    main()