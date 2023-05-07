'''
import pyspark
from pyspark.shell import spark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

def create_final_table():
    df_appearances = spark.read.csv("archive/appearances.csv", header=True, inferSchema=True)
    df_club_games = spark.read.csv("archive/club_games.csv", header=True, inferSchema=True)
    df_clubs = spark.read.csv("archive/clubs.csv", header=True, inferSchema=True)
    df_competitions = spark.read.csv("archive/competitions.csv", header=True, inferSchema=True)
    df_game_events = spark.read.csv("archive/game_events.csv", header=True, inferSchema=True)
    df_games = spark.read.csv("archive/games.csv", header=True, inferSchema=True)
    df_player_valuations = spark.read.csv("archive/player_valuations.csv", header=True, inferSchema=True)
    df_players = spark.read.csv("archive/players.csv", header=True, inferSchema=True)

    # join players and appearances
    df_players_appearances = df_players.join(df_appearances, ["player_id"], how='inner')

    # drop useless and duplicated features from df_players_appearances
    df_players_appearances = df_players_appearances.drop("current_club_id", "appearance_id",
                                                         "highest_market_value_in_eur", "current_club_name",
                                                         "city_of_birth", "market_value_in_eur",
                                                         "contract_expiration_date", "agent_name",
                                                         "current_club_domestic_competition_id", "image_url",
                                                         "last_season", "url", "game_id", "player_current_club_id",
                                                         "first_name", "last_name", "player_name", "player_code")

    # drop useless and duplicated features from df_players_valuations
    df_player_valuations = df_player_valuations.drop("datetime", "dateweek")

    # rename the column date of df_players_valuations in date_v to avoid confusion with the date of df_players_appearances
    df_player_valuations = df_player_valuations.withColumnRenamed("date", "date_v")

    # Join the two dataframes on player_id
    df_valuations_appearances = df_player_valuations.join(df_players_appearances, "player_id")

    # TODO decide if we want to keep the players with no appearances, in case we have to do a union with valuations
    # adding before the zeroed column of df_players_appearances

    # Assign aliases to the tables
    df_v = df_valuations_appearances.alias("df_v")

    # we want to keep only the rows where the appearence date is within 1 year from the valuation date
    df_filtered = df_v.filter(
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) <= month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) > month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) > dayofmonth(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) <= dayofmonth(df_v.date))
    )

    # Group by the player_id and the valuation date and extract all the important features
    df_result = df_filtered.groupBy(
        "player_id", col("market_value_in_eur").alias("market_value"), "year_v", "month_v", "day_v", ################## CORREGGERE!! (date_v al posto di year_v, month_v, day_v) ##################
        col("current_club_id").alias("club_id"), col("height_in_cm").alias("height"),
        col("country_of_citizenship").alias("citizenship"), year("date_of_birth").alias("year_b"), "position",
        "sub_position") \
        .agg(collect_set("competition_id").alias("competition_id"),
             collect_set("player_club_id").alias("player_club_id"),
             count("date").alias("app"),
             sum("assists").alias("assists"),
             sum("goals").alias("goals"),
             sum("minutes_played").alias("minutes_played"),
             sum("red_cards").alias("red_cards"),
             sum("yellow_cards").alias("yellow_cards"))

    #add last valuation in temporal terms
    df_result = df_result.withColumn("last_valuation", lag(df_result.market_value).over(Window.partitionBy("player_id").orderBy("year_v", "month_v", "day_v"))) ################ CORREGGERE!! (date_v al posto di year_v, month_v, day_v) ##################

    #TODO add the club trend
'''

import pyspark
from pyspark.shell import spark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import *

def create_final_table():
    df_appearances = spark.read.csv("archive/appearances.csv", header=True, inferSchema=True)
    df_club_games = spark.read.csv("archive/club_games.csv", header=True, inferSchema=True)
    df_clubs = spark.read.csv("archive/clubs.csv", header=True, inferSchema=True)
    df_competitions = spark.read.csv("archive/competitions.csv", header=True, inferSchema=True)
    df_game_events = spark.read.csv("archive/game_events.csv", header=True, inferSchema=True)
    df_games = spark.read.csv("archive/games.csv", header=True, inferSchema=True)
    df_player_valuations = spark.read.csv("archive/player_valuations.csv", header=True, inferSchema=True)
    df_players = spark.read.csv("archive/players.csv", header=True, inferSchema=True)

    # join players and appearances
    df_players_appearances = df_players.join(df_appearances, ["player_id"], how='inner')

    # drop useless and duplicated features from df_players_appearances
    df_players_appearances = df_players_appearances.drop("current_club_id", "appearance_id",
                                                         "highest_market_value_in_eur", "current_club_name",
                                                         "city_of_birth", "market_value_in_eur",
                                                         "contract_expiration_date", "agent_name",
                                                         "current_club_domestic_competition_id", "image_url",
                                                         "last_season", "url", "game_id", "player_current_club_id",
                                                         "first_name", "last_name", "player_name", "player_code")

    # drop useless and duplicated features from df_players_valuations
    df_player_valuations = df_player_valuations.drop("datetime", "dateweek")

    # rename the column date of df_players_valuations in date_v to avoid confusion with the date of df_players_appearances
    df_player_valuations = df_player_valuations.withColumnRenamed("date", "date_v")

    # Join the two dataframes on player_id
    df_valuations_appearances = df_player_valuations.join(df_players_appearances, "player_id")

    # TODO decide if we want to keep the players with no appearances, in case we have to do a union with valuations
    # adding before the zeroed column of df_players_appearances

    # Assign aliases to the tables
    df_v = df_valuations_appearances.alias("df_v")

    # we want to keep only the rows where the appearence date is within 1 year from the valuation date
    df_filtered = df_v.filter(
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) <= month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) > month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) > dayofmonth(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) <= dayofmonth(df_v.date))
    )

    # Group by the player_id and the valuation date and extract all the important features
    df_result = df_filtered.groupBy(
        "player_id", col("market_value_in_eur").alias("market_value"), "date_v",
        col("current_club_id").alias("club_id"), col("height_in_cm").alias("height"),
        col("country_of_citizenship").alias("citizenship"), year("date_of_birth").alias("year_b"), "position",
        "sub_position") \
        .agg(collect_set("competition_id").alias("competition_id"),
             collect_set("player_club_id").alias("player_club_id"),
             count("date").alias("app"),
             sum("assists").alias("assists"),
             sum("goals").alias("goals"),
             sum("minutes_played").alias("minutes_played"),
             sum("red_cards").alias("red_cards"),
             sum("yellow_cards").alias("yellow_cards"))

    #add last valuation in temporal terms

    df_result = df_result.withColumn("last_valuation", lag(df_result.market_value).over(Window.partitionBy("player_id").orderBy("date_v")))

    ####### ADD THE CLUB TREND COLUMNS #######

    # join the useful tables
    df_club_games_join = df_club_games.join(df_games.select("game_id", "date"), "game_id", how='inner')

    # Expands the player_club_id list into separate columns
    df_result_expanded = df_result.withColumn("club_id", explode(col("player_club_id")))

    # Peform join based on club_id and apply condition on date
    df_join = df_result_expanded.join(df_club_games_join, ["club_id"]) \
        .where(expr("date <= date_v AND date >= date_v - INTERVAL 1 YEAR"))
    
    # Select columns
    df_join = df_join.select(df_result.columns + df_club_games_join.columns)

    # add column draw that is equal to 1 if own_goals = opponent_goals
    df_join = df_join.withColumn("is_draw", when(df_join.own_goals == df_join.opponent_goals, 1).otherwise(0))

    #add column games_played that counts the instances with key (player_id, date_v)
    df_join = df_join.withColumn("games_played", count(df_join.date_v).over(Window.partitionBy("date_v", "player_id")))

    # add column games_won that counts the instances with key (player_id, date_v) and is_win = 1
    df_join = df_join.withColumn("games_won", count(when(df_join.is_win == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_draw that counts the instances with key (player_id, date_v) and is_draw = 1
    df_join = df_join.withColumn("games_draw", count(when(df_join.is_draw == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_lost that counts the instances with key (player_id, date_v) and is_win = 0 and is_draw = 0
    df_join = df_join.withColumn("games_lost", df_join.games_played - df_join.games_won - df_join.games_draw)

    #delete some columns
    df_join = df_join.drop("game_id", "club_id", "own_goals",  "own_position", "own_manager_name", "opponent_id", "opponent_goals", "opponent_position", "opponent_manager_name", "hosting", "is_win", "date", "is_draw")

    # delete duplicates
    df_join = df_join.dropDuplicates(["player_id", "date_v"])

    # for each (player_id, date_v) add a column with the result of (games_won*3 + games_draw)/games_played
    df_result = df_join.withColumn("winning_rate", (df_join.games_won*3 + df_join.games_draw)/df_join.games_played)