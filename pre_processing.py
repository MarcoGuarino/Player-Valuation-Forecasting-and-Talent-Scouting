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
        "player_id", col("market_value_in_eur").alias("market_value"), "year_v", "month_v", "day_v",
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
    df_result = df_result.withColumn("last_valuation", lag(df_result.market_value).over(Window.partitionBy("player_id").orderBy("year_v", "month_v", "day_v")))

    #TODO add the club trend