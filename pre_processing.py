

import pyspark
from pyspark.shell import spark
from pyspark.sql.functions import *
from pyspark.sql.window import *

def create_final_table():

    # load the tables
    df_appearances, df_club_games, df_clubs, df_competitions, df_game_events, df_games, df_player_valuations, df_players = load_table()

    # join players and appearances
    df_players_appearances = df_players.join(df_appearances, ["player_id"], how='inner')

    # join players_appearances and club_games to extract information about the games played by the player
    df_players_appearances = df_players_appearances.join(df_club_games, "game_id", how='inner')

    # drop useless and duplicated features from df_players_appearances
    df_players_appearances = df_players_appearances.drop("current_club_id", "appearance_id",
                                                         "highest_market_value_in_eur", "current_club_name",
                                                         "city_of_birth", "market_value_in_eur",
                                                         "contract_expiration_date", "agent_name",
                                                         "current_club_domestic_competition_id", "image_url",
                                                         "last_season", "url", "player_current_club_id",
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
    df_last_year = df_v.filter(
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) <= month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) > month(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date)) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) > dayofmonth(df_v.date)) |
        (year(df_v.date_v) == year(df_v.date) + 1) & (month(df_v.date_v) == month(df_v.date)) & (dayofmonth(df_v.date_v) <= dayofmonth(df_v.date))
    ).dropDuplicates(["player_id", "date", "date_v"])

    print(df_last_year.count())

    # add the trend of the club when the player is playing
    df_trend = add_player_trend(df_last_year)

    # TODO risolvere problema out of memory

    # Group by the player_id and the valuation date and extract all the important features
    df_result = df_trend.groupBy(
        "player_id", col("market_value_in_eur").alias("market_value"), "date_v",
        col("current_club_id").alias("club_id"), col("height_in_cm").alias("height"),
        col("country_of_citizenship").alias("citizenship"), year("date_of_birth").alias("year_b"), "position",
        "sub_position") \
        .agg(collect_set("competition_id").alias("competition_id"),
             collect_set("player_club_id").alias("player_club_id"),
             count("date").alias("appearances2"),
             sum("assists").alias("assists"),
             sum("goals").alias("goals"),
             sum("minutes_played").alias("minutes_played"),
             sum("red_cards").alias("red_cards"),
             sum("yellow_cards").alias("yellow_cards"))

    print(df_result.count())

    #add last valuation in temporal terms
    df_result = df_result.withColumn("last_valuation", lag(df_result.market_value).over(Window.partitionBy("player_id").orderBy("date_v")))

    #df_check = df_result.filter(df_result.games_won + df_result.games_draw + df_result.games_lost != (
    #    (df_result.appearances+df_result.appearances2)/2))

    #print(df_check.count())

    # add the trend column
    df_final = add_club_trend(df_result, df_club_games, df_games)

    print(df_result.count())


def load_table():
    df_appearances = spark.read.csv("archive/appearances.csv", header=True, inferSchema=True)
    df_club_games = spark.read.csv("archive/club_games.csv", header=True, inferSchema=True)
    df_clubs = spark.read.csv("archive/clubs.csv", header=True, inferSchema=True)
    df_competitions = spark.read.csv("archive/competitions.csv", header=True, inferSchema=True)
    df_game_events = spark.read.csv("archive/game_events.csv", header=True, inferSchema=True)
    df_games = spark.read.csv("archive/games.csv", header=True, inferSchema=True)
    df_player_valuations = spark.read.csv("archive/player_valuations.csv", header=True, inferSchema=True)
    df_players = spark.read.csv("archive/players.csv", header=True, inferSchema=True)
    return df_appearances, df_club_games, df_clubs, df_competitions, df_game_events, df_games, df_player_valuations, df_players


def add_player_trend(df_filtered):
    df_f = df_filtered.alias("df_f")

    # add the is_draw to count the draws
    df_f = df_f.withColumn("is_draw", when(df_f.own_goals == df_f.opponent_goals, 1).otherwise(0))

    # add column games_played that counts the instances with key (player_id, date_v)
    df_f = df_f.withColumn("appearances", count(df_f.date_v).over(Window.partitionBy("date_v", "player_id")))

    # add column games_won that counts the instances with key (player_id, date_v) and is_win = 1
    df_f = df_f.withColumn("games_won_pl",
                                 count(when(df_f.is_win == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_draw_pl that counts the instances with key (player_id, date_v) and is_draw = 1
    df_f = df_f.withColumn("games_draw_pl",
                                 count(when(df_f.is_draw == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_lost_pl that counts the instances with key (player_id, date_v) and is_win = 0 and is_draw = 0
    df_f = df_f.withColumn("games_lost_pl", df_f.appearances - df_f.games_won_pl - df_f.games_draw_pl)

    # delete some columns
    df_f = df_f.drop("game_id", "club_id", "own_goals", "own_position", "own_manager_name", "opponent_id",
                           "opponent_goals", "opponent_position", "opponent_manager_name", "hosting", "is_win",
                           "is_draw")

    # for each (player_id, date_v) add a column with the result of (games_won_pl*3 + games_draw_pl)/games_played
    df_trend = df_f.withColumn("winning_rate_pl",
                                     (df_f.games_won_pl * 3 + df_f.games_draw_pl) / df_f.appearances)

    return df_trend

def add_club_trend(df_result, df_club_games, df_games):
    # join club_games and games to extract the date from games
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

    #add column games_played_club that counts the instances with key (player_id, date_v)
    df_join = df_join.withColumn("games_played_club", count(df_join.date_v).over(Window.partitionBy("date_v", "player_id")))

    # add column games_won_club that counts the instances with key (player_id, date_v) and is_win = 1
    df_join = df_join.withColumn("games_won_club", count(when(df_join.is_win == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_draw_club that counts the instances with key (player_id, date_v) and is_draw = 1
    df_join = df_join.withColumn("games_draw_club", count(when(df_join.is_draw == 1, 1)).over(Window.partitionBy("date_v", "player_id")))

    # add column games_lost_club that counts the instances with key (player_id, date_v) and is_win = 0 and is_draw = 0
    df_join = df_join.withColumn("games_lost_club", df_join.games_played_club - df_join.games_won_club - df_join.games_draw_club)

    #delete some columns
    df_join = df_join.drop("game_id", "club_id", "own_goals",  "own_position", "own_manager_name", "opponent_id", "opponent_goals", "opponent_position", "opponent_manager_name", "hosting", "is_win", "date", "is_draw")

    # delete duplicates
    df_join = df_join.dropDuplicates(["player_id", "date_v"])

    # for each (player_id, date_v) add a column with the result of (games_won_club*3 + games_draw_club)/games_played_club
    df_result = df_join.withColumn("winning_rate_club", (df_join.games_won_club*3 + df_join.games_draw_club)/df_join.games_played_club)

    return df_result