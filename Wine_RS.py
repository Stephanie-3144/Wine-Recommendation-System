import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.title('Wine Recommendation System')

#load data
data = pd.read_csv('https://raw.githubusercontent.com/Stephanie-3144/Wine-Recommendation-System/XWines_Test_1K_ratings.csv')


# check null
data.isnull().sum()


#change the dataset to a user-item-rating matrix
matrix = data.pivot_table(index = ['UserID'],columns = ['WineID'],values = ['Rating'],fill_value = 0)

#user-based filtering

#calculate similarity between users
user_similarity = cosine_similarity(matrix)

#reset index
matrix = matrix.reset_index() 

user_id = st.text_input('Please input your user id:')
if user_id:
    user_id = int(user_id)
    
#create a function to help find the similar users based cosine similarity score
def get_user_similarities(user_id, matrix, user_similarity, top_k): 
    try:
        # Get index in the similarity matrix using userid
        user_idx = matrix[matrix['UserID'] == user_id].index[0]
        # Get similarity list based on the index
        user_similarities = user_similarity[user_idx]
        # Avoid self index and sort list descendingly
        sorted_indices = np.argsort(-user_similarities)[1:]  # Get all but the user's own score
        
        # Initialize a list to hold the top k users and their similarity scores
        top_users_scores = []
        
        # Iterate over the sorted indices and retrieve the similarity scores
        for index in sorted_indices:
            if user_similarities[index] == 0:
                # Stop if a similarity score of 0 is encountered
                break
            top_users_scores.append((matrix['UserID'].iloc[index], user_similarities[index]))
            if len(top_users_scores) == top_k:
                # Stop if we've found the top k users
                break
        
        return top_users_scores
    except IndexError:
        print("User ID not found in DataFrame.")
        return None

# Test the function
similar_users = get_user_similarities(user_id, matrix, user_similarity, top_k=100)


def get_score_for_user_id(user_id, similar_users):
    """
    Retrieve the similarity score for a specific user_id from the list of similar users.

    Parameters:
    user_id (int): The user ID for which we want to find the similarity score.
    similar_users (list of tuples): A list where each tuple contains a user ID and the corresponding similarity score.

    Returns:
    float: The similarity score for the user_id, or None if not found.
    """
    for similar_user, score in similar_users:
        if similar_user == user_id:
            return score
    return None  # Return None if the user_id is not found in the list


# find unrated wines of the user
def find_unrated_wines(user_id, matrix):
    # get the user's rating data
    user_ratings = matrix[matrix['UserID']==user_id]
    # find the unrated wines
    unrated_wines = user_ratings.iloc[0][user_ratings.iloc[0] == 0]
    unrated_wines_indices = [wine_id for rating, wine_id in unrated_wines.index]
    return unrated_wines_indices

# test the function
unrated_items = find_unrated_wines(user_id, matrix)

def get_similar_users_ratings(unrated_items, similar_users, matrix):
    """
    This function generates a dictionary with unrated items as keys and a list of ratings from similar users as values.
    
    Parameters:
    unrated_items (list): List of item IDs that the user has not rated yet.
    similar_users (list): List of user IDs that are similar to our target user.
    matrix (DataFrame): The ratings matrix with MultiIndex columns ('Rating', item_id) and 'UserID' as index.
    
    Returns:
    dict: A dictionary with items as keys and lists of (user_id, rating) tuples as values.
    """
    
    similar_users_ratings = {}
    user_ids = [user[0] for user in similar_users]

    # Loop through each unrated item
    for item in unrated_items:
        # Create a list to store ratings for this item
        ratings_for_item = []
        
        # Loop through each similar user
        for user in user_ids:
            # Check if the score exists for this item
            if ('Rating', item) in matrix.columns:
                # Get the score for this user and item
                rating = matrix.loc[matrix['UserID'] == user, ('Rating', item)].iloc[0]
                # Assuming that a 0 represents no rating, filter for scores greater than 0
                if rating > 0:
                    ratings_for_item.append((user, rating))
        
        # If there are ratings for the item, store them in the dictionary with the item as key
        if ratings_for_item:
            similar_users_ratings[item] = ratings_for_item
    
    return similar_users_ratings

similar_users_ratings = get_similar_users_ratings(unrated_items, similar_users, matrix)



def predict_ratings(user_id, similar_users_ratings, matrix):
    # First, calculate the target user's average rating
    user_ratings = matrix.loc[matrix['UserID'] == user_id]
    user_avg_rating = user_ratings.drop('UserID', axis=1).replace(0, np.nan).mean(axis=1).iloc[0]

    # Initialize a dictionary to store the predicted ratings for each item
    predictions = {}

    # Calculate the predicted rating for each unrated item
    for item, ratings in similar_users_ratings.items():
        # Initialize the weighted sum of ratings and the sum of similarity weights
        weighted_sum = 0
        sum_of_weights = 0

        # Iterate over each similar user's rating
        for user, rating in ratings:
            # Calculate the average rating of the similar user
            neighbour_avg = matrix.loc[matrix['UserID'] == user].drop('UserID', axis=1).replace(0, np.nan).mean(axis=1).iloc[0]
            # Retrieve the similarity weight
            similarity = get_score_for_user_id(user, similar_users)
            if similarity is None:
                continue
            # Calculate the weighted deviation
            weighted_sum += similarity * (rating - neighbour_avg)
            sum_of_weights += abs(similarity)

        # Compute the predicted rating
        if sum_of_weights > 0:
            predicted_rating = user_avg_rating + (weighted_sum / sum_of_weights)
        else:
            # If there are no ratings from similar users, the predicted rating is set to the user's average rating
            predicted_rating = user_avg_rating
        
        # Store the predicted rating
        predictions[item] = predicted_rating

    return predictions

# Execute the prediction function
predicted_scores = predict_ratings(user_id, similar_users_ratings, matrix)
predictions = sorted(predicted_scores.items(), key = lambda item:item[1],reverse=True)
Wine_list=[]
for i in predictions:
    Wine_list.append(i[0])
st.write(f'Here is the wines list we recommend for you: {Wine_list}')
