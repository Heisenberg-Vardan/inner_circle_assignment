# Python Engineer Case Assignment

Thank you for your interest in the Python Engineer role!
This technical case is designed to evaluate your ability to work with data pipelines, build a basic recommender system, and write clean, testable, and well-structured Python code.

---

## Overview

This case assignment is divided into **two parts**:

### Recommender System (Modeling & Algorithms)

You are given two datasets:

* **User data**: Contains metadata about users (e.g., user ID, gender, location, etc.)
* **Interaction data**: Contains interactions between users (e.g., user ids for the users that interact with each other, interaction type(likes, dislikes), timestamps)

Your task is to **build a recommender system** that outputs the **top k most relevant users** whom a given user might want to interact **positively** with, based on the interaction data.

You can use any approach: rule-based, collaborative filtering, content-based etc.
#### Requirements

  * Clear explanation of your approach (README)
  * Python code to build/train the recommender system and calculate recommendations
  * Example usage of your model
  * Any assumptions or limitations
  * Optional: Include evaluation metrics or visualizations if helpful.

### Data Aggregation (Data Engineering & SQL Modeling)

Using the same data:

* Load the data into a **SQL database** (e.g., SQLite, Postgres)
* Write a SQL model to generate an aggregate table with the following metrics:
  
  * **Percentage of likes**
  * **Percentage of dislikes**
  * **Total number of interactions**
* The above should be aggregated **per**:

  * `day` (based on `timestamp`)
  * `gender` of users that have sent a like or dislike
  * `city` of users that have sent a like or dislike

#### Requirements

  * Clear explanation of your approach (README)
  * Python code that loads the data into the database
  * SQL queries that produce the required metrics
  * Example usage of your code to create SQL tables and run the queries
  * Optional: Output example showing the aggregation

---

## Dataset Description

You’ll receive two CSV files:

### `users.csv`

| user\_id | first_name | last_name | gender | age | city   | email                      | about_me |
|----------|------------|-----------|--------|-----|--------|----------------------------|----------|
| 123      | Derrick    | Kirby     | male   | 29  | London | cunninghamtammy@yahoo.com  | I'm a passionate and dedicated individual who has always been interested in connecting with people who share my values. |
| 456      | Michelle   | Cummings  | female | 29  | London | kennedyjennifer@hotmail.com| I'm a passionate, independent woman with an insatiable curiosity about the world and people. |

### `interactions.csv`

| user\_id      | liked\_user\_id | timestamp           | like\_type |
|----------------|-----------------|---------------------|------------|
| 123            | 456             | 2025-06-01 10:00:00 | 0          |
| 456            | 123             | 2025-06-01 11:00:00 | 1          |

Here `like_type` can have 3 values:

- 0: Dislike (Negative), user has disliked other user
- 1: Like (Positive), user has liked other user
- 2: Match (Positive), user has liked other user which resulted in a match (both users have liked each other)
---

## Submission Instructions

Please share a GitHub repository that includes:

* Your code
* A README with:
  * Your approach to tasks 
  * How to set up and run your code locally for each task
  * Your thought process and modeling decisions
  * Any assumptions or limitations you encountered
  * Optional: any evaluation metrics or visualizations you created
* Invite sergey@theinnercircle.com and bjarnthor@theinnercircle.com to the repository so we can review your work.

We're excited to see how you approach this problem. Please feel free to be creative, pragmatic, and thoughtful in your solution.
