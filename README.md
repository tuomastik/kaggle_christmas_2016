# Kaggle Christmas 2016: Santa's Uncertain Bags

## Task description
* Fill Santa's bags as full as possible
    * [https://www.kaggle.com/c/santas-uncertain-bags](https://www.kaggle.com/c/santas-uncertain-bags)
    * Stochastic optimization task
    * Multiple knapsack problem
* Resources
    * 1000 bags
    * 9 types of gifts
* Constraints
    * Each bag must have >=3 gifts
    * 1 bag can contain <= 50 pounds of gifts
    * No gift may be used more than once
* Theoretical best: 50 pounds in 1000 bags --> 50000 pounds

## Python environment
    conda create -n kaggle_christmas_2016 python=3.5 numpy pandas seaborn matplotlib
