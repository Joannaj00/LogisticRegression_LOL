# Predicting Match Outcomes in League of Legends Using Logistic Regression

You can view the full project report [here](./project_report_template.html).

## Contents

- 1) Inspiration
- 2) Stakeholders
- 3) Task and Metrics
- 4) Data
- 5) PredictionLinear and Unregularized ModelNon-linear and Regularized ModelThe Interpretation of the Results
- Linear and Unregularized Model
- Non-linear and Regularized Model
- The Interpretation of the Results
- 6) InferenceEffect of Each Predictor on the Response (blueWins)ReliabilityVariation in the Response
- Effect of Each Predictor on the Response (blueWins)
- Reliability
- Variation in the Response
- 7) Recommendation to StakeholdersMain Action Items for StakeholdersLimitationsOvercoming Limitations
- Main Action Items for Stakeholders
- Limitations
- Overcoming Limitations
- 8) Conclusion
# Predicting Match Outcomes in League of Legends Using Logistic Regression

Project Report for STAT 303-2 Winter 2025

JOANNA JUNG

March 17, 2025

## 1) Inspiration

- I have enjoyed watching league of legend championships since Middle school and I recently started actually playing the game. I thought it would be fun to look into different components of league of legend and see how they affect the outcome.
- The dataset contains in-game statistics, making it ideal for building a predictive model.
- The insights gained from this model can help players, analysts, and esports teams make better decisions regarding team compositions and gameplay strategies.
## 2) Stakeholders

- Esports Coaches & Analysts – They can use the insights from this model to optimize game strategies. By understanding which in-game factors (e.g., gold difference, objective control, early kills) contribute most to winning, analysts can focus on the most impactful areas during training sessions.
- League of Legends Players – Competitive and ranked players can benefit from knowing which key statistics influence match outcomes. This information can help them prioritize objectives such as securing dragons or maintaining a gold lead.
- Game Developers – The insights from this project could be valuable for game balancing. If the model finds that certain features (e.g., early kills or dragons) have an overwhelming influence on victory, Riot Games could adjust gameplay mechanics to ensure more balanced matches.
## 3) Task and Metrics

This is a classification problem. Therefore, I will evaluate my models using multiple performance metrics:

- Accuracy Score: Measures overall correctness but can be misleading in imbalanced datasets.
- Precision and Recall: Helps evaluate how well the model predicts wins vs. losses.
- ROC-AUC Score: Evaluates the model’s ability to distinguish between wins and losses across different probability thresholds.
- Confusion Matrix: Provides insight into false positives and false negatives, which will help fine-tune model thresholds.
## 4) Data

- Data Source: The dataset is from Kaggle’s League of Legends dataset (https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).
- League of Legends is a multiplayer online battle by Riot Games.
- The dataset contains data on the first 10 minutes of solo queues from a high ELO (Diamond & Masters).
- The dataset was created by one person (Yi Lan Ma) and the expected update frequency is monthly.
How many observations and variables does the dataset have?

- There are 9879 observations and 40 variables
What does each observation and variable represent?

- Each observation represents a single game match snapshot at the 10-minute mark, with statistics for both the blue and red teams.
### Variables
- gameId: Unique identifier for each match.
- blueWins: Binary variable (1 if the blue team wins, 0 if the red team wins).
- blueWardsPlaced: Number of wards placed by the blue team in the first 10 minutes.
- blueWardsDestroyed: Number of enemy wards destroyed by the blue team.
- redWardsPlaced: Number of wards placed by the red team.
- redWardsDestroyed: Number of enemy wards destroyed by the red team.
- blueFirstBlood: 1 if the blue team got first blood (first kill of the game), 0 otherwise.
- blueKills: Number of kills secured by the blue team.
- blueDeaths: Number of times the blue team died.
- blueAssists: Number of assists made by the blue team.
- blueEliteMonsters: Number of elite monsters (Dragon or Rift Herald) killed by the blue team.
- blueDragons: Number of Dragons killed by the blue team.
- blueHeralds: Number of Rift Heralds killed by the blue team.
- redKills: Number of kills secured by the red team.
- redDeaths: Number of times the red team died.
- redAssists: Number of assists made by the red team.
- redEliteMonsters: Number of elite monsters (Dragon or Rift Herald) killed by the red team.
- redDragons: Number of Dragons killed by the red team.
- redHeralds: Number of Rift Heralds killed by the red team.
- blueTowersDestroyed: Number of towers destroyed by the blue team.
- blueTotalGold: Total gold accumulated by the blue team.
- blueAvgLevel: Average champion level of all blue team players.
- blueTotalExperience: Total experience accumulated by the blue team.
- blueTotalMinionsKilled: Total minions (lane creeps) killed by the blue team.
- blueTotalJungleMinionsKilled: Total jungle monsters killed by the blue team.
- blueGoldDiff: Gold difference between blue and red teams (positive if blue is ahead).
- blueExperienceDiff: Experience difference between blue and red teams.
- blueCSPerMin: Creep Score (CS) per minute for the blue team.
- blueGoldPerMin: Gold per minute for the blue team.
- redTowersDestroyed: Number of towers destroyed by the red team.
- redTotalGold: Total gold accumulated by the red team.
- redAvgLevel: Average champion level of all red team players.
- redTotalExperience: Total experience accumulated by the red team.
- redTotalMinionsKilled: Total minions (lane creeps) killed by the red team.
- redTotalJungleMinionsKilled: Total jungle monsters killed by the red team.
- redGoldDiff: Gold difference between red and blue teams (positive if red is ahead).
- redExperienceDiff: Experience difference between red and blue teams.
- redCSPerMin: Creep Score (CS) per minute for the red team.
- redGoldPerMin: Gold per minute for the red team.
How many variables are numeric and how many variables are categorical?

- There are 40 numeric variables and 0 categorical variables.
Which variables are the predictors and which variable is the response?

- Predictors: All variables related to the blue team except blueWins are predictors. These include game performance stats (kills, assists, gold, experience, objectives).We will discard the variables that start with “red” because we are focusing on the blue team.
- Response Variable: blueWins (indicating whether the blue team won).
Data cleaning

- Cleaning was not necessary because the dataset has no missing values and no categorical variables.
## 5) Prediction

### Linear and Unregularized Model

The first predictor used was “blueGoldDiff” because this variable had the highest correlation with “blueWins” with a correlation of 0.511. This is because gold is the primary economic resource in League of Legends, determining item purchases and power spikes.

Training and Test Performance:

After that, each predictor was added in a stepwise manner, and the training and test performance was recorded.

Key Takeaways:

- Adding gold-based predictors (blueTotalGold, blueGoldDiff) increased accuracy the most.
- Kills, deaths, and assists also improved performance, reinforcing that combat success is linked to winning.
- Wards placed/destroyed did not contribute much, suggesting that vision alone does not determine a game outcome.
### Non-linear and Regularized Model

To systematically introduce non-linearities, all interaction terms were created using PolynomialFeatures (degree=2). A Ridge Regularized Logistic Regression (L2 penalty) was applied to select the most informative non-linear terms.

- The non-linear model outperformed the linear model, particularly in ROC-AUC, showing it better distinguishes winning from losing games.
- Higher recall (0.7300) suggests fewer false negatives, meaning the model correctly predicts wins more often.
Key Takeaways for 5 Most Important Features:

- Tower destruction interactions are the most significant. This is because to win the game, players have to destroy the towers to get to the Nexus. Blue team also has to defend their towers so that the red team doesn’t destory their towers.
- If the blue team gets First Blood and the red team secures Rift Herald, it has an effect on the game’s direction. This is interesting because I didn’t think these two variables would affect each other.
- Warding and elite monster interactions have a negative impact, implying warding alone does not guarantee objective control.
Key Takeaways for 5 Least Important Features:

- The number of elite monsters taken by red team has almost no impact on experience difference.This suggests getting objectives like Dragon/Herald doesn’t always translate to an XP lead.
- This suggests getting objectives like Dragon/Herald doesn’t always translate to an XP lead.
- If both teams take Herald/Dragon/First blood, it has no impact on the model’s predictions.Likely means that trading cancels out its effect on game advantage.
- Likely means that trading cancels out its effect on game advantage.
Currently, the features include both the red team and blue team variables. I decided to try using only the blue team data to investigate if the coefficients of the features and test performance improve.

Here, we can see that the coefficients for the most important features increased compared to the previous model that included red team data.

- This is because when red team data was included, some features were competing for importance (e.g., redGoldDiff and blueGoldDiff both represent the same concept in opposite directions).
- Now that only blue team data is used, there is no redundancy, so the model assigns higher weights to the remaining important features.
- This means that the model is more confident in the importance of blueGoldDiff and its interactions, leading to larger coefficients.
Moreover, ROC-AUC of 0.8193 is nearly identical to the model that included both teams.

- This suggests that red team data was not providing additional useful information—likely because gold, experience, and objectives are already reflected in blue team stats.
### The Interpretation of the Results

Most Informative Predictors

- Gold-related features (blueGoldDiff, blueTotalGold, blueGoldPerMin) consistently showed the highest predictive power.blueGoldDiff achieved an accuracy of 0.7267 and an ROC-AUC of 0.8145.blueGoldPerMin performed similarly, with an accuracy of 0.7363 and ROC-AUC of 0.8186.Justification: Gold difference is a direct measure of a team’s economic strength, affecting item purchases and overall game control. Higher gold often correlates with better team fights and objective control.
- blueGoldDiff achieved an accuracy of 0.7267 and an ROC-AUC of 0.8145.
- blueGoldPerMin performed similarly, with an accuracy of 0.7363 and ROC-AUC of 0.8186.
- Justification: Gold difference is a direct measure of a team’s economic strength, affecting item purchases and overall game control. Higher gold often correlates with better team fights and objective control.
- Experience-based predictors (blueExperienceDiff, blueAvgLevel) also ranked high in importance.blueExperienceDiff had an accuracy of 0.7358 and an ROC-AUC of 0.8186.blueAvgLevel performed similarly with an accuracy of 0.7231 and ROC-AUC of 0.8039.Justification: Experience difference determines champion level, which translates to stronger abilities and better survivability. A higher experience lead can mean winning skirmishes and fights more easily.
- blueExperienceDiff had an accuracy of 0.7358 and an ROC-AUC of 0.8186.
- blueAvgLevel performed similarly with an accuracy of 0.7231 and ROC-AUC of 0.8039.
- Justification: Experience difference determines champion level, which translates to stronger abilities and better survivability. A higher experience lead can mean winning skirmishes and fights more easily.
- Objective control, particularly blueTowersDestroyed, played a crucial role.blueTowersDestroyed had an accuracy of 0.7110 and an ROC-AUC of 0.7906.Justification: Towers provide vision, map pressure, and strategic control, allowing a team to dominate rotations and jungle camps.
- blueTowersDestroyed had an accuracy of 0.7110 and an ROC-AUC of 0.7906.
- Justification: Towers provide vision, map pressure, and strategic control, allowing a team to dominate rotations and jungle camps.
Least Informative Predictors

- Warding statistics (blueWardsPlaced, blueWardsDestroyed) had minimal impact.blueWardsPlaced resulted in an accuracy of just 0.5025, slightly above random chance.Justification: While vision is important for strategy, placing wards alone does not directly lead to winning. It depends on how teams act on the vision gained.
- blueWardsPlaced resulted in an accuracy of just 0.5025, slightly above random chance.
- Justification: While vision is important for strategy, placing wards alone does not directly lead to winning. It depends on how teams act on the vision gained.
- First Blood (blueFirstBlood) was moderately useful but not as strong as economy-based stats.Accuracy: 0.6022 | ROC-AUC: 0.5976Justification: Getting the first kill provides an early advantage, but it does not guarantee sustained dominance throughout the game.
- Accuracy: 0.6022 | ROC-AUC: 0.5976
- Justification: Getting the first kill provides an early advantage, but it does not guarantee sustained dominance throughout the game.
## 6) Inference

In constructing our final model, I chose to use only blue team variables rather than incorporating both blue and red team statistics. This decision was based on the observation that blue team metrics alone provided sufficient predictive power without redundancy. When both teams’ data were included, many features were highly correlated with one another (e.g., blueGoldDiff and redGoldDiff are essentially the same but in opposite directions), leading to redundancy in the model without adding new information. By selecting only blue team variables, I aimed to create a more interpretable model while maintaining high accuracy.

To determine the most important predictors, I used absolute coefficient values from the full model, as well as their impact on performance metrics such as ROC-AUC, accuracy, and precision. The five selected predictors— blueGoldDiff, blueTotalJungleMinionsKilled, blueAvgLevel, blueCSPerMin, and blueTotalMinionsKilled— consistently showed the highest contribution to predicting game outcomes. These features directly influence gold accumulation, experience gains, and farming efficiency, which are the most reliable indicators of team strength.

### Effect of Each Predictor on the Response (blueWins)

- blueGoldDiff (Coefficient: 0.3064)For each 1,000 gold increase in blueGoldDiff, the log-odds of blueWins increase by 0.3064.The exponent of 0.3064 is around 1.36, meaning that for each additional 1,000 gold lead, the odds of the blue team winning increase by 36%.A higher gold difference translates into better itemization, stronger champions, and more map control, increasing the likelihood of victory.
- For each 1,000 gold increase in blueGoldDiff, the log-odds of blueWins increase by 0.3064.
- The exponent of 0.3064 is around 1.36, meaning that for each additional 1,000 gold lead, the odds of the blue team winning increase by 36%.
- A higher gold difference translates into better itemization, stronger champions, and more map control, increasing the likelihood of victory.
- blueTotalJungleMinionsKilled * blueGoldDiff (Coefficient: 0.1893)For each unit increase in blueTotalJungleMinionsKilled * blueGoldDiff, the log-odds of blueWins increase by 0.1893.The exponent of 0.1893 is around 1.21, meaning that for each additional jungle minion farmed while ahead in gold, the odds of winning increase by 21%.Jungle farming is important for maintaining a gold lead, and teams that farm the jungle efficiently while leading in gold have a significantly higher chance of winning.
- For each unit increase in blueTotalJungleMinionsKilled * blueGoldDiff, the log-odds of blueWins increase by 0.1893.
- The exponent of 0.1893 is around 1.21, meaning that for each additional jungle minion farmed while ahead in gold, the odds of winning increase by 21%.
- Jungle farming is important for maintaining a gold lead, and teams that farm the jungle efficiently while leading in gold have a significantly higher chance of winning.
- blueAvgLevel * blueGoldDiff (Coefficient: 0.1468)For each unit increase in blueAvgLevel * blueGoldDiff, the log-odds of blueWins increase by 0.1468.The exponent of 0.1468 is around 1.16, meaning that for each additional level gained while ahead in gold, the odds of winning increase by 16%.A higher level advantage combined with a gold lead significantly increases the blue team’s chances of winning, as champions with more experience deal more damage, have stronger abilities, and survive longer in fights.
- For each unit increase in blueAvgLevel * blueGoldDiff, the log-odds of blueWins increase by 0.1468.
- The exponent of 0.1468 is around 1.16, meaning that for each additional level gained while ahead in gold, the odds of winning increase by 16%.
- A higher level advantage combined with a gold lead significantly increases the blue team’s chances of winning, as champions with more experience deal more damage, have stronger abilities, and survive longer in fights.
- blueGoldDiff * blueCSPerMin (Coefficient: 0.1397)For each unit increase in blueGoldDiff * blueCSPerMin, the log-odds of blueWins increase by 0.1397.The exponent of 0.1397 is around 1.15, meaning that for each increase in CS per minute while ahead in gold, the odds of winning increase by 15%.Teams that continue farming efficiently after getting ahead in gold sustain their lead and make it harder for the enemy to recover, ensuring a higher win probability.
- For each unit increase in blueGoldDiff * blueCSPerMin, the log-odds of blueWins increase by 0.1397.
- The exponent of 0.1397 is around 1.15, meaning that for each increase in CS per minute while ahead in gold, the odds of winning increase by 15%.
- Teams that continue farming efficiently after getting ahead in gold sustain their lead and make it harder for the enemy to recover, ensuring a higher win probability.
- blueTotalMinionsKilled * blueGoldDiff (Coefficient: 0.1397)For each unit increase in blueTotalMinionsKilled * blueGoldDiff, the log-odds of blueWins increase by 0.1397.The exponent of 0.1397 is around 1.15, meaning that for each additional minion killed while ahead in gold, the odds of winning increase by 15%.Consistently farming minions increases a team’s gold lead, which directly contributes to their chances of winning.Minion kills alone aren’t enough—teams must translate that farm into meaningful economic and objective advantages.
- For each unit increase in blueTotalMinionsKilled * blueGoldDiff, the log-odds of blueWins increase by 0.1397.
- The exponent of 0.1397 is around 1.15, meaning that for each additional minion killed while ahead in gold, the odds of winning increase by 15%.
- Consistently farming minions increases a team’s gold lead, which directly contributes to their chances of winning.
- Minion kills alone aren’t enough—teams must translate that farm into meaningful economic and objective advantages.
### Reliability

To assess the reliability of each predictor, we need to evaluate whether the estimated coefficients are statistically significant.

From the summary, we can see that blueGoldDiff, blueTotalJungleMinionsKilled, blueAvgLevel, blueGoldDiff:blueTotalJungleMinionsKilled are statistically significant and reliably contributes to predicting wins because their p-values are below 0.05. blueGoldDiff:blueAvgLevel, blueGoldDiff:blueCSPerMin, blueGoldDiff:blueTotalMinionsKilled might not be significant, meaning its effect could be due to random chance. The p-values for blueCSPerMin and blueTotalMinionsKilled are NaN could be due to multicollinearity. To check this, I plan to compute VIF for each predictor

My Variance Inflation Factor (VIF) results indicate severe multicollinearity in my dataset. blueCSPerMin and blueTotalMinionsKilled are highly problematic—both their original and squared terms show infinite multicollinearity. Other interaction terms (e.g., blueGoldDiff * blueCSPerMin, blueTotalJungleMinionsKilled * blueTotalMinionsKilled) also have perfect collinearity. This is a problem because multicollinearity makes coefficient estimates unstable.

### Variation in the Response

Looking at the model summary, we can see that the Pseudo R-squared value is 0.222. This means the model explains 22.22% of the variation in blueWins compared to a null model that only includes an intercept, which indicates a moderately strong predictive ability. The model is significantly better than random guessing (R^2 = 0). However, there is still 77.78% of variation unexplained, which could be explained by adding more variables.

## 7) Recommendation to Stakeholders

### Main Action Items for Stakeholders

For esports coaches and analysts, the key takeaway from this analysis is that early-game advantages, particularly gold difference, jungle farming, and level leads, are crucial for setting up a win. Since blueGoldDiff was the strongest predictor of victory, teams should focus on securing a gold lead through efficient farming, early objective control, and kill conversions. Players can apply this by prioritizing dragons and towers and making sure their gold advantage leads to action rather than passive farming. Since this analysis only looks at the first 10 minutes of Diamond-ranked games, these findings are most relevant for high-level play, where small advantages can snowball quickly. For Riot Games, this data might highlight potential balance concerns—if gold or jungle control is too dominant, adjustments may be needed to keep matches competitive rather than one-sided from the start.

### Limitations

While the model offers valuable insights, it comes with some important limitations. The biggest one is that correlation doesn’t mean causation—just because a gold lead strongly correlates with winning doesn’t mean simply increasing gold guarantees a win. The model also lacks real-time decision-making context—it only looks at game stats, not the choices players make in response to their opponents. Things like team communication, rotations, and overall strategy are missing from this data. Another major gap is that the analysis doesn’t factor in champion matchups, player skill, or team compositions, all of which can completely change how a game plays out. There’s also the issue of scope—since this data only comes from Diamond-ranked games, the findings might not hold for lower-ranked players, where mistakes and playstyles are different, or for Challenger-level games, where coordination is much tighter. Finally, the model’s pseudo R-squared of 0.2222 tells us that while it captures some of what leads to a win, a lot of variation remains unexplained.

### Overcoming Limitations

To improve on this analysis, future work should bring in more contextual data—things like champion picks, team compositions, and player rankings could provide a clearer picture of how early-game decisions influence the rest of the match. It would also be useful to repeat this analysis across different ranks or game patches to see if the same trends hold, since strategies that work in Diamond might not be as effective in lower or higher tiers. Finally, expanding the dataset to track games beyond the first 10 minutes would give a better understanding of how early leads translate into mid and late-game wins, rather than just assuming a strong start always leads to victory.

## 8) Conclusion

My analysis shows that in Diamond-ranked League of Legends matches, early-game advantages in gold, jungle farming, and level leads play a major role in determining the outcome. Teams that build a gold lead while efficiently farming jungle camps and minions set themselves up for success, as these factors work together to create a snowball effect. That said, the model only captures part of the picture—it doesn’t account for champion matchups, team strategy, or what happens after the 10-minute mark, which are all crucial in shaping a game. To get a fuller understanding, future work should bring in more contextual data and explore more complex models to better capture the dynamics of high-level play.
