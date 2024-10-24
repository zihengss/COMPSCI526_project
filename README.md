# COMPSCI526 Project: Team 12
## Introduction
This project investigates the relationship between education and economic growth using data from the World Bank's World Development Indicators (WDI). The central focus is to explore how educational investments affect a country's GDP growth over time. By comparing various countries with different levels of educational development, we aim to uncover patterns and correlations that highlight the importance of education for sustained economic growth. Understanding the relationship between education and economic growth can serve as guidance for policymakers in data-driven decision making and facilitates scholars to forecast a certain country’s future economic condition based on current level of educational attainment. 

## [Dataset](https://datacatalog.worldbank.org/search/dataset/0037712/World-Development-Indicators)
The dataset used in this analysis is sourced from the World Bank's World Development Indicators (WDI). It includes a wide range of indicators related to GDP, education, and other socio-economic factors for countries over several decades. Key indicators include GDP per capita, school enrollment rates, and government expenditure on education.

Sample data:
| Country      | Indicator Name                              | Indicator Code | 2018     | 2019     | 2020     |
|--------------|---------------------------------------------|----------------|----------|----------|----------|
| United States| GDP (current US$)                           | NY.GDP.MKTP.CD | 20,580,223| 21,433,226| 20,936,600|
| China        | GDP (current US$)                           | NY.GDP.MKTP.CD | 13,894,745| 14,687,738| 14,722,740|
| India        | GDP (current US$)                           | NY.GDP.MKTP.CD | 2,869,746 | 2,869,706 | 2,660,240 |
| South Korea  | GDP (current US$)                           | NY.GDP.MKTP.CD | 1,619,423 | 1,631,055 | 1,647,977 |
| South Korea  | Literacy rate, adult total (% of people 15+) | SE.ADT.LITR.ZS | 97.90     | 97.92     | 97.93     |
| China        | Literacy rate, adult total (% of people 15+) | SE.ADT.LITR.ZS | 96.82     | 96.83     | 96.84     |
| India        | Literacy rate, adult total (% of people 15+) | SE.ADT.LITR.ZS | 74.00     | 74.25     | 74.37     |
| United States| School enrollment, primary (% net)           | SE.PRM.NENR    | 92.50     | 93.05     | 93.16     |
| China        | School enrollment, primary (% net)           | SE.PRM.NENR    | 94.10     | 94.40     | 94.52     |
| India        | School enrollment, primary (% net)           | SE.PRM.NENR    | 91.85     | 92.00     | 92.12     |
