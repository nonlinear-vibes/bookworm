## Summary

| Query type | Metric | semantic | lexical | hybrid | agentic |
|---|---|---|---|---|---|
| Known item (n=7) | Hit@1 | 1.00 | 0.86 | 1.00 | 0.71 |
| Known item (n=7) | Hit@5 | 1.00 | 1.00 | 1.00 | 0.86 |
| Known item (n=7) | MRR@5 | 1.00 | 0.93 | 1.00 | 0.79 |
| Metadata (n=5) | P@5 | 0.44 | 0.56 | 0.52 | 0.76 |
| Metadata (n=5) | R@5 | 0.52 | 0.68 | 0.63 | 0.87 |
| Topic / mood (pooled) (n=5) | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |
| Topic / mood (pooled) (n=5) | R@5 |  |  |  |  |
| All queries | Mean latency (s) | 0.09 | 0.00 | 0.08 | 2.03 |

## Per query

| ID | Variant | Metric | semantic | lexical | hybrid | agentic |
|---|---|---|---|---|---|---|
| K1 | exact title + author | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| K2 | exact title | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| K3 | misremembered title | Hit@5 | 1.00 | 1.00 | 1.00 | 0.00 |
| K4 | plot description, no title | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| K5 | plot description, no title | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| K6 | plot description, no title | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| K7 | content description, no title | Hit@5 | 1.00 | 1.00 | 1.00 | 1.00 |
| M1 | author | P@5 | 0.60 | 1.00 | 0.80 | 1.00 |
| M2 | author | P@5 | 0.20 | 0.80 | 0.60 | 0.80 |
| M3 | author | P@5 | 0.60 | 0.60 | 0.60 | 0.60 |
| M4 | genre + year | P@5 | 0.60 | 0.20 | 0.40 | 1.00 |
| M5 | genre + year | P@5 | 0.20 | 0.20 | 0.20 | 0.40 |
| T1 | mood | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |
| T2 | mood / genre | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |
| T3 | topic | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |
| T4 | topic | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |
| T5 | topic | P@5 | 0.00 | 0.00 | 0.00 | 0.00 |

## Query Processor plans (agentic)

| ID | Keyword query | Semantic query | Filter |
|---|---|---|---|
| K1 | Camera Lucida |  | {'author': 'Roland Barthes'} |
| K2 | Frommer's South Africa |  | {'author': '', 'genre': ''} |
| K3 | Welding with Kids | A collection of short stories about unconventional settings or characters | {'genre': 'Short Stories'} |
| K4 | America's leading taxidermist, rural New York, 1917 | Novels about America's leading taxidermist and his estate in rural New York around 1917. | {'year_from': 1917} |
| K5 | Haitian Revolution Trilogy last book | The final book of a trilogy that explores the Haitian Revolution. | {'genre': 'Historical Fiction'} |
| K6 | Ouija board, teenage girl | Books about a teenage girl who uses a Ouija board to communicate with the other side | {'genre': 'Horror', 'year_from': 2000} |
| K7 | classic American roadside diner food recipe book | Recipe books featuring classic American roadside diner food | {'genre': 'Cookbook'} |
| M1 | Mercer Mayer |  | {'author': 'Mercer Mayer'} |
| M2 | Eric Jensen |  | {'author': 'Eric Jensen'} |
| M3 | Raymond Chandler |  | {'author': 'Raymond Chandler'} |
| M4 | cookbooks | Cookbooks with recipes and culinary techniques from various cuisines. | {'genre': 'Cookbook', 'year_to': 1999} |
| M5 | computer | Books about computers and their evolution during the 1990s. | {'year_from': 1990, 'year_to': 1999} |
| T1 |  | Cozy, uplifting, comforting books for a relaxing read. |  |
| T2 | dark psychological thriller | Dark, suspenseful, and thought-provoking psychological thrillers. | {'genre': 'Psychological thriller'} |
| T3 | parenting, child rearing, raising children | Books that provide advice, tips, and strategies for parents to raise their children effectively. | {'genre': 'Self-help', 'year_from': 2000} |
| T4 | Second World War | History books about the Second World War, providing insights into the events, causes, and consequences of the conflict. | {'genre': 'History'} |
| T5 | computer programming, beginner-friendly | Books for beginners to learn computer programming in an easy-to-understand manner. | {'genre': 'Computer Programming', 'year_from': 2010} |
