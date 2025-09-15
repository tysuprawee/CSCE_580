# My name is Suprawe Pongpeeradech
## I chose the project: When Your Neighbor is a Zombie: Zombie KNN
Link: https://modelai.gettysburg.edu/2022/zombie/

**Authors:** Yim Register (University of Washington) and Dan Schneider (Code.org)

---

## Overview
This lesson introduces the K-Nearest Neighbor (KNN) algorithm to students in a post-apocalypse, zombie-survival context. Students apply the KNN algorithm to predict how many zombies are likely to be at a new unknown location by comparing the features of the unknown location to other known locations.

This lesson is entirely unplugged and designed to be introduced to students without any prior machine learning experience, and by teachers with minimal machine learning knowledge. This document contains a lesson plan, slides, and activity guides for students to engage in the lesson.

More information about KNN and this lesson can be found in the abstract and summary below.

---

## Resources
- Lesson Plan  
- Slides  
- Activity Guide – Zombie Prediction  
- Activity Guide – Numerical Accuracy  
- Zombie Map  
- KEY – Zombie Prediction  
- KEY – Numerical Accuracy  

---

## Abstract
A powerful, interpretable, and multipurpose AI algorithm is k-nearest neighbors (KNN). KNN capitalizes off of similarities between data points for either regression or classification. KNN is overwhelmingly taught using abstract data like red triangles and blue squares on a 2D plane. We present a completely unplugged KNN lesson that engages its K12 audience in a thrilling adventure to cross a map to safety during the zombie apocalypse.

Using a dataset that characterizes how many zombies congregate in certain areas (using features like noise level), students use the information to infer which areas on the map are likely to have fewer zombies and grant them safe passage. Students learn to make a prediction of how many zombies are likely in an area by tallying up the number of similarities between the new location and locations in their dataset. Next, they learn to take the mean number of zombies from those similar locations, and use this as a prediction. Finally, the lesson engages students with ideas about model performance in high-stakes scenarios.

While the zombie game is just for fun, we prompt students to also think about model accuracy for problems of distributing resources, gauging real world infection rates, or health decisions.

---

## Summary
We present a KNN lesson that engages its K12 audience in a thrilling adventure to cross a map to safety during the zombie apocalypse. Using a dataset that characterizes how many zombies congregate in certain areas (using features like noise level), students use the information to infer which areas on the map are likely to have fewer zombies and grant them safe passage. Students learn to make a prediction of how many zombies are likely to be in an area by tallying up the number of similarities between the new location and locations in their dataset.

---

## Topics
Introduction to AI, K-Nearest Neighbors, Accuracy, Data, Algorithm

---

## Audience
Grades 6–12. This assignment was designed with middle-school students in mind and requires students to take averages of numbers. However, it is appropriate as an introductory lesson for high school students too.

---

## Difficulty
No prior knowledge required; intended for beginners to AI concepts. The entire activity is designed to be completed within a 45 minute class period.

---

## Strengths
- Playful yet authentic introduction to the K-Nearest Neighbors algorithm in a way that transfers to more rigorous explorations of the algorithm  
- Lends itself to ethical discussions around accuracy when deciding the threshold for "good enough", and what the real-world consequences could be  
- Incorporates active learning strategies such as peer discussion and class debate  

---

## Weaknesses
- Zombies require prior pop culture knowledge or zombies may not be an appropriate entry point in some cultures – teachers may need to adapt the context of the activity depending on the students in the classroom  
- Doesn't explore how the value of K adjusts the results of the algorithm – this could be incorporated in a later lesson or extension to this activity  

---

## Dependencies
- Mathematical Knowledge: students should be familiar with how to take averages of numbers, even if they are still learning this in another class  
- There are no computational dependencies – this lesson is entirely unplugged and students do not need to use computers  

---

## Variants
Instructors can use this framework for examining machine learning algorithms to adapt other unplugged activities for students. Instructors can also extend this activity and continue to explore the Zombie scenario to explore other machine learning algorithms, or extend their understanding of KNN (such as using different values of K, or choosing different features in the data). Beyond this lesson, instructors may decide to explore online activities such as ml-playground to see computational versions of the KNN algorithm.

---

**Deliverables (in repo):**
- `slides/Slides.pdf` — teacher slides
- `activities/activity1.pdf` — Activity Guide: Zombie Prediction
- `activities/activity2.pdf` — Activity Guide: Numerical Accuracy
- `assets/zombie_map.png` — survival map
- `ProjectInfo.md` — project summary and links

**Learning outcomes (students will be able to):**
1. Use similarity (KNN) to make numerical predictions.
2. Compute accuracy for numerical predictions with tolerance thresholds (exact / ±5 / ±20).
3. Discuss ethical stakes of “good enough” accuracy.

**How to run (no computers):**
1. Print slides and both activity guides (1 per group).  
2. Print the map (1–2 per class).  
3. Model Location A (find top-3 similar rows → average).  
4. Students complete B & C, then compute accuracy; discuss thresholds.

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
