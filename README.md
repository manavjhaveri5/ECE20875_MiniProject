Mini Project
Due: 28th April, 2024 at 11:59pm ET
Note: A crucial aspect of this project involves submitting a part of the project to GRADESCOPE.
Up until now, we have given you fairly detailed instructions for how to design data analyses to answer specific questions about data -- in particular, how to set up a particular analysis and what steps to take to run it. In this project, you will put that knowledge to use!

Put yourself in the shoes of a data scientist being given a data set and asked to draw conclusions from it. Your job will be to understand what the data is showing you, design the analyses you need, justify those choices, draw conclusions from running the analyses, and explain why they do (or do not) make sense.

We are deliberately not giving you detailed directions on how to solve these problems, but feel free to come to office hours to brainstorm.

Objectives
There are three possible paths through this project:

You may use dataset (path) #1, which captures information about student behavior and performance in an online course. See below for the analysis questions we want you to answer.
You may use dataset (path) #2, which captures information about bike usage in New York City. See below for the analysis questions we want you to answer.
You may use dataset (path) #3, which are images of digits. See below for the analysis questions we want you to answer.
Partners
On this project you are allowed to work with one partner (from the same section). Working with a partner is optional, and working with a partner will not impact how the project is graded. If you want to work with a partner, it is your responsibility to pair up; feel free to use Piazza's "Search for Teammates" feature (https://piazza.com/class/lr0itzy6jto1lr/post/5) to facilitate this. Note that you can only team-up with a person from the same section.

It is highly essential to note that, if you are planning to work with a member, then only the team lead, i.e., one of the two members (decided among yourselves), needs to submit the completed code (on GitHub) and the PDF report (on GitHub and Gradescope). The points on this project obtained by the team leader will be replicated on to its team-mate. To make sure the scores match among the teammates we request you to utilize Gradescope's group submission feature (https://youtu.be/rue7p_kATLA).

The team leader will be in charge of pushing the last version of the project to his/her repository.

If you decide to work solo, then you are your group's leader.

Path 1: Student performance related to video-watching behavior
behavior_performance.txt contains data for an online course on how students watched videos (e.g., how much time they spent watching, how often they paused the video, etc.) and how they performed on in-video quizzes. readme-behavior_performance.pdf details the information contained in the data fields. There might be some extra data fields present than the ones mentioned here. Feel free to ignore/include them in your analysis. In this path, the analysis questions we would like you to answer are as follows: You will run prediction algorithm(s) for ALL students for ONE video, and repeat this process for all videos.

How well can the students be naturally grouped or clustered by their video-watching behavior (fracSpent, fracComp, fracPaused, numPauses, avgPBR, numRWs, and numFFs)? You should use all students that complete at least five of the videos in your analysis. Hints: KMeans or distribution parameters(mean and standard deviation) of Gaussians
Can student's video-watching behavior be used to predict a student's performance (i.e., average score s across all quizzes)?
Taking this a step further, how well can you predict a student's performance on a particular in-video quiz question (i.e., whether they will be correct or incorrect) based on their video-watching behaviors while watching the corresponding video? You should use all student-video pairs in your analysis.

You must turn in two sets of files, by pushing them to your team leader's Github repository. For the report PDF, it has to be submitted to Gradescope as well:

report.pdf: A project report, which should consist of:

A section with the names of the team members (maximum of two), your Purdue username(s), the path (1 or 2 or 3) you have taken, and most importantly the your mini-project repository's github link.
A section describing the dataset you are working with.
A section describing the analyses you chose to use for each analysis question (with a paragraph or two justifying why you chose that analysis and what you expect the analysis to tell you).
A section (or more) describing the results of each analysis, and what your answers to the questions are based on your results. Visual aids are helpful here to back up your conclusions. Note that, it is OK if you do not get "positive" answers from your analysis, but you must explain why that might be.
All Python .py code files you wrote to complete the analysis steps
