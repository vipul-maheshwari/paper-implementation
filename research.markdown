## Problems
1. How to decide which summarized node should be selected as the root and following the left subtree and the right subtree
- If we consider the cosine similarity between the current query and the rest of the summarized nodes and then create the trees, it would be different for each query and that's as same as what we did in the simple retrieval!
- We want to retrieve the relevant nodes or the relevant textual chunks based on a single tree which is being created earlier.
2. There should be a singlealone weighing criteria to choose which node should be selected as the root and it's children

3. If we want to dynamically create the tree for each query, it's going to be a hell lot of time consuming and therefore would be hectic too, I mean what's the point of creating the dynamic tree then, as we will be using some kind of semantic search algorithm like cosine similarity to check the similiarty first and then 


## Points to remember
1. Soft clustering is better compared to the hard one because it relatively helps to add textual nodes to more than one cluster node!

## Summarization notes:
1. Chossing a better summarization pipeline, currently setting to Falconsai/text_summarization for the starters!

## Clustering notes:
1. K-means clustering is not performing well enough considering the numeric embeddings into account.
 - There is no way we could get an optimal number for the number of clusters needed for accomdate all the textual chunks into the varied clusters.
 - It's too randomized in terms of choosing the optimal dataset point which then become a raw point as a cluster and a reference for the other dataset points
 - We might need to run the randomized iteration multiple times to get a optimal threshold setting for the clusters!
 - We would want as much as less variation between each cluster for the best setting of the K value, which is just a trial error approach!. Can't take the risk for the large Datasets!
 - Choosing a k value which gives the variations as low as possible is what we want! 

2. Agglomerative Clustering : 
 - Bottom up, so it start with each node being treated like a cluster first and based on the similarilty scores, we started clustering them in the tree structure!
 - Time complexity is is O(n^3) and space complexity is O(n^2)
 - The runtime for the general cases can be reduced to the O(n^2 log n) but it come at the cost of using more memory!
 - Heatmaps to analyze the overall cluster setting 

3. Sentence similiarity option
 - Clustering based on the most viable and similiar sentence availabe based on a certain threshold.

## Difficulties mitigation
1. GMM (Gausian mixture models) offers both the flexibility and a probablistic framework for clusering!