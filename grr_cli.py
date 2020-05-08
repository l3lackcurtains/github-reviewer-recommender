import argparse
import getopt
import itertools
import re
import string
import sys
from pprint import pprint

import gensim
import networkx as nx
import nltk
import numpy as np
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from github import Github
from networkx.algorithms import bipartite
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(2018)
nltk.download('wordnet', quiet=True)
stemmer = nltk.stem.porter.PorterStemmer()

REPO = 'mrdoob/three.js'
ACCESS_TOKEN = 'ae312d0463a722ea616aa73d11992d6475ad0271'
OPEN_PR_ID = 19266

###############################################################################
########### Data Preprocessing Helper Function  ###############################
###############################################################################
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(stemmer.stem(
                WordNetLemmatizer().lemmatize(token, pos='v')))
    return result

###############################################################################
########### LDA FUNCTION WITH COSINE SIMILARITY  ##############################
###############################################################################
'''
This function takes three parameters:
closed_prs_metaL List of documents generated from closed pull requests
closed_prs_corpus: List of document corpus generated from closed pull requests
open_pr_corpus: Document generated for open PR
'''
def lda_cosine_sim(closed_prs_meta, closed_prs_corpus, open_pr_corpus):
    
    corpus_data = []
    
    # preprocess each of documents and insert in corpus data
    for i, pr in enumerate(closed_prs_meta):
        preprocessed_data = preprocess(closed_prs_corpus[pr['id']])
        corpus_data.append(preprocessed_data)
    # Also, add preprocessed open PR document to end of corpus data
    corpus_data.append(preprocess(open_pr_corpus))
    
    # Map between normalized words and integer ID in dictionary
    dictionary = gensim.corpora.Dictionary(corpus_data)
    
    # FIlter the dictionary items
    dictionary.filter_extremes(no_below=15, no_above=0.8, keep_n=100000)
    
    # Convert the documents into bag of words format
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_data]
    
    # Apply TF-IDF in bag of words
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    # Apply LDA to TF_IDF corpus
    lda_model_tfidf = gensim.models.LdaMulticore(
        corpus_tfidf, num_topics=100, id2word=dictionary, passes=2, workers=4)
    
    # Compare the open PR and closed PRs documents corpus to get cosine similarity
    similarity_matrix = []
    for i in range(len(corpus_data) - 1):
        sim = gensim.matutils.cossim(
            lda_model_tfidf[bow_corpus][i], lda_model_tfidf[bow_corpus][len(corpus_data) - 1])
        similarity_matrix.append(sim)
    return similarity_matrix

###############################################################################
########### CUSTOM WEIGHT FUNCTION FOR PROJECTION #############################
###############################################################################
'''
This function calculate the weight from bipartite graph and transfer into
weights of projected graph
'''
def custom_weight(G, u, v, weight='weight'):
    weight_val = 0
    for nbr in set(G[u]) & set(G[v]):
        weight_val += (G[u][nbr]['weight'] + G[v][nbr]['weight'])*G.nodes[nbr]['similarity']
    return weight_val

###############################################################################
########### MAIN REVIEWER RECOMMENDATION FUNCTION #############################
###############################################################################
'''
This function takes the following parameters:
repo_name: Name of repository in github
access_token: Access token to use Github API
limit_pr: Value to limit number of closed PRs to process
open_pr_id: Open pull request ID. Typically found in github
pull requests section of repository
limit_recomm: Limit the number of recommendation
similarity_threshold: consine similarity tuning parameter
'''
def get_reviewer_recommendation(repo_name, access_token, open_pr_id=None, similarity_threshold=0.2, limit_pr=None, limit_recomm=5):

    # Get the access to Github API
    client = Github(access_token, per_page=300)

    print("[✔️] Connected to Github API.")

    # Get the repository object from Github API
    repo = client.get_repo(repo_name)

    # Get the maintainer of the repo
    repo_maintainer = repo.full_name.split("/")[0]

    # Get all the closed pull requests
    closed_prs = list(repo.get_pulls(state='closed'))
       
    if len(closed_prs) < 1:
        raise Exception("Insufficient number of closed pull requests. Use different repository.")

    # Limit number of pull requests if limit_pr is set
    if limit_pr != None:
        closed_prs = closed_prs[:limit_pr]

    print("[✔️] Parsed closed PRs.")

    # Initialize a graph
    graphz = nx.Graph()

    # It inserts all the reviewers node we add to graph
    closed_prs_reviewers = []

    # Save the data loaded from API for future use
    closed_prs_meta = []

    # Iterate through all the closed pull requests
    for pr in closed_prs:

        # If PR doesnt have comments continue with next
        if pr.get_issue_comments().totalCount == 0:
            continue

        # Get the user who submitted this PR
        pull_requester = pr.user.login

        # Get the PR number
        pr_number = 'PR #' + str(pr.number)

        ## Insert PR into graph node
        graphz.add_node(pr_number, type='Pull Request', bipartite=0)

        # Get all the comments of the PR
        comments = pr.get_issue_comments()

        # Get the meta data from PR and insert in closed_prs_meta
        pr_data = {}
        pr_data['id'] = pr_number
        pr_data['title'] = pr.title
        pr_data['body'] = pr.body
        pr_data['comments'] = comments
        closed_prs_meta.append(pr_data)

        # Iterate through all the comments
        for comment in comments:
            
            # Exclude user who are bots, maintainer, or PR submitter
            if comment.user != None and 'bot' not in comment.user.login and repo_maintainer != comment.user.login and pull_requester != comment.user.login:

                # Get the reviewer from comment
                reviewer = comment.user.login

                # Insert reviewer into graph node and closed_prs_reviewers list
                if reviewer not in closed_prs_reviewers:
                    closed_prs_reviewers.append(reviewer)
                    graphz.add_node(reviewer, type='user', bipartite=1)
                
                # If there is occurence of multiple comment, then add the occurence to the edge weight
                if graphz.has_edge(reviewer, pr_number):
                    new_weight = graphz.get_edge_data(
                        reviewer, pr_number)['weight'] + 1
                    graphz[reviewer][pr_number]['weight'] = new_weight
                else:
                    graphz.add_edge(reviewer, pr_number,
                                    weight=1, type='reviews')

    print("[✔️] Built a bipartite graph.")
    # Generate document corpus for closed pull requests
    closed_prs_corpus = {}
    for pr in closed_prs_meta:
        title = str(pr['title'])
        body = str(pr['body'])
        doc = title + " " + body
        for comment in pr['comments']:
            doc += comment.body
        # Remove the code, mentions and URLS
        doc = re.sub(r'\```[^```]*\```', '', doc)
        doc = re.sub(r"(?:\@|#|https?\://)\S+", "", doc)

        # insert document into corpus with index of corpus id
        closed_prs_corpus[pr['id']] = doc

    print("[✔️] Closed PRs corpus generated.")

    # Get the list of closed PRS
    open_prs = list(repo.get_pulls(state='open', sort='created'))
    if len(open_prs) == 0:
        raise Exception("Insufficient number of open pull requests. Use different repository.")
    
    # Get the first open PR
    open_pr = open_prs[0]

    # If Id is provided in function, choose this one
    if open_pr_id != None:
        for pr in open_prs:
            if open_pr_id == pr.number:
                open_pr = pr

    print("[✔️] Using PR ID #", open_pr.number, ".")

    # Get corpus document for open PR
    open_pr_corpus = str(open_pr.title) + "\n" + str(open_pr.body)
    for comment in open_pr.get_issue_comments():
        open_pr_corpus += comment.body

    # Remove the code, mentions and URLS
    open_pr_corpus = re.sub(r"\```[^```]*\```", "", open_pr_corpus)
    open_pr_corpus = re.sub(r"(?:\@|#|https?\://)\S+", "", open_pr_corpus)


    print("[✔️] Open PR corpus generated.")

    # Get the open PR submitter
    open_pr_requester = open_pr.user.login

    # Get the actual reviewers of open PR
    open_pr_reviewers = []
    for comment in open_pr.get_issue_comments():
        reviewer = comment.user.login
        # Exclude bot, maintainer and PR submitter
        if open_pr_requester != reviewer and reviewer not in open_pr_reviewers and 'bot' not in reviewer and repo_maintainer != reviewer:
            open_pr_reviewers.append(reviewer)

    # Remove the open PR reviewers that are not in our graph
    for open_pr_rv in open_pr_reviewers:
        if open_pr_rv not in closed_prs_reviewers:
            open_pr_reviewers.remove(open_pr_rv)

    # Get the similarity matrix between all the closed PRs and open PR
    similarity_matrix = lda_cosine_sim(closed_prs_meta, closed_prs_corpus, open_pr_corpus)

    print("[✔️] Calculated cosine similarity.")
    
    
    # Sort the similarity matrix in reverse order
    similarity_matrix = sorted(similarity_matrix, reverse=True)
    

    # Get the top similarity matrix by PR and filter with threshold
    top_similarity_matrix = {}
    for i, pr in enumerate(closed_prs_meta):
        top_similarity_matrix[pr['id']] = similarity_matrix[i]
        
    # Get top similarity matrix using similarity threshold value
    top_similarity_matrix = dict(itertools.islice(top_similarity_matrix.items(), int(len(top_similarity_matrix)*similarity_threshold)))

    print("[✔️] Selected top ", similarity_threshold*100, "% PRs using similarity threshold.")

    # Copy the bipartite graph into new one
    copied_barpartite_graphz = graphz.copy()

    # Get the top PR from similarity rank
    pr_nodes = []
    for similarity_id in top_similarity_matrix:
        pr_nodes.append(similarity_id)
    
    # Remove PR nodes other than top selected PR nodes
    for node in list(copied_barpartite_graphz.nodes):
        if 'PR #' in node and node not in pr_nodes and copied_barpartite_graphz.has_node(node):
            copied_barpartite_graphz.remove_node(node)
            
    # Insert similarity scores in PR nodes for further use in custom weight
    for node in copied_barpartite_graphz.nodes:
        if node in pr_nodes:
            copied_barpartite_graphz.nodes[node]['similarity'] = top_similarity_matrix[node]
    
    print("[✔️] Generated subgraph.")

    # Initialize a projected graph
    projected_graphz = nx.Graph()

    # Project the copied bipartate graph into reviewers graph considering the weights
    projected_graphz = bipartite.generic_weighted_projected_graph(
        copied_barpartite_graphz, closed_prs_reviewers, weight_function=custom_weight)

    # Remove isolatated nodes from the projected graph
    for node in list(nx.isolates(projected_graphz)):
        projected_graphz.remove_node(node)

    if len(projected_graphz.nodes) == 0:
        raise Exception("Use more similarity threshold.")

    print("[✔️] Subgraph projected into reviewer's graph.")
        
    # Run page rank algorithm in projected graph
    pagerank = nx.pagerank(projected_graphz, alpha=0.85, personalization=None, max_iter=100, tol=1e-06,
                           nstart=None, weight='weight', dangling=None)

    print("[✔️] Page rank calculated.")
    # Sort the page rank result by score
    pagerank = list(sorted(pagerank.items(), reverse=True, key=lambda x: x[1]))

    # Get only users from page rank result
    pagerank_reviewers = [pg[0] for pg in pagerank]
    
    # If there is recommendation limitation, limit it
    if limit_recomm != None:
        pagerank_reviewers = pagerank_reviewers[:limit_recomm]

    print("[✔️] Success.")
    # Print the current reviewers
    print("Current reviewers", open_pr_reviewers)
    
    # Print the recommended reviewers
    print("Recommended reviewers", pagerank_reviewers)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Command documendation')
    parser.add_argument("--repo", action='store', type=str, default="sveltejs/svelte", help="Name of the repository")
    parser.add_argument("--opr", action="store", type=int,
                        help="Open PR ID")
    parser.add_argument("--simthres", action="store", type=float, default=0.1,
                        help="Threshold value for selecting top similar PRs")
    parser.add_argument("--prlimit", action="store", type=int,
                        help="Limit the number of open PR to use.")

    args = parser.parse_args()
    repo = args.repo
    opr = args.opr
    simthres = args.simthres
    prlimit = args.prlimit

    print("########################################")
    print("Repo: sveltejs/svelte")
    if opr != None:
        print("Open PR ID: ", opr)
    else:
        print("Open PR ID: None (Using random PR ID)")
    print("Similarity threshold: ", simthres)
    if prlimit != None:
        print("Closed PR Limit: ", prlimit)
    else:
        print("Closed PR Limit: None (Using all PR available)")
    print("########################################")

    get_reviewer_recommendation(repo, ACCESS_TOKEN, open_pr_id=opr, similarity_threshold=simthres, limit_pr=prlimit)
