# Github Pull Request Reviewer Recommendation

 Pull Request (PR) is used in github for the contribution of code from thousands of developers in millions of repository. PR reviewing is an essential activity in software development to maintain the quality of software projects. When a developer contributes to a project by submitting PR, the maintainer of the project needs to deal with it by taking all the opinions of reviewers into consideration and merging into the main branch. It is a time-consuming process especially when the project is large. The time between the submission of the PR and reviewing the PR can be reduced if we can assign new open PRs to appropriate reviewers. However, this feature is not available in Github. To this extent, we have purposed a reviewer recommendation system, which predicts the highly relevant reviewers for the forthcoming open PRs. Our method combines the topic modeling and social network analysis taking full advantage of the textual semantic of historical PRs and social relationships between developers. This project provides a python script that takes an input of the repository name and the new open PR ID to find the potential reviewers from the crowd of developers. It can reach up to 83\% of average accuracy for the top 5 recommended reviewers.

## Installation
The requirements.txt file includes most of the libraries used this project. However, it doesn't include common libraries. You might need to install it manually.
```bash
pip install -r .\requirements.txt
```

## How to use

### Running in CLI

```bash
python grr_cli.py --repo=[REPO] --opr=[OPEN_PULL_REQUEST_ID] --simthres=[0.2] --prlimit=[NUMBER_OF_CLOSED_PRS_TO_USE]
```

#### Arguments
* **--repo**: Name of repository to use (Repo with sufficient PRs preferable). If not provided, sveltejs/svelte is used as default repository.
* **--opr**: Open PR ID to get the recommendation. If not provided random open PR is used.
* **--simthres**: The threshold value to select % of closed PRs in the result. If not provided value of 0.2 is used.
* **--prlimit**: Number of closed PRs to use for topic modeling and social network analysis. If not provided all closed PRs in repository is used.

#### Example
```bash
python grr_cli.py

python grr_cli.py --repo=mrdoob/three.js

python grr_cli.py --repo=mrdoob/three.js --opr=19266

python grr_cli.py --repo=mrdoob/three.js --opr=19266 --simthres=0.3

python grr_cli.py --repo=mrdoob/three.js --opr=19266 --simthres=0.3 --prlimit=1000
```

### Notebook
 You can run the notebook step by step. Also, It provides graph visualization, plots, and results in each steps.

 ## Important Note
 It is highly recommended that you generate your own github token and edit on the scipts. Github has set a limit to the number of API requests per hour. You might experience this problem frequently when same token from the repository is used.

 ```python
ACCESS_TOKEN = <ACCESS_TOKEN_FROM_GITHUB>
 ```

 ## Poster
 ![alt text](./poster.svg "Project poster")
