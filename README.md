FairRec
This code is the implementaion of the paper "F𝑎𝑖𝑟𝑅𝑒𝑐: On the fair and topic aware influence maximization in online social
networks"

A toy graph is being provided which is fully anonymized, This toy graph is the influence propagation graph and it is made from meetup dataset. The toy graph is in the form of dictionary and store in pickle file

Run `python FairRec.py number_of_influencial_users number_of_influence_tags' to run FairRec and get the output as expected influence spread, top-k influencial users, top-r influence tags and L1 norm.
Example : python FairRec.py 5 3' to run FairRec and get the output as expected influence spread, top-5 influencial users, top-3 influence tags and L1 norm


Only Run 'python FairRec.py' to run the code with k(number of influencial user), r(number of influence tags) as 5 and 3.
