# flower-poc
Proof of concept for training flower federated learning models in vantage6

## v6/flower proxy
One option for integrating flower with vantage6 is to create 2 proxies: one for the central node (superlink) and one for
the client nodes. The nodes would have to talk to eachother as follows.

Flow:
1. Central v6 node starts up central v6 algorithm including flower superlink, flower central proxy, and superexec central.
2. Algorithm sends request to run partial v6 algorithm on a different node including supernode, client proxy and superexec client.
3. Supernode starts up and  will try to reach out to superlink.
4. Instead of superlink it will find client proxy acting as superlink.
5. Response from client proxy should whatever makes that supernode happy.
6. Superexec will start, which means the federated model at the clients will be trained.
7. After one iteration, the weights will be sent as output of the v6 algorithm and client nodes will shut down.
8. Central node with central superexec will receive the weights and aggregate them.
9. Go back to step 2 until training is done.


If we want to integrate flower into vantage6 with the legacy v6 communication we would need to fake some of the communication between the client and supernode.
Moreover, we would need to go over all flower messages and determine if we can fake them this way. These messages are very
likely to change between versions so this approach is very brittle.