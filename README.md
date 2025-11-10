# flower-poc
Proof of concept for training flower federated learning models in vantage6

## Why integrate flower with vantage6
Vantage6 is a framework for federated analysis that focuses on orchestration and administration. It is very flexible in terms
of analyses because and can deal with both vertically- and horizontally-partitioned data. However, vantage6 does not provide
any federated learning algorithms out of the box. Flower is a framework for federated learning that focuses applying deep
learning models in a federated setting. However, flower lacks the administration, permission systems and UI that vantage6
brings. By integrating flower with vantage6 we can get the best of both worlds.

## Things to consider
The main challenge with integrating flower with vantage6 is that both frameworks have their own communication protocols.
Vantage6's legacy communication uses its own protocol based on REST and websockets, while flower has its own protocol based on gRPC.
If we would integrate the full flower framework into a vantage6 algorithm we would somehow make the flower communication
go through vantage6's channels. We are considering a number of ways to do this.

### node-to-node communication
The most flexible way integrating flower or any other federated learning framework into vantage6 would be to allow direct
node-to-node communication. In the legacy vantage6 communication model, all communication goes through the central server
via  REST api. This means there will always have to be some translation step between the vantage6 communication and the
FL framework. To solve this issue, node-to-node communication has been implemented in vantage6 version 3. Unfortunately,
this method turned out to be quite complex to implement at the nodes and has not been widely adopted. The architecture of
vantage6 has been changed considerably in version 5 and the n2n communication has been removed.

However, the new kubernetes-based architecture opens up new possibilities for a more stable version of n2n communication.
lstio has been considered as a potential candidate for implementing n2n communication in vantage6. However, this would
require a considerable  amount of work. 

### v6 algorithm with proxy
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

### Deconstructed flower
Accommodating the flower communication within the current version of vantage6 is quite complex. It might be a better idea
to extract the essential parts of flower that are needed to get flower apps running, and run them in the vantage6 way.

One of the main components of flower is the strategy. The strategy defines how a model is trained and aggregated.
The strategy is implemented as an abstract base class [`Strategy`](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy)
that has been extended to implement various strategies such as [`FedAvg`](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html#flwr.serverapp.strategy.FedAvg).

It might be possible to subclass the `Strategy` class and override the `start` method to implement the vantage6 communication protocol.
We would have to look into what other components need to be adapted to make this work. Ideally we would be able to reuse
existing flower server and client apps within vantage6.
