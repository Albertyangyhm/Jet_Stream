# Jet_Stream
This repo is based on the work of: https://github.com/SebastianMacaluso/ginkgo

a few noteworthy things that were changed/added:

1. invMass_ginkgo_node.py is the new code for generating trees. invMass_ginkgo.py is the original. The original is included for reference and is not called anywhere.
2. The main recursive function is at invMass_ginkgo_node.py, line 213. I create nodes first, and then send the recursion down the created nodes.
3. A bfs function was included at invMass_ginkgo_node.py, line 31.
4. Use the jet_example.ipynb notebook and run everything until cell 5 and 6 to see new output.
