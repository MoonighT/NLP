import sys
sys.path.append("parsing/")
sys.path.append("util/")
import dependency_parser as depp
dp = depp.DependencyParser()
dp.read_data("english")

dp.train_perceptron(10)
dp.test()